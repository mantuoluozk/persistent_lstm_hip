from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PipelineVersion = Literal[
    "scalar",
    "partitioned",
    "cached_b2",
    "cached_b4",
    "cached_b8",
    "gemm_scan",
    "xdlops_v1",
    "xdlops_v2",
]


@dataclass(frozen=True)
class HiddenUpdateLaunch:
    hidden_size: int
    max_batch: int
    max_active_threads: int
    total_work: int
    read_block: int
    total_items: int
    items_per_group: int
    workgroup_count: int
    global_size: int


@dataclass(frozen=True)
class CKGemmTileTraits:
    """A compact CK-style trait record for future recurrent GEMM kernels."""

    pipeline: PipelineVersion
    block_size: int
    m_per_block: int
    n_per_block: int
    k_per_block: int
    num_k_prefetch_stage: int
    use_lds_for_a: bool
    use_lds_for_b: bool
    wave_tile_m: int
    wave_tile_n: int
    wave_tile_k: int
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecurrentKernelPlan:
    name: str
    hidden_size: int
    partitions: int
    batch_tile: int
    pipeline: PipelineVersion
    tile_traits: CKGemmTileTraits


@dataclass(frozen=True)
class HiddenPointwiseTraits:
    read_block: int
    items_per_group: int
    global_size: int
    vectorized: bool


def choose_hidden_update_launch(
    *,
    max_active_threads: int,
    max_batch: int,
    hidden_size: int,
) -> HiddenUpdateLaunch:
    """Mirror MIOpen `LSTMForwardHiddenStateUpdate()` launch sizing."""

    if max_active_threads <= 0:
        raise ValueError("max_active_threads must be positive")
    if max_batch <= 0:
        raise ValueError("max_batch must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")

    total_work = max_batch * hidden_size
    if total_work >= 4 * max_active_threads and hidden_size % 4 == 0:
        read_block = 4
    elif total_work >= 2 * max_active_threads and hidden_size % 2 == 0:
        read_block = 2
    else:
        read_block = 1

    total_items = max(total_work // read_block, 1)
    if total_items <= 64:
        items_per_group = 64
    elif total_items <= 128:
        items_per_group = 128
    else:
        items_per_group = 256

    global_size = min(total_items, max_active_threads)
    workgroup_count = (global_size + items_per_group - 1) // items_per_group
    global_size = workgroup_count * items_per_group

    return HiddenUpdateLaunch(
        hidden_size=hidden_size,
        max_batch=max_batch,
        max_active_threads=max_active_threads,
        total_work=total_work,
        read_block=read_block,
        total_items=total_items,
        items_per_group=items_per_group,
        workgroup_count=workgroup_count,
        global_size=global_size,
    )


def choose_recurrent_kernel_plan(hidden_size: int, batch_size: int) -> RecurrentKernelPlan:
    """Select a CK/MIOpen-inspired recurrent kernel family."""

    if hidden_size == 128:
        traits = CKGemmTileTraits(
            pipeline="gemm_scan",
            block_size=512,
            m_per_block=4,
            n_per_block=128,
            k_per_block=32,
            num_k_prefetch_stage=1,
            use_lds_for_a=True,
            use_lds_for_b=False,
            wave_tile_m=1,
            wave_tile_n=32,
            wave_tile_k=32,
            notes=(
                "MIOpen-style hidden update launch",
                "CK-style weight-resident recurrent pipeline",
            ),
        )
        return RecurrentKernelPlan(
            name="h128_gemm_scan",
            hidden_size=hidden_size,
            partitions=4,
            batch_tile=4,
            pipeline="gemm_scan",
            tile_traits=traits,
        )

    if hidden_size % 64 == 0 and hidden_size >= 256:
        traits = CKGemmTileTraits(
            pipeline="xdlops_v1",
            block_size=1024,
            m_per_block=1,
            n_per_block=min(hidden_size, 256),
            k_per_block=64,
            num_k_prefetch_stage=2,
            use_lds_for_a=True,
            use_lds_for_b=True,
            wave_tile_m=1,
            wave_tile_n=64,
            wave_tile_k=32,
            notes=("future MFMA/XDLops recurrent tile",),
        )
        return RecurrentKernelPlan(
            name="xdlops_recurrent_candidate",
            hidden_size=hidden_size,
            partitions=8,
            batch_tile=1,
            pipeline="xdlops_v1",
            tile_traits=traits,
        )

    traits = CKGemmTileTraits(
        pipeline="partitioned",
        block_size=256,
        m_per_block=1,
        n_per_block=hidden_size,
        k_per_block=32,
        num_k_prefetch_stage=1,
        use_lds_for_a=True,
        use_lds_for_b=False,
        wave_tile_m=1,
        wave_tile_n=32,
        wave_tile_k=32,
        notes=("portable partitioned fallback",),
    )
    return RecurrentKernelPlan(
        name="partitioned_hidden_update",
        hidden_size=hidden_size,
        partitions=4 if hidden_size < 256 else 8,
        batch_tile=1,
        pipeline="partitioned",
        tile_traits=traits,
    )
