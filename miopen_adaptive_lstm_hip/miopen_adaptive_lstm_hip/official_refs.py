from __future__ import annotations

OFFICIAL_REFERENCE_MAP = {
    "MIOpen selector": "projects/miopen/src/rnn/selector.cpp",
    "MIOpen public RNN dispatch": "projects/miopen/src/rnn.cpp",
    "MIOpen hidden update launch": "projects/miopen/src/rnn/rnn_util.cpp",
    "MIOpen forward modular algo": "projects/miopen/src/rnn/Solutions/Base/fw_data_modular.cpp",
    "MIOpen backward modular algo": "projects/miopen/src/rnn/Solutions/Base/bw_data_modular.cpp",
    "MIOpen hidden update kernel": "projects/miopen/src/kernels/MIOpenRNNHiddenStateUpdate.cpp",
    "CK gridwise XDLops": "include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r3.hpp",
    "CK gridwise pipeline v1": "include/ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v1.hpp",
    "CK gridwise pipeline v2": "include/ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v2.hpp",
    "CK blockwise XDLops pipeline": "include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops.hpp",
    "CK XDLops utility": "include/ck/utility/amd_xdlops.hpp",
    "CK tile FMHA pipeline": "include/ck_tile/ops/fmha/pipeline/",
    "CK tile FMHA example": "example/ck_tile/01_fmha/",
}

