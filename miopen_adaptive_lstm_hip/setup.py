from __future__ import annotations

import os
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent

# K100_AI = gfx928
ROCM_ARCH = os.environ.get("PYTORCH_ROCM_ARCH", "gfx928").strip()

CXX_ARGS = ["-O3", "-std=c++17"]
NVCC_ARGS = ["-O3", "-std=c++17"]
EXTRA_LINK_ARGS = []

if torch.version.hip is not None:
    os.environ["PYTORCH_ROCM_ARCH"] = ROCM_ARCH
    os.environ["AMDGPU_TARGETS"] = ROCM_ARCH
    os.environ["HCC_AMDGPU_TARGET"] = ROCM_ARCH

    NVCC_ARGS += [
        f"--offload-arch={ROCM_ARCH}",
        "--gpu-max-threads-per-block=256",

        "-DMIOPEN_ADAPTIVE_LSTM_ENABLE_DIRECT_BLAS=1",
        "-DMIOPEN_ADAPTIVE_LSTM_ENABLE_MFMA_BUILTIN=1",

        "-Wno-return-type",
        "-Wno-unused-command-line-argument",
    ]

    EXTRA_LINK_ARGS.append("-lhipblas")

print("ROCM_ARCH =", ROCM_ARCH)
print("NVCC_ARGS =", NVCC_ARGS)


setup(
    name="miopen_adaptive_lstm_hip",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            "miopen_adaptive_lstm_hip._C",
            [
                str(ROOT / "csrc" / "bindings.cpp"),
                str(ROOT / "csrc" / "adaptive_lstm_hip.cu"),
            ],
            extra_compile_args={
                "cxx": CXX_ARGS,
                "nvcc": NVCC_ARGS,
            },
            extra_link_args=EXTRA_LINK_ARGS,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
