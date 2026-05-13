from __future__ import annotations

from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent
NVCC_ARGS = ["-O3", "-std=c++17"]
EXTRA_LINK_ARGS = []
if torch.version.hip is not None:
    NVCC_ARGS.append("--gpu-max-threads-per-block=256")
    NVCC_ARGS.append("-DMIOPEN_ADAPTIVE_LSTM_ENABLE_DIRECT_BLAS=1")
    EXTRA_LINK_ARGS.append("-lhipblas")


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
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": NVCC_ARGS,
            },
            extra_link_args=EXTRA_LINK_ARGS,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
