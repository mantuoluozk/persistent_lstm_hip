from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent
NVCC_ARGS = ["-O3", "-std=c++17"]
if torch.version.hip is not None:
    NVCC_ARGS.append("--gpu-max-threads-per-block=512")


setup(
    name="persistent_lstm_hip",
    version="0.1.0",
    description="HIP/C++ skeleton for a persistent LSTM inference op",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="persistent_lstm_hip_ext",
            sources=[
                str(ROOT / "csrc" / "bindings.cpp"),
                str(ROOT / "csrc" / "persistent_lstm_op.cpp"),
                str(ROOT / "csrc" / "persistent_lstm_reference.cpp"),
                str(ROOT / "csrc" / "persistent_lstm_hip.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": NVCC_ARGS,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
