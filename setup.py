import os
import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.1"

setup_dir = os.path.dirname(os.path.abspath(__file__))

include_dirs = [Path(setup_dir) / "csrc/include"]


def get_nvcc_compile_args():
    # Check for --debug flag in command line arguments
    debug_mode = "--debug" in sys.argv
    if debug_mode:
        sys.argv.remove("--debug")
    sub_dir = "debug" if debug_mode else "release"
    intermediate_dir = Path(setup_dir) / "build" / "nvcc_temp" / sub_dir
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "-std=c++20",
        '-Xcudafe="--diag_suppress=3189"',  # pytorch warnings for c++20
        "--use_fast_math",
        "--keep",
        f"--keep-dir={intermediate_dir}",
        # build info flags
        "--generate-line-info",
        "--resource-usage",
        "--expt-relaxed-constexpr",
        "-Xptxas=-warn-lmem-usage",
        "-Xptxas=-warn-spills",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-Xcompiler=-fdiagnostics-color=always",
        "--ftemplate-backtrace-limit=0",
        "--keep",
        "-gencode",
        "arch=compute_80,code=sm_80",
    ]
    if debug_mode:
        args.extend(["-g", "-G", "-O0"])
    else:
        args.extend(["-O3"])

    return args


ext_modules = [
    CUDAExtension(
        name="afa_flash_attention_kernels",
        sources=[
            "csrc/ampere_flash_attention.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-fdiagnostics-color=always"],
            "nvcc": get_nvcc_compile_args(),
        },
        extra_link_args=["-Wl,--no-as-needed", "-lcuda"],
        include_dirs=include_dirs,
    )
]

setup(
    name="flash_attention",
    packages=find_packages(exclude=["build", "py", "src", "test", "tools"]),
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch",
        "einops",
    ],
    setup_requires=[
        "packaging",
        "psutil",
        "ninja",
    ],
)
