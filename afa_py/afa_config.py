import torch
import math
from dataclasses import dataclass
from enum import IntEnum


class DType(IntEnum):
    # https://github.com/pytorch/pytorch/blob/c37ddcaefbe9b877e1816ce97dedb8ad26d09450/c10/core/ScalarType.h
    # These are the enum values for the torch types
    FP16 = 5
    BF16 = 15

    def to_cpp_str(self) -> str:
        if self == DType.FP16:
            return "torch::kFloat16"
        elif self == DType.BF16:
            return "torch::kBFloat16"
        else:
            raise ValueError(f"Invalid DType: {self}")

    def to_torch_dtype(self):
        if self == DType.FP16:
            return torch.float16
        elif self == DType.BF16:
            return torch.bfloat16
        else:
            raise ValueError(f"Invalid DType: {self}")



# 与 C++ AFAForwardKernelConfig 对应的 Python 类
@dataclass
class KernelConfig:
    dtype: DType = DType(5)  # 默认为 FP16,
    d_head: int = 128
    B_r: int = 64
    B_c: int = 32
    n_warps: int = 4
    async_copy: bool = True
    eager_load_blocks: bool = True
    swizzled: bool = False
    Q_mma_load_K_tiles: int = 2
    K_mma_load_K_tiles: int = 2
    V_mma_load_K_tiles: int = 0
    mma_double_buffer_loads: bool = False
    optimized_softmax: bool = False

    def to_torch_dtype(self):
        return self.dtype