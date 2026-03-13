import torch
import math
from dataclasses import dataclass
from enum import IntEnum
import itertools


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
class AFAForwardKernelConfig:
    dtype: DType = DType(5)
    d_head: int = 128
    B_r: int = 64
    B_c: int = 32
    n_warps: int = 4
    async_copy: bool = True
    eager_load_blocks: bool = True
    swizzled: bool = True
    mma_double_buffer_loads: bool = False
    optimized_softmax: bool = False

    Q_col_fragments_per_warp_gemm: int = 2
    K_col_fragments_per_warp_gemm: int = 2
    V_col_fragments_per_warp_gemm: int = 0

    def to_torch_dtype(self):
        return self.dtype

    def __str__(self):
        return self.string_config()

    def string_config(self):
        base = f"({self.dtype.name}, {self.d_head}, {self.B_r}, {self.B_c}, {self.n_warps}): "
        strs = []
        if self.async_copy:
            strs.append("async")
        if self.eager_load_blocks:
            strs.append("eager")
        if self.swizzled:
            strs.append("swizzled")

        if self.mma_double_buffer_loads:
            strs.append("buffer")
        if self.optimized_softmax:
            strs.append("opt_softmax")

        strs.append(
            f"load_{self.Q_col_fragments_per_warp_gemm}_{self.K_col_fragments_per_warp_gemm}_{self.V_col_fragments_per_warp_gemm}_tiles"
        )

        return base + "+".join(strs)
    
    def to_cpp_struct(self) -> str:
        def vstr(v):
            if isinstance(v, bool):
                return str(v).lower()
            else:
                return str(v)

        return (
            f"AFAForwardKernelConfig{{"
            f"{self.dtype.to_cpp_str()}, {self.d_head}, {self.B_r}, {self.B_c}, {self.n_warps}, "
            f"{vstr(self.async_copy)}, {vstr(self.eager_load_blocks)}, "
            f"{vstr(self.swizzled)}, {vstr(self.mma_double_buffer_loads)}, "
            f"{vstr(self.optimized_softmax)}, "
            f"{self.Q_col_fragments_per_warp_gemm}, {self.K_col_fragments_per_warp_gemm}, "
            f"{self.V_col_fragments_per_warp_gemm}"
            f"}}"
        )

    @classmethod
    def from_string_config(cls, config_str: str) -> "AFAForwardKernelConfig":
        """
        从字符串配置解析为 AFAForwardKernelConfig 对象。
        
        字符串格式: (FP16, 128, 64, 32, 4): async+eager+swizzled+buffer+opt_softmax+load_2_2_0_tiles
        
        Args:
            config_str: 配置字符串
            
        Returns:
            AFAForwardKernelConfig 对象
        """
        # 分离基础部分和标志部分
        if ":" not in config_str:
            raise ValueError(f"Invalid config string format: {config_str}")
        
        base_part, flags_part = config_str.split(":", 1)
        base_part = base_part.strip()
        flags_part = flags_part.strip()
        
        # 解析基础部分: (FP16, 128, 64, 32, 4)
        if not (base_part.startswith("(") and base_part.endswith(")")):
            raise ValueError(f"Invalid base part format: {base_part}")
        
        base_part = base_part[1:-1]  # 移除括号
        base_values = [v.strip() for v in base_part.split(",")]
        
        if len(base_values) != 5:
            raise ValueError(f"Expected 5 values in base part, got {len(base_values)}")
        
        dtype_str = base_values[0]
        if dtype_str == "FP16":
            dtype = DType.FP16
        elif dtype_str == "BF16":
            dtype = DType.BF16
        else:
            raise ValueError(f"Invalid dtype: {dtype_str}")
        
        # 解析数值参数
        d_head = int(base_values[1])
        B_r = int(base_values[2])
        B_c = int(base_values[3])
        n_warps = int(base_values[4])
        
        # 解析标志部分
        flags = [f.strip() for f in flags_part.split("+")]
        
        async_copy = "async" in flags
        eager_load_blocks = "eager" in flags
        swizzled = "swizzled" in flags
        mma_double_buffer_loads = "buffer" in flags
        optimized_softmax = "opt_softmax" in flags
        
        # 解析 load tiles
        Q_col_fragments = 0
        K_col_fragments = 0
        V_col_fragments = 0
        
        for flag in flags:
            if flag.startswith("load_") and flag.endswith("_tiles"):
                # 提取 load_2_2_0_tiles 中的数字
                load_part = flag[5:-6]  # 移除 "load_" 和 "_tiles"
                load_values = load_part.split("_")
                if len(load_values) == 3:
                    Q_col_fragments = int(load_values[0])
                    K_col_fragments = int(load_values[1])
                    V_col_fragments = int(load_values[2])
                break
        
        return cls(
            dtype=dtype,
            d_head=d_head,
            B_r=B_r,
            B_c=B_c,
            n_warps=n_warps,
            async_copy=async_copy,
            eager_load_blocks=eager_load_blocks,
            swizzled=swizzled,
            mma_double_buffer_loads=mma_double_buffer_loads,
            optimized_softmax=optimized_softmax,
            Q_col_fragments_per_warp_gemm=Q_col_fragments,
            K_col_fragments_per_warp_gemm=K_col_fragments,
            V_col_fragments_per_warp_gemm=V_col_fragments,
        )

    

def is_valid_config(cfg: AFAForwardKernelConfig) -> bool:
    if not cfg.async_copy and cfg.eager_load_blocks:
        return False
    if (
        cfg.Q_col_fragments_per_warp_gemm != cfg.K_col_fragments_per_warp_gemm
        and cfg.Q_col_fragments_per_warp_gemm != 0
    ):
        return False

    if cfg.B_r == 64:
        if cfg.n_warps == 8:
            return False
        elif (
            cfg.B_c == 32 and cfg.Q_col_fragments_per_warp_gemm == 0
        ):
            return False
        elif cfg.B_c == 64 and cfg.Q_col_fragments_per_warp_gemm != 0:
            return False
    elif cfg.B_r == 128:
        if cfg.Q_col_fragments_per_warp_gemm == 0:
            return False

    return True


def get_preset_kernel_configs():
    dtypes=[DType.FP16]   # [DType.BF16, DType.FP16]
    d_heads = [128]      #[64, 128]
    B_rs = [64, 128]
    B_cs = [32, 64]
    n_warps_cfgs = [4, 8]
    async_copy = [True]
    eager_load_blocks = [True]
    swizzleds = [True]
    mma_double_buffer_loads = [False, True]
    optimized_softmax = [False, True]
    Q_col_fragments_per_warp_gemm = [0, 2]
    K_col_fragments_per_warp_gemm = [0, 2]
    V_col_fragments_per_warp_gemm = [0, 2]

    params = [
        dtypes,
        d_heads,
        B_rs,
        B_cs,
        n_warps_cfgs,
        async_copy,
        eager_load_blocks,
        swizzleds,
        mma_double_buffer_loads,
        optimized_softmax,
        Q_col_fragments_per_warp_gemm,
        K_col_fragments_per_warp_gemm,
        V_col_fragments_per_warp_gemm,
    ]

    return [
        AFAForwardKernelConfig(*cfg)
        for cfg in itertools.product(*params)
        if is_valid_config(AFAForwardKernelConfig(*cfg))
    ]


def get_progressive_configs() -> list[AFAForwardKernelConfig]:
    base = "(FP16, 128, 64, 32, 4)"
    option_steps = [
        "",  # base config without any optimizations
        "async",
        "async+eager",
        "async+eager+swizzled",
        "async+eager+swizzled+load_2_2_2_tiles",
        "async+eager+swizzled+buffer+load_2_2_2_tiles",
        "async+eager+swizzled+buffer+opt_softmax+load_2_2_2_tiles",
    ]
    result = []
    for opts in option_steps:
        config_str = f"{base}: {opts}" if opts else f"{base}: "
        try:
            cfg = AFAForwardKernelConfig.from_string_config(config_str)
        except ValueError:
            continue
        # if is_valid_config(cfg):
        result.append(cfg)
    return result



# if __name__ == "__main__":
#     configs = get_progressive_configs()
#     for cfg in configs:
#         print(str(cfg))
#         print(cfg.to_cpp_struct())