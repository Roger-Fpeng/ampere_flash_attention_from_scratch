import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Keep defaults conservative to avoid exploding compile time.
DTYPES = ["torch::kFloat16", "torch::kBFloat16"]
D_HEADS = [128, 64]
B_RS = [128, 64]
B_CS = [128, 64, 32]
N_WARPS = [4, 8]
ASYNC_COPY = [True]
EAGER_LOAD_BLOCKS = [True]
SWIZZLED = [False]
MMA_DOUBLE_BUFFER_LOADS = [False]
OPTIMIZED_SOFTMAX = [False]
Q_COL_FRAGMENTS_PER_WARP_MMA = [0, 2]
K_COL_FRAGMENTS_PER_WARP_MMA = [0, 2]
V_COL_FRAGMENTS_PER_WARP_MMA = [0]


@dataclass(frozen=True)
class AFAForwardKernelConfigSpec:
    dtype: str
    d_head: int
    B_r: int
    B_c: int
    n_warps: int
    async_copy: bool
    eager_load_blocks: bool
    swizzled: bool
    mma_double_buffer_loads: bool
    optimized_softmax: bool
    Q_col_fragments_per_warp_mma: int
    K_col_fragments_per_warp_mma: int
    V_col_fragments_per_warp_mma: int

    @staticmethod
    def _cpp_bool(v: bool) -> str:
        return "true" if v else "false"

    @staticmethod
    def _is_power_of_2_and_not_1(v: int) -> bool:
        # Mirror include/afa_kernel_traits.cuh static_assert behavior.
        if v == 1:
            return False
        return (v & (v - 1)) == 0

    def _valid_col_fragments(self, loaded: int, dim: int) -> bool:
        if not self._is_power_of_2_and_not_1(loaded):
            return False
        max_frags = (dim // 2 if self.mma_double_buffer_loads else dim) // 8
        return loaded <= max_frags

    def is_valid(self) -> bool:
        if self.n_warps == 8 and self.B_r != 128:
            return False
        if self.Q_col_fragments_per_warp_mma != self.K_col_fragments_per_warp_mma and self.Q_col_fragments_per_warp_mma != 0:
            return False
        if not self._valid_col_fragments(self.Q_col_fragments_per_warp_mma, self.d_head):
            return False
        if not self._valid_col_fragments(self.K_col_fragments_per_warp_mma, self.d_head):
            return False
        if not self._valid_col_fragments(self.V_col_fragments_per_warp_mma, self.B_c):
            return False
        if self.swizzled and not self.async_copy:
            return False
        return True

    def to_cpp_initializer(self) -> str:
        return (
            "AFAForwardKernelConfig{"
            f"{self.dtype}, {self.d_head}, {self.B_r}, {self.B_c}, {self.n_warps}, "
            f"{self._cpp_bool(self.async_copy)}, "
            f"{self._cpp_bool(self.eager_load_blocks)}, "
            f"{self._cpp_bool(self.swizzled)}, "
            f"{self._cpp_bool(self.mma_double_buffer_loads)}, "
            f"{self._cpp_bool(self.optimized_softmax)}, "
            f"{self.Q_col_fragments_per_warp_mma}, "
            f"{self.K_col_fragments_per_warp_mma}, "
            f"{self.V_col_fragments_per_warp_mma}"
            "}"
        )

    def to_map_entry(self) -> str:
        cfg = self.to_cpp_initializer()
        return (
            f"        {{{cfg}, "
            f"&afa_forward_kernel<AFAForwardKernelTraits<{cfg}>>}},"
        )


def iter_candidate_configs() -> Iterable[AFAForwardKernelConfigSpec]:
    for values in itertools.product(
        DTYPES,
        D_HEADS,
        B_RS,
        B_CS,
        N_WARPS,
        ASYNC_COPY,
        EAGER_LOAD_BLOCKS,
        SWIZZLED,
        MMA_DOUBLE_BUFFER_LOADS,
        OPTIMIZED_SOFTMAX,
        Q_COL_FRAGMENTS_PER_WARP_MMA,
        K_COL_FRAGMENTS_PER_WARP_MMA,
        V_COL_FRAGMENTS_PER_WARP_MMA,
    ):
        yield AFAForwardKernelConfigSpec(*values)


def generate_preset_kernels_header(valid_configs: list[AFAForwardKernelConfigSpec]) -> str:
    entries = "\n".join(cfg.to_map_entry() for cfg in valid_configs)
    return f"""#pragma once

#include <map>
#include \"afa_config.cuh\"
#include \"afa_forward_kernel.cuh\"

namespace afa {{

typedef void (*forward_kernel_fn)(const AFAForwardParams);

std::map<AFAForwardKernelConfig, forward_kernel_fn>
    forward_kernels = {{
{entries}
    }};

}} // namespace afa
"""


def main(output_path: str | None) -> None:
    if output_path is None:
        output_path = (
            Path(__file__).resolve().parent.parent / "include" / "afa_preset_kernels.cuh"
        )
    else:
        output_path = Path(output_path)

    valid_configs = [cfg for cfg in iter_candidate_configs() if cfg.is_valid()]
    if not valid_configs:
        raise ValueError("No valid kernel configuration is generated.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generate_preset_kernels_header(valid_configs))
    print(f"Generated {len(valid_configs)} kernel preset entries to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate include/afa_preset_kernels.cuh based on AFAForwardKernelConfig combinations.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        help="Output header path. Default: include/afa_preset_kernels.cuh",
    )
    args = parser.parse_args()
    main(args.output)
