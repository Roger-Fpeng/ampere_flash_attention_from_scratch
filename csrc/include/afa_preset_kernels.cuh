#pragma once

#include <map>
#include "afa_config.cuh"
#include "afa_forward_kernel.cuh"

namespace afa {

typedef void (*forward_kernel_fn)(const AFAForwardParams);

std::map<AFAForwardKernelConfig, forward_kernel_fn>
    forward_kernels = {
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles
        {AFAForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, false, false, false, 2, 2, 0}, &afa_forward_kernel<AFAForwardKernelTraits<AFAForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, false, false, false, 2, 2, 0}>>},
        {AFAForwardKernelConfig{torch::kFloat16, 256, 64, 32, 4, true, true, false, false, false, 2, 2, 0}, &afa_forward_kernel<AFAForwardKernelTraits<AFAForwardKernelConfig{torch::kFloat16, 256, 64, 32, 4, true, true, false, false, false, 2, 2, 0}>>},
        // {AFAForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, false, false, false, 2, 2, 0}, &afa_forward_kernel<AFAForwardKernelTraits<AFAForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, false, false, false, 2, 2, 0}>>},
    
    };
} // namespace afa

