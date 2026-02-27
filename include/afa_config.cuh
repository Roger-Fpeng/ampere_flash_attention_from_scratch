#pragma once

#include <torch/torch.h>

namespace afa {

struct AFAForwardParams {
    using index_t = int64_t;

    // Q, K, V, O are all [batch_size, seq_len, n_heads, d_head] tensors.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void *__restrict__ o_ptr;

    /*
     * For simplicity, all strides are the same across all inputs, which means
     * the inputs are in the same layout. While it is not the common case.
     */
    const index_t batch_stride;
    const index_t seq_stride;
    const index_t head_stride;

    const index_t seq_len;
    const index_t n_heads;

    /*
     * Block number of tiles. For example, if seq_len = 1024, B_r = 128, 
     * and B_c = 64, then n_Q_blocks = 1024 / 128 = 8, 
     * n_KV_blocks = 1024 / 64 = 16.
     */
    const int n_Q_blocks;
    const int n_KV_blocks;
};


/*
 * AFAForwardKernelConfig: configurations for the AFA forward kernel.
 * https://github.com/Dao-AILab/flash-attention uses static for kernel selection
 * at runtime. For AFA, we use a map of kernel configs to kernels, which is more
 * flexible while not fast as the static switches.
 * 
*/
struct AFAForwardKernelConfig {
    const torch::ScalarType dtype;
    const int d_head;  // [64, 128]
    // Tile width of the Q and O blocks.
    const int B_r;     // [64, 128]
    // Tile width of the K and V blocks.
    const int B_c;     // [32, 64, 128]
    const int n_warps; // [4, 8]. 8 only when B_r = 128

    /*
     * switch for optimizations:
     *   - async_copy: whether to use async copy for loading Q, K, V tiles.
     *   - eager_load_blocks: whether to load the next Q, K, V blocks before the
     *                        current block is consumed. 
     *   - swizzled: whether to use swizzled layout for Q, K, V tiles in shared 
     *              memory. Only applicable when async_copy is true.
     *   - mma_double_buffer_loads: whether to double buffer the loads for mma.
     *   - optimized_softmax: whether to use the optimized softmax kernel.
    */ 
    const bool async_copy;
    const bool eager_load_blocks;
    const bool swizzled;
    const bool mma_double_buffer_loads;
    const bool optimized_softmax;

    /*
     * Number of col fragments each warp loads for one warp mma. For example:
     * For d_head = 128, fragment shape is [8, 8] and Q_col_fragments_per_warp_mma = 8,
     * warp will load 8 col fragments and execute mma on Q[:, 0:64] and Q[:, 64:128] 
     * for the first and second warp mma respectively.
    */
    const int Q_col_fragments_per_warp_mma;
    const int K_col_fragments_per_warp_mma;
    const int V_col_fragments_per_warp_mma;

    bool operator<(const AFAForwardKernelConfig &other) const {
        if (dtype != other.dtype) {
            return dtype < other.dtype;
        }
        if (d_head != other.d_head) {
            return d_head < other.d_head;
        }
        if (B_r != other.B_r) {
            return B_r < other.B_r;
        }
        if (B_c != other.B_c) {
            return B_c < other.B_c;
        }
        if (n_warps != other.n_warps) {
            return n_warps < other.n_warps;
        }
        if (async_copy != other.async_copy) {
            return async_copy < other.async_copy;
        }
        if (eager_load_blocks != other.eager_load_blocks) {
            return eager_load_blocks < other.eager_load_blocks;
        }
        if (swizzled != other.swizzled) {
            return swizzled < other.swizzled;
        }
        if (mma_double_buffer_loads != other.mma_double_buffer_loads) {
            return mma_double_buffer_loads < other.mma_double_buffer_loads;
        }
        if (optimized_softmax != other.optimized_softmax) {
            return optimized_softmax < other.optimized_softmax;
        }
        if (Q_col_fragments_per_warp_mma != other.Q_col_fragments_per_warp_mma) {
            return Q_col_fragments_per_warp_mma < other.Q_col_fragments_per_warp_mma;
        }
        if (K_col_fragments_per_warp_mma != other.K_col_fragments_per_warp_mma) {
            return K_col_fragments_per_warp_mma < other.K_col_fragments_per_warp_mma;
        }
        if (V_col_fragments_per_warp_mma != other.V_col_fragments_per_warp_mma) {
            return V_col_fragments_per_warp_mma < other.V_col_fragments_per_warp_mma;
        }
        
        return false;
    }
};

} // namespace afa