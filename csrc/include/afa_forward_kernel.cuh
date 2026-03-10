#pragma once

#include <cuda/std/limits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "afa_common.h"
#include "afa_config.cuh"
#include "afa_gemm.cuh"
#include "afa_ptx_warper.cuh"
#include "afa_softmax.cuh"
#include "afa_kernel_traits.cuh"
#include "afa_ldst.cuh"

namespace afa {

template <typename KernelTraits>
__global__ void
afa_forward_kernel(__grid_constant__ const AFAForwardParams params) {
    using accum_t = float;
    using index_t = int64_t;
    using TileScheduler = typename KernelTraits::TileScheduler;

    using value_t = typename KernelTraits::value_t;
    using QMatLDST = typename KernelTraits::QMatrixLDST;
    using KMatLDST = typename KernelTraits::KMatrixLDST;
    using VMatLDST = typename KernelTraits::VMatrixLDST;
    constexpr int async = KernelTraits::async_copy;

    const int sample = blockIdx.z;
    const int head = blockIdx.y;
    const int q_seq_block = blockIdx.x;

    const index_t gmem_seq_stride = params.seq_stride;

    const index_t sample_head_offset =
        sample * params.batch_stride + head * params.head_stride;
    // We only read/write one block for Q and O.
    // These offsets are the same for the whole thread-block.
    const index_t QO_gmem_block_offset =
        sample_head_offset + q_seq_block * KernelTraits::B_r * gmem_seq_stride;
    // We read the entire key sequence.
    const index_t KV_gmem_block_offset = sample_head_offset;

    value_t *gmem_Q = &static_cast<value_t *>(params.q_ptr)[QO_gmem_block_offset];
    value_t *gmem_O = &static_cast<value_t *>(params.o_ptr)[QO_gmem_block_offset];
    value_t *gmem_K = &static_cast<value_t *>(params.k_ptr)[KV_gmem_block_offset];
    value_t *gmem_V = &static_cast<value_t *>(params.v_ptr)[KV_gmem_block_offset];

    extern __shared__ __align__(16) char ch_smem[];
    value_t *smem_Q = reinterpret_cast<value_t *>(ch_smem);
    value_t *smem_O = smem_Q;
    value_t *smem_K = &smem_Q[KernelTraits::B_r * KernelTraits::d_head];
    value_t *smem_V = &smem_K[KernelTraits::B_c * KernelTraits::d_head];

    // Pointers to the K&V locations in smem that the warp copies to.
    QMatLDST Q(gmem_Q, gmem_seq_stride, smem_Q);
    KMatLDST K(gmem_K, gmem_seq_stride, smem_K);
    VMatLDST V(gmem_V, gmem_seq_stride, smem_V);
    // S is only stored in registers.
    typename KernelTraits::SAccumMatrixLDST S_accum(nullptr, -1, nullptr);
    // P is only stored in registers.
    typename KernelTraits::PValueMatrixLDST P_b16(nullptr, -1, nullptr);
    // The accumulator for O is only kept in registers. At the end of the
    // kernel, it is then converted into a 16-bit type and then copied into
    // gmem.
    typename KernelTraits::OAccumMatrixLDST O_accum(nullptr, -1, nullptr);
    typename KernelTraits::OValueMatrixLDST O_b16(gmem_O, gmem_seq_stride, smem_O);

    // Start the async copy of the Q and K tiles.
    Q.copy_GM2SM();
    cp_async_commit<async>();
    if constexpr (KernelTraits::eager_load_blocks) {
        K.copy_GM2SM();
        K.advance_gmem_block();
        cp_async_commit<async>();
    }

    O_accum.zero();

    // Initialize softmax_scale, m, and l.
    const accum_t softmax_scale = rsqrt(static_cast<accum_t>(KernelTraits::d_head)) *
                                  (KernelTraits::optimized_softmax ? M_LOG2E : 1.0);
    constexpr accum_t neg_inf = -cuda::std::numeric_limits<float>::infinity();
    accum_t m[TileScheduler::QO_row_fragments_per_warp];
    accum_t l[TileScheduler::QO_row_fragments_per_warp];
    AFA_UNROLL
    for (int q = 0; q < TileScheduler::QO_row_fragments_per_warp; ++q) {
        m[q] = neg_inf;
        l[q] = 0.0;
    }

    if constexpr (QMatLDST::load_entire_block_into_rf) {
        if constexpr (KernelTraits::eager_load_blocks) {
            // We only wait for the Q block to finish loading.
            cp_async_wait<1, async>();
        } else {
            cp_async_wait<0, async>();
        }
        // We need the __syncwarp() in addition to the cp_async_wait()
        // because cp_async_wait() only blocks until the current thread has
        // finished loading. The entire warp will read this block from
        // smem, so we need to wait on a warp-wide barrier.
        // For K and V, we will need a __syncthread() instead.
        __syncwarp();
        Q.copy_SM2RF();
    }

    for (int j = 0; j < params.n_KV_blocks; ++j) {
        if constexpr (!KernelTraits::eager_load_blocks) {
            K.copy_GM2SM();
            K.advance_gmem_block();
            cp_async_commit<async>();
        }
        // Initialize the registers for S to 0.
        S_accum.zero();

        // Block until we've copied the K block-tile for this iteration into
        // shared memory.
        cp_async_wait<0, async>();
        // After this barrier, it is safe to load the next V block, because all
        // warps have done the previous PV matmul.
        __syncthreads();

        if constexpr (KernelTraits::eager_load_blocks) {
            // Start the (async) copy for the V matrix from gmem to smem but
            // do not wait until after the S=QK matmul.
            V.copy_GM2SM();
            V.advance_gmem_block();
            cp_async_commit<async>();
        }
        if constexpr (KMatLDST::load_entire_block_into_rf) {
            K.copy_SM2RF();
        }

        matmul<KernelTraits::S_QK_GEMM>(Q, K, S_accum);
        cp_async_wait<0, async>();
        // After this barrier, it is safe to load the next block of K.
        __syncthreads();

        if constexpr (KernelTraits::eager_load_blocks) {
            // Start the async copy for the next K block-tile from gmem to
            // smem, but do not wait for the copy until the next iteration
            // when we need it.
            if (j < params.n_KV_blocks - 1) {
                K.copy_GM2SM();
                K.advance_gmem_block();
                cp_async_commit<async>();
            }
        }

        // Online softmax
        accum_t m_next[TileScheduler::QO_row_fragments_per_warp];
        if constexpr (!KernelTraits::optimized_softmax) {
            scale_S_accum(S_accum.data(), softmax_scale);
        }
        calc_row_max(S_accum.data(), m_next, m);
        scale_l_O<KernelTraits::optimized_softmax>(m_next, m, l, O_accum.data(),
                                             softmax_scale);
        exponentiate_tensor<KernelTraits::optimized_softmax>(S_accum.data(), m_next,
                                                       softmax_scale);
        update_row_exp_sum(S_accum.data(), l);

        // Convert the S accumulator block into P fp16 input block.
        convert_to_16_bit_dtype<value_t>(S_accum.data(), P_b16.data());

        if constexpr (!KernelTraits::eager_load_blocks) {
            // Load V from gmem to smem and block until it is done.
            V.copy_GM2SM();
            V.advance_gmem_block();
            cp_async_commit<async>();
            cp_async_wait<0, async>();
            __syncthreads();
        }

        if constexpr (VMatLDST::load_entire_block_into_rf) {
            V.copy_SM2RF();
        }

        matmul<typename KernelTraits::O_PV_GEMM>(P_b16, V, O_accum);
    }

    final_softmax_normalization(O_accum.data(), l);

    convert_to_16_bit_dtype<value_t>(O_accum.data(), O_b16.data());
    // Instead of writing directly to gmem, we write to smem as an intermediary
    // step. This allows us to
    // - use 16B vectorized stores, as opposed to 4B stores
    // - fully coalesce our stores
    //   - each warp can store 4x128B aligned lines (512B/warp) instead
    //   of 8x16B uncoalesced rows (128B/warp)
    O_b16.copy_RF2SM();

    // Wait until all threads in the same warp have written to smem.
    // We do not need __syncthreads() here because the warps operate on
    // independent chunks of O.
    __syncwarp();

    // Copy the final O tile from smem to gmem.
    O_b16.copy_SM2GM();
}

} // namespace flash
