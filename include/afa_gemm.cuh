#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

#include "afa_common.h"
#include "afa_ptx_warper.cuh"
#include "afa_utils.h"

namespace afa {

// m16n8k16 mma instruction configuration:
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

// with [8, 8] fragment shape
#define MMA_M_FRAGMENTS_PER_ITER 2
#define MMA_N_FRAGMENTS_PER_ITER 1
#define MMA_K_FRAGMENTS_PER_ITER 2

template <typename AMatrixLDST_, typename BMatrixLDST_, 
        typename CMatrixLDST_, int gemm_col_fragments,
        int load_K_fragments_per_iter, typename value_t_>
struct GEMM {
    using AMatrixLDST = AMatrixLDST_;
    using BMatrixLDST = BMatrixLDST_;
    using CMatrixLDST = CMatrixLDST_;
    using value_t = value_t_;

    static constexpr int TotalKTiles = gemm_col_fragments;
    static constexpr int LoadKTilesPerIter = load_K_fragments_per_iter;

    static constexpr bool DoubleBufferA =
        !AMatrixLDST::load_entire_block_into_rf && AMatrixLDST::mma_load_stages > 1;
    static constexpr bool DoubleBufferB =
        !BMatrixLDST::load_entire_block_into_rf && BMatrixLDST::mma_load_stages > 1;
    static constexpr bool DoubleBuffer = DoubleBufferA || DoubleBufferB;
};

/*
 * warp_fragment_mma_f32_accum: perform fragment-level mma and accumulate into f32 registers.
 * For example, when calculating QK^T, if Q and K are tiled:
 *   Q_tile fragments shape: [Q_row_fragments_per_warp_mma, Q_col_fragments_per_warp_mma]
 *   K_tile fragments shape: [K_row_fragments_per_warp_mma, K_col_fragments_per_warp_mma]
 * Here, K is transposed and Q_col_fragments_per_warp_mma = K_row_fragments_per_warp_mma
 * but Q_col_fragments_per_warp_mma may not be equal to d_head_fragments.
 */
template <typename value_t, const int M_fragments, const int N_fragments,
          const int A_col_fragments, /* A_col_fragments_per_warp_mma */
          const int B_col_fragments, /* B_col_fragments_per_warp_mma */
          typename accum_t = float>
AFA_DEVICE_CONSTEXPR void warp_fragment_mma_f32_accum(
    uint32_t (&regs_A)[M_fragments][A_col_fragments],
    uint32_t (&regs_B)[N_fragments][B_col_fragments],
    accum_t (&regs_C)[M_fragments][N_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT],
    int A_col_fragment_offset = 0, int B_col_fragment_offset = 0) {
    constexpr int K_iters = constexpr_min(A_col_fragments, B_col_fragments);

    AFA_UNROLL
    for (int k = 0; k < K_iters; k += MMA_K_FRAGMENTS_PER_ITER) {
        AFA_UNROLL
        for (int m = 0; m < M_fragments; m += MMA_M_FRAGMENTS_PER_ITER) {
            AFA_UNROLL
            for (int n = 0; n < N_fragments; n +=  MMA_N_FRAGMENTS_PER_ITER) {
                mma_m16n8k16_f32_accum<value_t>(
                    regs_C[m][n * 2], regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2], regs_C[m + 1][n * 2 + 1],
                    regs_A[m][k + A_col_fragment_offset],
                    regs_A[m + 1][k + A_col_fragment_offset],
                    regs_A[m][k + 1 + A_col_fragment_offset],
                    regs_A[m + 1][k + 1 + A_col_fragment_offset],
                    regs_B[n][k + B_col_fragment_offset],
                    regs_B[n][k + 1 + B_col_fragment_offset], regs_C[m][n * 2],
                    regs_C[m][n * 2 + 1], regs_C[m + 1][n * 2],
                    regs_C[m + 1][n * 2 + 1]);
            }
        }
    }
}

template <typename GEMM>
AFA_DEVICE_CONSTEXPR void matmul(typename GEMM::AMatrixLDST &A,
                                typename GEMM::BMatrixLDST &B,
                                typename GEMM::CMatrixLDST &C) {
    using AMat = typename GEMM::AMatrixLDST;
    using BMat = typename GEMM::BMatrixLDST;
    using value_t = typename GEMM::value_t;

    /*
     * In DoubleBuffer mode, A_stage_toggle = 1, A_stage switches between 0 and 1 
     * to indicate which buffer to load from or compute on. Same for B.
     */
    constexpr int A_stage_toggle = AMat::mma_load_stages - 1;
    constexpr int B_stage_toggle = BMat::mma_load_stages - 1;
    int A_stage = 0;
    int B_stage = 0;

    if constexpr (GEMM::DoubleBufferA) {
        A.copy_SM2RF(A_stage);
    }
    if constexpr (GEMM::DoubleBufferB) {
        B.copy_SM2RF(B_stage);
    }

    AFA_UNROLL
    for (int k_outer_fragment = 0; k_outer_fragment < GEMM::TotalKTiles;
         k_outer_fragment += GEMM::LoadKTilesPerIter) {
        if constexpr (!AMat::load_entire_block_into_rf ||
                      !BMat::load_entire_block_into_rf) {
            int k_load_fragment =
                    k_outer_fragment +
                    (GEMM::DoubleBuffer ? GEMM::LoadKTilesPerIter : 0);
            if (k_load_fragment < GEMM::TotalKTiles) {
                if constexpr (!AMat::load_entire_block_into_rf) {
                    A.copy_SM2RF(A_stage_toggle ^ A_stage, k_load_fragment);
                }
                if constexpr (!BMat::load_entire_block_into_rf) {
                    B.copy_SM2RF(B_stage_toggle ^ B_stage, k_load_fragment);
                }
            }
        }

        int A_col_offset =
            AMat::load_entire_block_into_rf ? k_outer_fragment : 0;
        int B_col_offset =
            BMat::load_entire_block_into_rf ? k_outer_fragment : 0;
        
        /*
         * template arg <value_t> is necessary to correctly handle the case when 
         * A and B are in fp16/bf16. 
         */
        // Perform tile-wise outer products.
        warp_fragment_mma_f32_accum<value_t>(A.data(A_stage), B.data(B_stage),
                                             C.data(), A_col_offset,
                                             B_col_offset);

        A_stage ^= A_stage_toggle;
        B_stage ^= B_stage_toggle;
    }
}

} // namespace afa