#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "afa_common.h"
#include "afa_ptx_warper.cuh"
#include "afa_swizzling.cuh"

namespace afa {

struct LDSTConfig {
    const bool swizzled;
    const bool async_copy;
};

// eg: for 8 x 8 fragment, the 16 x 64 tile layout is [2, 8]
struct TileLayout {
    const int row_fragments;
    const int col_fragments;
};

// constexpr non-type template parameter of LDST for a
// block (Q, K, V, or O) from gmem to smem and vice versa, and also loading from
// smem to the RF.
struct MatrixLDSTConfig {
    // This contains the # of (8, 8) tiles that each warp will load/store
    // between gmem and smem.
    const TileLayout gsm_layout;
    // contains the # of fragments that each warp will compute on.
    const TileLayout rf_layout;
    const LDSTConfig ldst_config;
    const bool transposed;
    const int block_size;
    const int col_elements;  // d_head for Q, K, V

    /*
     * Row number of matrix that a warp will independently load/store.
     * It is equivalent to gsm_layout.row_fragments * 8.
     */
    const int warp_ldst_rows;

    /*
     * Whether not the warp will compute over the entire block. 
     * This is false for (Q&O&S) and true for (K&V).
     */
    const bool compute_over_entire_block;

    const bool load_entire_block_into_rf;
    const int mma_load_stages;
};

template <typename T>
struct GM2SM_async {
    AFA_DEVICE_CONSTEXPR void operator()(T *gmem, T *smem) {
        cp_async<BYTES_PER_VEC4_ACCESS>(smem, gmem);
    }
};

/* uint4 = 4 * int = 16 bytes */
template <typename T>
struct GM2SM {
    AFA_DEVICE_CONSTEXPR void operator()(T *gmem, T *smem) {
        reinterpret_cast<uint4 *>(smem)[0] = reinterpret_cast<uint4 *>(gmem)[0];
    }
};

template <typename T>
struct SM2GM {
    AFA_DEVICE_CONSTEXPR void operator()(T *gmem, T *smem) {
        reinterpret_cast<uint4 *>(gmem)[0] = reinterpret_cast<uint4 *>(smem)[0];
    }
};


/*
 * copy_block_GSM: copy block between global memory and shared memory.
 * Each warp independently copies a (seq_len_per_warp, d_head) block, 
 * where seq_len_per_warp = gsm_layout.row_fragments * 8. 
 * Each inner iteration copies a (4, 64) tile, where each row is 
 * copied by a group of 8 consecutive threads. 
 * For example, when loading a (128, 64) block with 8 warps, each warp 
 * will load a (16, 64) block and there will be 4 iterations of the inner 
 * loop to load the entire block.
 */
template <typename op, /* either GM2SM_async or SM2GM */
          MatrixLDSTConfig MatLDSTCfg, typename value_t, typename index_t = int64_t>
AFA_DEVICE_CONSTEXPR void copy_block_GSM(value_t *gmem, value_t *smem,
                                        index_t gmem_seq_stride,
                                        const int lane_id) {
    // Each warp operates [MatLDSTCfg.gsm_layout.row_fragments * ROWS_PER_FRAGMENT, d_heads] block
    constexpr int n_row_iters =
        MatLDSTCfg.gsm_layout.row_fragments * ROWS_PER_FRAGMENT / GSM_LDST_ROWS_PER_ITER;

    /* each copy_s2g or g2s op will copy a (4, 64) tile, 
     * which corresponds to 8 col fragments, so each warp needs 
     * col_elements / 64 times copy to cover the entire d_head dimension
     */
    constexpr int col_fragments_per_iter = WARP_SIZE / GSM_LDST_ROWS_PER_ITER;
    constexpr int col_fragments_per_row = MatLDSTCfg.col_elements / COLS_PER_FRAGMENT;

    const int thread_row = lane_id / col_fragments_per_iter;
    const int thread_col_fragment = lane_id % col_fragments_per_iter;

    /*
     * We can also chose tile shape as [1, 4 * 64], if d_head % (4 * 64) == 0.
     */
    AFA_UNROLL
    for (int r = 0; r < n_row_iters; ++r) {
        const int cur_row = r * GSM_LDST_ROWS_PER_ITER + thread_row;
        AFA_UNROLL
        for (int c = 0; c < col_fragments_per_row; c += col_fragments_per_iter) {
            const int gmem_col_fragment = c + thread_col_fragment;
            const int smem_col_fragment =
                get_smem_col_fragment<col_fragments_per_row,
                                      MatLDSTCfg.ldst_config.swizzled>(cur_row,
                                                           gmem_col_fragment);

            op()(&gmem[cur_row * gmem_seq_stride +
                       gmem_col_fragment * COLS_PER_FRAGMENT],
                 &smem[cur_row * MatLDSTCfg.smem_cols +
                       smem_col_fragment * COLS_PER_FRAGMENT]);
        }
    }
}

// Loads matrix fragments in smem into registers.
// Each ldmatrix.x4 instruction loads a (16, 16) chunk, i.e. (2, 2) fragments.
// For this non-transposed version, the shape of the smem matches rmem, i.e.
// shape(rmem) = (r_r, r_c) = (s_r / 8, s_c / 8).
// This will be used to copy Q and K.
/*
 * copy_warp_fragment_SM2RF: copy a (16, 64) tile from shared memory to 
 * registers for each warp. 
 */
template <MatrixLDSTConfig MatLDSTCfg, typename value_t>
AFA_DEVICE_CONSTEXPR void copy_warp_fragment_SM2RF(
        // // QO_fragments_per_warp = 2, d_head_fragments = 8, Q_mma_load_K_fragments = 8, QO_rows_per_warp = 16
        // make_ldst_config({N::QO_fragments_per_warp, N::d_head_fragments},
        //     {N::QO_fragments_per_warp, N::Q_mma_load_K_fragments},
        //     false /*transposed*/, AFAFwdCfg.B_r, N::QO_rows_per_warp,
        //     false /*compute_over_entire_block*/,
        //     AFAFwdCfg.Q_mma_load_K_fragments == 0, N::Q_mma_load_stages);

    // 每个 regs 负责 8*8，所以一个warp 加载时需要的 reg 就是 2 * 8 
    uint32_t (&regs)[MatLDSTCfg.rf_layout.row_fragments][MatLDSTCfg.rf_layout.col_fragments], value_t *smem,
    const int lane_id, const int col_fragment_offset = 0) {
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter;

    constexpr int col_fragments = MatLDSTCfg.smem_cols / ELEMS_PER_VEC4_ACCESS;
    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;

    const int thread_row = lane_id % rows_per_iter;
    const int thread_col_fragment = lane_id / rows_per_iter;

    AFA_UNROLL
    for (int r = 0; r < MatLDSTCfg.rf_layout.row_fragments; r += row_fragments_per_iter) {
        const int cur_row = thread_row + r * ROWS_PER_FRAGMENT;
        AFA_UNROLL
        for (int c = 0; c < MatLDSTCfg.rf_layout.col_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment =
                get_smem_col_fragment<col_fragments, MatLDSTCfg.ldst_config.swizzled>(
                    cur_row, thread_col_fragment + c + col_fragment_offset);

            // 将一个连续 4 * (8 * 8) 的 float16 数据，加载到 4 个寄存器中,对于 64 * 64 的 tile 而言，每次其实就是
            // 加载 4 * 64 到 RF，每个 RF 有一行，总共四行
            ldmatrix_x4(&smem[cur_row * MatLDSTCfg.smem_cols +
                              smem_col_fragment * ELEMS_PER_VEC4_ACCESS],
                        regs[r][c], regs[r + 1][c], regs[r][c + 1],
                        regs[r + 1][c + 1]);
        }
    }
}

// Loads matrix fragments in smem into registers.
// Each ldmatrix.x4 instruction loads a (16, 16) chunk, i.e. (2, 2) fragments.
// For this transposed version, the shape of the smem matches the transpose of
// rmem, i.e. shape(rmem) = (r_r, r_c) = (s_c / 8, s_r / 8).
// This will be used to copy V.
template <MatrixLDSTConfig MatLDSTCfg, typename value_t>
AFA_DEVICE_CONSTEXPR void copy_warp_fragment_transposed_SM2RF(
    uint32_t (&regs)[MatLDSTCfg.rf_layout.row_fragments][MatLDSTCfg.rf_layout.col_fragments],
    value_t *smem,
    const int lane_id,
    const int row_fragment_offset = 0)
{
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter;

    constexpr int col_fragments = MatLDSTCfg.smem_cols / ELEMS_PER_VEC4_ACCESS;
    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;

    const int thread_row = lane_id % rows_per_iter;
    const int thread_col_fragment = lane_id / rows_per_iter;

    AFA_UNROLL
    for (int r = 0; r < MatLDSTCfg.rf_layout.col_fragments; r += row_fragments_per_iter) {
        const int cur_row =
            thread_row + (r + row_fragment_offset) * ROWS_PER_FRAGMENT;
        AFA_UNROLL
        for (int c = 0; c < MatLDSTCfg.rf_layout.row_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment =
                get_smem_col_fragment<col_fragments, MatLDSTCfg.ldst_config.swizzled>(
                    cur_row, thread_col_fragment + c);

            ldmatrix_x4_transpose(
                &smem[cur_row * MatLDSTCfg.smem_cols +
                      smem_col_fragment * ELEMS_PER_VEC4_ACCESS],
                regs[c][r], regs[c][r + 1], regs[c + 1][r], regs[c + 1][r + 1]);
        }
    }
}

// Copies matrix fragments in rmem to smem.
// Each iteration of the inner loop copies a (8, 8) tile, i.e. a single
// fragment. This will be used to copy O.
/*
 * 
 */
template <MatrixLDSTConfig MatLDSTCfg, typename value_t>
AFA_DEVICE_CONSTEXPR void copy_warp_fragment_RF2SM(
    uint32_t (&regs)[MatLDSTCfg.rf_layout.row_fragments][MatLDSTCfg.rf_layout.col_fragments],
    value_t *smem,
    const int lane_id)
{
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT;
    constexpr int col_fragments_per_iter = 1;
    constexpr int col_fragments = MatLDSTCfg.smem_cols / ELEMS_PER_VEC4_ACCESS;

    constexpr int elems_per_store = 2;
    const int thread_row = lane_id / 4;
    const int thread_inner_col = (lane_id % 4) * elems_per_store;

    AFA_UNROLL
    for (int r = 0; r < MatLDSTCfg.rf_layout.row_fragments; ++r) {
        const int cur_row = thread_row + r * rows_per_iter;
        AFA_UNROLL
        for (int c = 0; c < MatLDSTCfg.rf_layout.col_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment =
                get_smem_col_fragment<col_fragments, MatLDSTCfg.ldst_config.swizzled>(
                    cur_row, c);

            reinterpret_cast<uint32_t *>(
                &smem[cur_row * MatLDSTCfg.smem_cols +
                      (smem_col_fragment * ELEMS_PER_VEC4_ACCESS +
                       thread_inner_col)])[0] = regs[r][c];
        }
    }
}

template <typename value_t, int M_fragments, int N_fragments>
AFA_DEVICE_CONSTEXPR void
convert_to_16_bit_dtype(float (&src_float)[M_fragments][N_fragments * 2],
                        uint32_t (&dest_uint)[M_fragments][N_fragments])
{
    using value2_t =
        std::conditional_t<std::is_same_v<value_t, half>, half2, nv_bfloat162>;

    float2(&src)[M_fragments][N_fragments] =
        reinterpret_cast<float2(&)[M_fragments][N_fragments]>(src_float);
    value2_t(&dest)[M_fragments][N_fragments] =
        reinterpret_cast<value2_t(&)[M_fragments][N_fragments]>(dest_uint);
    AFA_UNROLL
    for (int m = 0; m < M_fragments; ++m) {
        AFA_UNROLL
        for (int n = 0; n < N_fragments; ++n) {
            if constexpr (std::is_same_v<value_t, half>) {
                dest[m][n] = __float22half2_rn(src[m][n]);
            } else {
                dest[m][n] = __float22bfloat162_rn(src[m][n]);
            }
        }
    }
}

} // namespace afa