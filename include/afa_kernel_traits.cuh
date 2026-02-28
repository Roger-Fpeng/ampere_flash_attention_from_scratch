#pragma once

#include "afa_common.h"
#include "afa_config.cuh"
#include "afa_gemm.cuh"
#include "afa_ldst.cuh"
#include "afa_tensor.cuh"
#include "afa_utils.h"

// static_assert(CUTE_STATIC_V(size(tensor)) % FragmentSize == 0, "Fragment size does not vectorize properly");
namespace afa {

template <int col_fragments_loaded, int d_head, bool double_buffer>
constexpr void static_assert_valid_kernel_config() {
    static_assert(((col_fragments_loaded & (col_fragments_loaded - 1)) == 0) 
                && col_fragments_loaded != 1,
                "col fragments loaded must be power of 2 and not 1");

    constexpr int max_frags = (double_buffer ? d_head / 2 : d_head) / 8;
    static_assert(col_fragments_loaded <= max_frags,
            "col fragments loaded must be less or equal to max fragments");
}

template <AFAForwardKernelConfig FwdKernelCfg>
constexpr bool valid_kernel_config() {
    static_assert_valid_kernel_config<FwdKernelCfg.Q_col_fragments_per_warp_mma, 
            FwdKernelCfg.d_head, FwdKernelCfg.mma_double_buffer_loads>();
    static_assert_valid_kernel_config<FwdKernelCfg.K_col_fragments_per_warp_mma,
            FwdKernelCfg.d_head, FwdKernelCfg.mma_double_buffer_loads>();
    static_assert_valid_kernel_config<FwdKernelCfg.V_col_fragments_per_warp_mma,
            FwdKernelCfg.B_c, FwdKernelCfg.mma_double_buffer_loads>();

    static_assert((FwdKernelCfg.Q_col_fragments_per_warp_mma == 
                FwdKernelCfg.K_col_fragments_per_warp_mma) ||
                FwdKernelCfg.Q_col_fragments_per_warp_mma == 0);

    return true;
}


/*
 * AFAForwardTileScheduler computes the tiling block shape. It is used for 
 * loading/storing Q, K, V, O tiles and the corresponding
 * GEMM configurations.
 */
template <AFAForwardKernelConfig FwdKernelCfg>
struct AFAForwardTileScheduler {
    static_assert(valid_kernel_config<FwdKernelCfg>());

    /*
     * The col fragments: i.e. d_head = 128 with fragment size 8, 
     * d_head_fragments = 16. For m16n8k16, each institution operates
     * 4 * [8, 8] Q fragments, it needs 4 times of m16n8k16 to cover 
     * the entire d_head dimension. 
     */
    static constexpr int d_head_fragments = FwdKernelCfg.d_head / COLS_PER_FRAGMENT;
   

    /*
     * The row number of the tile that each warp operates on, 
     * which corresponds to a (B_r/n_warps, d_head) chunk.
     */
    static constexpr int QO_rows_per_warp = FwdKernelCfg.B_r / FwdKernelCfg.n_warps;
    static constexpr int QO_row_fragments_per_warp =
        QO_rows_per_warp / ROWS_PER_FRAGMENT;

    /*
     * Each warp loads and operates on a K/V tile independently, which 
     * corresponds to a (B_c, d_head) chunk, but each Q/O chunk will perform 
     * computations with the entire K/V block loaded by the thread-block.
     */
    static constexpr int KV_tile_row_fragments = 
                            FwdKernelCfg.B_c / ROWS_PER_FRAGMENT;

    /*
     * i.e. B_c = 64, fragment [8, 8], each tile has 8 row fragments, 
     * with 4 warps, each warp loads 2 row fragments
     */
    static constexpr int KV_row_fragments_per_warp =
                                KV_tile_row_fragments / FwdKernelCfg.n_warps;

    
    static constexpr int KV_rows_per_warp =
                                FwdKernelCfg.B_c / FwdKernelCfg.n_warps;

    // col fragments to load in warp matmuls which execute mma.
    static constexpr int Q_col_fragments_per_warp_mma =
        FwdKernelCfg.Q_col_fragments_per_warp_mma == 0 ? d_head_fragments
                                    : FwdKernelCfg.Q_col_fragments_per_warp_mma;
    static constexpr int Q_mma_load_stages =
                (FwdKernelCfg.Q_col_fragments_per_warp_mma > 0 && 
                    FwdKernelCfg.mma_double_buffer_loads) ? 2 : 1;

    static constexpr int K_col_fragments_per_warp_mma =
        FwdKernelCfg.K_col_fragments_per_warp_mma == 0 ? d_head_fragments
                                    : FwdKernelCfg.K_col_fragments_per_warp_mma;
    static constexpr int K_mma_load_stages =
                (FwdKernelCfg.K_col_fragments_per_warp_mma > 0 && 
                    FwdKernelCfg.mma_double_buffer_loads) ? 2 : 1;

    static constexpr int V_col_fragments_per_warp_mma =
        FwdKernelCfg.V_col_fragments_per_warp_mma == 0 ? KV_calc_fragments
                                        : FwdKernelCfg.V_col_fragments_per_warp_mma;
    static constexpr int V_mma_load_stages =
                (FwdKernelCfg.V_col_fragments_per_warp_mma > 0 &&
                    FwdKernelCfg.mma_double_buffer_loads) ? 2 : 1;
};

template <AFAForwardKernelConfig FwdKernelCfg>
struct AFAForwardKernelTraits {
    using accum_t = float;
    using value_t = typename std::conditional_t<FwdKernelCfg.dtype == torch::kBFloat16,
                                                nv_bfloat16, half>;
    using TileScheduler = AFAForwardTileScheduler<FwdKernelCfg>;

    // Static configuration fields accessed from the original CFG
    static constexpr bool async_copy = FwdKernelCfg.async_copy;
    static constexpr int B_r = FwdKernelCfg.B_r;
    static constexpr int B_c = FwdKernelCfg.B_c;
    static constexpr int d_head = FwdKernelCfg.d_head;
    static constexpr bool eager_load_blocks = FwdKernelCfg.eager_load_blocks;
    static constexpr bool optimized_softmax = FwdKernelCfg.optimized_softmax;

    static constexpr LDSTConfig LDSTCfg{FwdKernelCfg.swizzled, FwdKernelCfg.async_copy};

    static constexpr TensorLDSTConfig make_ldst_config(
        TileLayout GSMem_layout, TileLayout RF_layout, bool transposed, int tile_block_size,
        int ldst_rows_per_warp, bool compute_over_entire_block,
        bool load_entire_block_into_rf = true, int mma_load_stages = 1) {

        return TensorLDSTConfig{GSMem_layout,
                                RF_layout,
                                LDSTCfg,
                                transposed,
                                tile_block_size,
                                FwdKernelCfg.d_head,
                                ldst_rows_per_warp,
                                compute_over_entire_block,
                                load_entire_block_into_rf,
                                mma_load_stages};
    }

    static constexpr TensorLDSTConfig Q_LDST =
        make_ldst_config({TileScheduler::QO_row_fragments_per_warp, TileScheduler::d_head_fragments},
                         {TileScheduler::KV_row_fragments_per_warp, TileScheduler::Q_col_fragments_per_warp_mma},
                         false /*transposed*/, FwdKernelCfg.B_r, TileScheduler::QO_rows_per_warp,
                         false /*compute_over_entire_block*/,
                         FwdKernelCfg.Q_col_fragments_per_warp_mma == 0,
                         TileScheduler::Q_mma_load_stages);
    using QMatrixLDST = MatrixLDST<Q_LDST, value_t>;

    static constexpr TensorLDSTConfig K_LDST = make_ldst_config(
        {TileScheduler::KV_row_fragments_per_warp, TileScheduler::d_head_fragments},
        {TileScheduler::KV_row_fragments_per_warp, TileScheduler::K_col_fragments_per_warp_mma},
        false /*transposed*/,FwdKernelCfg.B_c, TileScheduler::KV_rows_per_warp,
        true /*compute_over_entire_block*/,
        FwdKernelCfg.K_col_fragments_per_warp_mma == 0,
        TileScheduler::K_mma_load_stages);
    using KMatrixLDST = MatrixLDST<K_LDST, value_t>;

    static constexpr TensorLDSTConfig V_LDST = make_ldst_config(
        {TileScheduler::KV_row_fragments_per_warp,TileScheduler::d_head_fragments},
        {TileScheduler::KV_row_fragments_per_warp, TileScheduler::V_col_fragments_per_warp_mma},
        true /*transposed*/, FwdKernelCfg.B_c, 
        TileScheduler::KV_rows_per_warp,
        true /*compute_over_entire_block*/,
        FwdKernelCfg.V_col_fragments_per_warp_mma == 0,
        TileScheduler::V_mma_load_stages);
    using VMatrixLDST = MatrixLDST<V_LDST, value_t>;
    
    static constexpr TensorLDSTConfig O_LDST =
        make_ldst_config({TileScheduler::QO_row_fragments_per_warp, TileScheduler::d_head_fragments},
                         {TileScheduler::QO_row_fragments_per_warp, TileScheduler::d_head_fragments},
                         false /*transposed*/,
                         FwdKernelCfg.B_r,
                         TileScheduler::QO_rows_per_warp,
                         false /*compute_over_entire_block*/, 
                         true);
    using OAccumMatrixLDST = MatrixLDST<O_LDST, accum_t>;
    using OValueMatrixLDST = MatrixLDST<O_LDST, value_t>;

    // S/P is kept entirely in the RF during the entire duration of the kernel.
    static constexpr TensorLDSTConfig S_LDST = make_ldst_config(
        {TileScheduler::QO_row_fragments_per_warp, TileScheduler::d_head_fragments},
        {TileScheduler::QO_row_fragments_per_warp, TileScheduler::d_head_fragments},
        false, FwdKernelCfg.B_r, false,
        0 /* only stored in RF, not smem or gmem */,
        false /*compute_over_entire_block*/);
    using SAccumMatrixLDST = MatrixLDST<S_LDST, accum_t>;
    using PValueMatrixLDST = MatrixLDST<S_LDST, value_t>;

    using S_QK_GEMM = GEMM<QMatrixLDST, KMatrixLDST, SAccumMatrixLDST,
                            TileScheduler::d_head_fragments,
                            constexpr_min(TileScheduler::Q_col_fragments_per_warp_mma,
                                         TileScheduler::K_col_fragments_per_warp_mma),
                           value_t>;
    using O_PV_GEMM = GEMM<PValueMatrixLDST, VMatrixLDST, OAccumMatrixLDST,
                           TileScheduler::KV_row_fragments_per_warp,
                           TileScheduler::V_col_fragments_per_warp_mma,
                           value_t>;
};

} // namespace afa
