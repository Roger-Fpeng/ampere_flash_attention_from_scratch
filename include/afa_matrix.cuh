#pragma once

#include "afa_common.h"
#include "afa_ldst.cuh"

namespace afa {

template <typename value_t_, int n_buffers, int row_elements, int col_elements>
struct RFStorage {
    using value_t = std::conditional_t<sizeof(value_t_) == 4, float, uint32_t>;

    // For 16-bit types (half/bf16): ldmatrix uses 1 uint32_t per fragment position.
    // For float (accumulator): 2 floats per N-fragment (N_REGS_PER_F32_ACCUM_FRAGMENT).
    static constexpr int regs_per_fragment = (sizeof(value_t_) == 4) ? 2 : 1;
    static constexpr int rows = row_elements;
    static constexpr int cols = col_elements * regs_per_fragment;

    // In afa, n_buffers is 1 or 2 (double buffer mode).
    value_t regs[n_buffers][rows][cols];

    AFA_DEVICE_CONSTEXPR value_t (&data(const int stage = 0))[rows][cols] {
        return reinterpret_cast<value_t(&)[rows][cols]>(regs[stage]);
    }

    AFA_DEVICE_CONSTEXPR void zero() {
        AFA_UNROLL
        for (int i = 0; i < n_buffers; ++i) {
            AFA_UNROLL
            for (int j = 0; j < rows; ++j) {
                AFA_UNROLL
                for (int k = 0; k < cols; ++k) {
                    regs[i][j][k] = 0;
                }
            }
        }
    }
};

/*
 * MatrixLDST encapsulates the logic for loading/storing a matrix tile from/to
 * global memory, shared memory, and registers. It provides a unified interface
 * for these operations, abstracting away the details of how data is moved between
 * different memory levels.
 */
template <MatrixLDSTConfig ldst, typename value_t, typename index_t = int64_t>
struct MatrixLDST {
    using MatStorage =
        RFStorage<value_t, ldst.mma_load_stages,
                    ldst.rf_layout.row_fragments,
                    ldst.rf_layout.col_fragments>;
    using GM2SM_op = std::conditional_t<ldst.ldst_config.async_copy,
                            GM2SM_async<value_t>, /* cp.async.cg.shared.global.L2::128B */
                            GM2SM<value_t>>;

    using SM2GM_op = SM2GM<value_t>;
    static constexpr int mma_load_stages = ldst.mma_load_stages;
    static constexpr bool load_entire_block_into_rf =
        ldst.load_entire_block_into_rf;
    static constexpr bool transposed = ldst.transposed;

    // Global memory pointer to the tile block that the warp will load/store from/to.
    value_t *gmem_ptr;
    index_t gmem_seq_stride;

    // Shared mem pointer LD/ST between shared memory and registers.
    value_t *smem_srm_ptr;
    /*
     * Shared mem pointer for LD/ST between global memory and shared memory.
     * Can not load entire block When load from global memory to shared memory.
     * smem_gsm_ptr = smem_ptr + warp_seq * ldst.smem_cols;
     */
    value_t *smem_gsm_ptr;

    const int lane_id;

    MatStorage storage;

    AFA_DEVICE MatrixLDST(value_t *gmem_block_ptr, index_t _gmem_seq_stride,
                         value_t *_smem_ptr)
        : lane_id(threadIdx.x % WARP_SIZE) {
        const int warp_rank = threadIdx.x / WARP_SIZE;

        /*
         * The begine of the tile block that the warp will operate on in global
         * memory. Each warp operates on a [ldst.warp_ldst_rows, d_heads] tile.
         */
        const index_t warp_seq = ldst.warp_ldst_rows * warp_rank;

        gmem_seq_stride = _gmem_seq_stride;
        gmem_ptr = gmem_block_ptr + warp_seq * gmem_seq_stride;

        smem_gsm_ptr = _smem_ptr + warp_seq * ldst.col_elements;
        smem_srm_ptr =
            ldst.compute_over_entire_block ? _smem_ptr : smem_gsm_ptr;
    }

    AFA_DEVICE_CONSTEXPR void zero() { storage.zero(); }

    AFA_DEVICE_CONSTEXPR typename MatStorage::value_t (&data(
        const int stage = 0))[MatStorage::rows][MatStorage::cols]
    {
        return storage.data(stage);
    }

    AFA_DEVICE_CONSTEXPR void advance_gmem_block() {
        gmem_ptr += ldst.block_size * gmem_seq_stride;
    }

    AFA_DEVICE_CONSTEXPR void copy_GM2SM() {
        copy_block_GSM<GM2SM_op, ldst>(gmem_ptr, smem_gsm_ptr, gmem_seq_stride,
                                       lane_id);
    }

    AFA_DEVICE_CONSTEXPR void copy_SM2GM() {
        copy_block_GSM<SM2GM_op, ldst>(gmem_ptr, smem_gsm_ptr, gmem_seq_stride,
                                       lane_id);
    }

    AFA_DEVICE_CONSTEXPR void copy_SM2RF(int stage = 0, int tile_offset = 0) {
        if constexpr (!transposed) {
            copy_warp_fragment_SM2RF<ldst, value_t>(
                storage.data(stage), smem_srm_ptr, lane_id, tile_offset);
        } else {
            copy_warp_fragment_transposed_SM2RF<ldst, value_t>(
                storage.data(stage), smem_srm_ptr, lane_id, tile_offset);
        }
    }

    AFA_DEVICE_CONSTEXPR void copy_RF2SM() {
        copy_warp_fragment_RF2SM<ldst, value_t>(data(), smem_srm_ptr, lane_id);
    }
};

} // namespace afa