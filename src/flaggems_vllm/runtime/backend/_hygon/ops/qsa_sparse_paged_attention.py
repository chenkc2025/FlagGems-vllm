# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors
"""Frozen Hygon QSA kernels and the complete vLLM v0.32.0 call contract.

Native source: Hygon BoltOps boltops/qsa/triton/qsa.py, from vLLM
0.28.1rc1 DTK26041 20261013-0612-hcu-bert-opt; original SHA256:
a200bbb59e621fdc72bcbc9944af4cb7c7de287c23ffbd3d123d0d5f43d8405e.
Original SPDX identifies Apache-2.0 and vLLM project contributors. The vendor's
exact source commit is unknown. Upstream: https://github.com/vllm-project/vllm.
Changes: delayed V loads, HCU M16 for Hq6/D256, exact per-split extent scan,
split FP8 QK/PV dimensions, BF16 dequantization, FP32 merge, frozen 940 policy.
No new native kernel optimization or tuning is introduced by this integration.

Complete interface reference: vLLM commit
5df63a7203cdb9328d824a27b682e77c7b9928be, v0.32.0,
vllm/models/qwen4_exp/nvidia/ops/qsa.py. E5M2 KV is a Hygon extension.
Packed-count conversion and sigmoid gate are new; full-entry HCU validation is
pending. Native-only historical Graph results do not measure these adapters.
This Hygon specialization does not modify any generic operator implementation.
"""
from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime

_TARGET_SHAPE = (1, 2_051, 256)


class SparseLaunchConfig(NamedTuple):
    block_n: int
    target_splits: int
    num_warps: int
    num_stages: int
    merge_warps: int
    merge_dim_tile: int


def _sparse_baseline_values(
    rows: int,
    kv_heads: int,
    block_m: int = 8,
) -> SparseLaunchConfig:
    base_programs = rows * kv_heads
    small_profile_limit = 8 if block_m <= 8 else 4
    if base_programs <= small_profile_limit:
        block_n, target_splits, num_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, num_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, num_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, num_warps = 64, 4, 2
    else:
        block_n, target_splits, num_warps = 64, 1, 2
    # Small decode batches are occupancy-bound and benefit from DIM_TILE=64.
    # Larger merges are bandwidth-bound; one warp and the full head minimize
    # repeated LSE reduction work.
    merge_dim_tile = 64 if base_programs <= 8 else 256
    return SparseLaunchConfig(block_n, target_splits, num_warps, 1, 1, merge_dim_tile)


def _sparse_config(
    values: SparseLaunchConfig,
) -> dict[str, int]:
    return values._asdict()


# Only measured non-baseline segments belong here.  Rows outside a segment use
# the source-equivalent baseline dispatch above.  Segment boundaries follow the
# measured decode/prefill crossover points for the target shape on HCU.
TUNED_TARGET_SEGMENTS: tuple[
    tuple[int, int, tuple[int, int, int, int, int, int]], ...
] = (
    (2, 2, (32, 32, 4, 1, 1, 256)),
    (3, 23, (32, 16, 2, 1, 1, 256)),
    (24, 256, (32, 1, 2, 1, 1, 256)),
    (257, 512, (16, 1, 2, 1, 1, 256)),
    (513, 262_144, (16, 1, 2, 1, 1, 256)),
)


def _tuned_sparse_values(
    rows: int,
) -> SparseLaunchConfig | None:
    for start, end, values in TUNED_TARGET_SEGMENTS:
        if start <= rows <= end:
            return SparseLaunchConfig(*values)
    return None


def _validate_sparse_config_shape(
    rows: int,
    kv_heads: int,
    topk: int,
    head_dim: int,
) -> None:
    if not all(isinstance(value, int) for value in (rows, kv_heads, topk, head_dim)):
        raise TypeError("QSA sparse configuration dimensions must be integers.")
    if rows < 1 or kv_heads < 1 or topk < 1 or head_dim < 1:
        raise ValueError("QSA sparse configuration dimensions must be positive.")


@lru_cache(maxsize=128)
def _select_qsa_sparse_config(
    rows: int,
    kv_heads: int,
    topk: int,
    head_dim: int,
    block_m: int = 8,
) -> dict[str, int]:
    _validate_sparse_config_shape(rows, kv_heads, topk, head_dim)
    tuned = (
        _tuned_sparse_values(rows)
        if (kv_heads, topk, head_dim) == _TARGET_SHAPE
        else None
    )
    values = tuned or _sparse_baseline_values(rows, kv_heads, block_m)
    return _sparse_config(values)


def _baseline_qsa_sparse_config(
    rows: int,
    kv_heads: int,
    topk: int,
    head_dim: int,
    block_m: int = 8,
) -> dict[str, int]:
    _validate_sparse_config_shape(rows, kv_heads, topk, head_dim)
    return _sparse_config(_sparse_baseline_values(rows, kv_heads, block_m))


@triton.jit
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    k_scale_ptr,
    v_scale_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FP8: tl.constexpr,
) -> None:
    """Sparse paged GQA, ported from vLLM's Apache-2.0 QSA kernel."""
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    if IS_FP8 and HEAD_DIM >= 32:
        dim_half = tl.arange(0, HEAD_DIM // 2)
        query_lo = tl.load(
            q_ptr
            + row * stride_q_row
            + (first_head + head_offsets[:, None]) * stride_q_head
            + dim_half[None, :],
            mask=head_offsets[:, None] < GROUP_SIZE,
            other=0.0,
        )
        query_hi = tl.load(
            q_ptr
            + row * stride_q_row
            + (first_head + head_offsets[:, None]) * stride_q_head
            + (dim_half[None, :] + HEAD_DIM // 2),
            mask=head_offsets[:, None] < GROUP_SIZE,
            other=0.0,
        )
    else:
        query = tl.load(
            q_ptr
            + row * stride_q_row
            + (first_head + head_offsets[:, None]) * stride_q_head
            + dim_offsets[None, :],
            mask=head_offsets[:, None] < GROUP_SIZE,
            other=0.0,
        )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if IS_FP8 and HEAD_DIM >= 32:
        accumulator_lo = tl.zeros((BLOCK_M, HEAD_DIM // 2), dtype=tl.float32)
        accumulator_hi = tl.zeros((BLOCK_M, HEAD_DIM // 2), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
            & (columns < TOPK)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # The block-stride product can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        if IS_FP8 and HEAD_DIM >= 32:
            # Accumulate QK in two head-dimension chunks so each K tile and
            # FP8 conversion temporary is half the size of the original.
            keys = tl.load(
                k_cache_ptr
                + safe_page[None, :] * stride_k_block
                + page_offset[None, :] * stride_k_token
                + kv_head * stride_k_head
                + dim_half[:, None],
                mask=valid[None, :],
                other=0.0,
            )
            if IS_FP8:
                keys = keys.to(tl.bfloat16)
            scores = tl.dot(query_lo, keys)
            keys = tl.load(
                k_cache_ptr
                + safe_page[None, :] * stride_k_block
                + page_offset[None, :] * stride_k_token
                + kv_head * stride_k_head
                + (dim_half[:, None] + HEAD_DIM // 2),
                mask=valid[None, :],
                other=0.0,
            )
            if IS_FP8:
                keys = keys.to(tl.bfloat16)
            scores = tl.dot(query_hi, keys, acc=scores)
        else:
            keys = tl.load(
                k_cache_ptr
                + safe_page[None, :] * stride_k_block
                + page_offset[None, :] * stride_k_token
                + kv_head * stride_k_head
                + dim_offsets[:, None],
                mask=valid[None, :],
                other=0.0,
            )
            if IS_FP8:
                keys = keys.to(tl.bfloat16)
            scores = tl.dot(query, keys)
        if IS_FP8:
            scores *= tl.load(k_scale_ptr)
        # Scale scores so the query is not re-quantized after scaling.
        scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        if IS_FP8 and HEAD_DIM >= 32:
            # Split the output dimension of PV to shorten each FP8 V tile.
            values = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + dim_half[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.bfloat16)
            accumulator_lo = tl.dot(
                probabilities.to(values.dtype),
                values,
                acc=accumulator_lo * alpha[:, None],
            )
            values = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + (dim_half[None, :] + HEAD_DIM // 2),
                mask=valid[:, None],
                other=0.0,
            ).to(tl.bfloat16)
            accumulator_hi = tl.dot(
                probabilities.to(values.dtype),
                values,
                acc=accumulator_hi * alpha[:, None],
            )
        else:
            values = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + dim_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            if IS_FP8:
                values = values.to(tl.bfloat16)
            accumulator = tl.dot(
                probabilities.to(values.dtype),
                values,
                acc=accumulator * alpha[:, None],
            )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    inv_normalizer = 1.0 / tl.maximum(normalizer, 1.0e-20)
    if IS_FP8 and HEAD_DIM >= 32:
        normalized_lo = tl.where(
            has_values[:, None], accumulator_lo * inv_normalizer[:, None], 0.0
        )
        normalized_hi = tl.where(
            has_values[:, None], accumulator_hi * inv_normalizer[:, None], 0.0
        )
        v_scale = tl.load(v_scale_ptr)
        normalized_lo *= v_scale
        normalized_hi *= v_scale
    else:
        normalized_output = tl.where(
            has_values[:, None],
            accumulator * inv_normalizer[:, None],
            0.0,
        )
        if IS_FP8:
            normalized_output *= tl.load(v_scale_ptr)
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        if IS_FP8 and HEAD_DIM >= 32:
            dst = (
                output_ptr
                + row * stride_output_row
                + (first_head + head_offsets[:, None]) * stride_output_head
                + dim_half[None, :]
            )
            tl.store(dst, normalized_lo, mask=output_mask)
            tl.store(dst + HEAD_DIM // 2, normalized_hi, mask=output_mask)
        else:
            tl.store(
                output_ptr
                + row * stride_output_row
                + (first_head + head_offsets[:, None]) * stride_output_head
                + dim_offsets[None, :],
                normalized_output,
                mask=output_mask,
            )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        partial_row = (split_id * num_rows + row).to(tl.int64)
        if IS_FP8 and HEAD_DIM >= 32:
            dst = (
                partial_output_ptr
                + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
                * HEAD_DIM
                + dim_half[None, :]
            )
            tl.store(dst, normalized_lo, mask=output_mask)
            tl.store(dst + HEAD_DIM // 2, normalized_hi, mask=output_mask)
        else:
            tl.store(
                partial_output_ptr
                + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
                * HEAD_DIM
                + dim_offsets[None, :],
                normalized_output,
                mask=output_mask,
            )
        tl.store(
            partial_lse_ptr + partial_row * NUM_QUERY_HEADS + first_head + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit
def _qsa_sparse_paged_gqa_splitk_clamped_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    k_scale_ptr,
    v_scale_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FP8: tl.constexpr,
    SCAN_BLOCK: tl.constexpr,
) -> None:
    """Sparse paged GQA, ported from vLLM's Apache-2.0 QSA kernel."""
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    if IS_FP8 and HEAD_DIM >= 32:
        dim_half = tl.arange(0, HEAD_DIM // 2)
        query_lo = tl.load(
            q_ptr
            + row * stride_q_row
            + (first_head + head_offsets[:, None]) * stride_q_head
            + dim_half[None, :],
            mask=head_offsets[:, None] < GROUP_SIZE,
            other=0.0,
        )
        query_hi = tl.load(
            q_ptr
            + row * stride_q_row
            + (first_head + head_offsets[:, None]) * stride_q_head
            + (dim_half[None, :] + HEAD_DIM // 2),
            mask=head_offsets[:, None] < GROUP_SIZE,
            other=0.0,
        )
    else:
        query = tl.load(
            q_ptr
            + row * stride_q_row
            + (first_head + head_offsets[:, None]) * stride_q_head
            + dim_offsets[None, :],
            mask=head_offsets[:, None] < GROUP_SIZE,
            other=0.0,
        )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if IS_FP8 and HEAD_DIM >= 32:
        accumulator_lo = tl.zeros((BLOCK_M, HEAD_DIM // 2), dtype=tl.float32)
        accumulator_hi = tl.zeros((BLOCK_M, HEAD_DIM // 2), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    # Find the last nonnegative index only within this split's own tile
    # range. Later nonempty splits do not require this split to do work.
    extent_best = tl.full((), -1, dtype=tl.int32)
    local_first_tile = split_id * NUM_TILES // NUM_SPLITS
    local_after_tile = (split_id + 1) * NUM_TILES // NUM_SPLITS
    local_first_col = local_first_tile * BLOCK_N
    local_after_col = tl.minimum(local_after_tile * BLOCK_N, TOPK)
    for extent_start in range(local_first_col, local_after_col, SCAN_BLOCK):
        extent_cols = extent_start + tl.arange(0, SCAN_BLOCK)
        extent_vals = tl.load(
            indices_ptr + row * stride_indices_row + extent_cols,
            mask=extent_cols < local_after_col,
            other=-1,
        )
        extent_best = tl.maximum(
            extent_best,
            tl.max(tl.where(extent_vals >= 0, extent_cols, -1), axis=0),
        )
    row_extent = extent_best + 1
    n_useful_tiles = tl.minimum(
        tl.maximum((row_extent + BLOCK_N - 1) // BLOCK_N, 0), NUM_TILES
    )
    split_tile_start = tl.minimum(split_id * NUM_TILES // NUM_SPLITS, n_useful_tiles)
    split_tile_end = tl.minimum(
        (split_id + 1) * NUM_TILES // NUM_SPLITS, n_useful_tiles
    )
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
            & (columns < TOPK)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # The block-stride product can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        if IS_FP8 and HEAD_DIM >= 32:
            # Accumulate QK in two head-dimension chunks so each K tile and
            # FP8 conversion temporary is half the size of the original.
            keys = tl.load(
                k_cache_ptr
                + safe_page[None, :] * stride_k_block
                + page_offset[None, :] * stride_k_token
                + kv_head * stride_k_head
                + dim_half[:, None],
                mask=valid[None, :],
                other=0.0,
            )
            if IS_FP8:
                keys = keys.to(tl.bfloat16)
            scores = tl.dot(query_lo, keys)
            keys = tl.load(
                k_cache_ptr
                + safe_page[None, :] * stride_k_block
                + page_offset[None, :] * stride_k_token
                + kv_head * stride_k_head
                + (dim_half[:, None] + HEAD_DIM // 2),
                mask=valid[None, :],
                other=0.0,
            )
            if IS_FP8:
                keys = keys.to(tl.bfloat16)
            scores = tl.dot(query_hi, keys, acc=scores)
        else:
            keys = tl.load(
                k_cache_ptr
                + safe_page[None, :] * stride_k_block
                + page_offset[None, :] * stride_k_token
                + kv_head * stride_k_head
                + dim_offsets[:, None],
                mask=valid[None, :],
                other=0.0,
            )
            if IS_FP8:
                keys = keys.to(tl.bfloat16)
            scores = tl.dot(query, keys)
        if IS_FP8:
            scores *= tl.load(k_scale_ptr)
        # Scale scores so the query is not re-quantized after scaling.
        scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        if IS_FP8 and HEAD_DIM >= 32:
            # Split the output dimension of PV to shorten each FP8 V tile.
            values = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + dim_half[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.bfloat16)
            accumulator_lo = tl.dot(
                probabilities.to(values.dtype),
                values,
                acc=accumulator_lo * alpha[:, None],
            )
            values = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + (dim_half[None, :] + HEAD_DIM // 2),
                mask=valid[:, None],
                other=0.0,
            ).to(tl.bfloat16)
            accumulator_hi = tl.dot(
                probabilities.to(values.dtype),
                values,
                acc=accumulator_hi * alpha[:, None],
            )
        else:
            values = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + dim_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            if IS_FP8:
                values = values.to(tl.bfloat16)
            accumulator = tl.dot(
                probabilities.to(values.dtype),
                values,
                acc=accumulator * alpha[:, None],
            )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    inv_normalizer = 1.0 / tl.maximum(normalizer, 1.0e-20)
    if IS_FP8 and HEAD_DIM >= 32:
        normalized_lo = tl.where(
            has_values[:, None], accumulator_lo * inv_normalizer[:, None], 0.0
        )
        normalized_hi = tl.where(
            has_values[:, None], accumulator_hi * inv_normalizer[:, None], 0.0
        )
        v_scale = tl.load(v_scale_ptr)
        normalized_lo *= v_scale
        normalized_hi *= v_scale
    else:
        normalized_output = tl.where(
            has_values[:, None],
            accumulator * inv_normalizer[:, None],
            0.0,
        )
        if IS_FP8:
            normalized_output *= tl.load(v_scale_ptr)
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        if IS_FP8 and HEAD_DIM >= 32:
            dst = (
                output_ptr
                + row * stride_output_row
                + (first_head + head_offsets[:, None]) * stride_output_head
                + dim_half[None, :]
            )
            tl.store(dst, normalized_lo, mask=output_mask)
            tl.store(dst + HEAD_DIM // 2, normalized_hi, mask=output_mask)
        else:
            tl.store(
                output_ptr
                + row * stride_output_row
                + (first_head + head_offsets[:, None]) * stride_output_head
                + dim_offsets[None, :],
                normalized_output,
                mask=output_mask,
            )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        partial_row = (split_id * num_rows + row).to(tl.int64)
        if IS_FP8 and HEAD_DIM >= 32:
            dst = (
                partial_output_ptr
                + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
                * HEAD_DIM
                + dim_half[None, :]
            )
            tl.store(dst, normalized_lo, mask=output_mask)
            tl.store(dst + HEAD_DIM // 2, normalized_hi, mask=output_mask)
        else:
            tl.store(
                partial_output_ptr
                + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
                * HEAD_DIM
                + dim_offsets[None, :],
                normalized_output,
                mask=output_mask,
            )
        tl.store(
            partial_lse_ptr + partial_row * NUM_QUERY_HEADS + first_head + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_output_row,
    stride_output_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    DIM_TILE: tl.constexpr,
) -> None:
    """Merge independently normalized QSA splits in FP32 log-sum-exp space."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    dim_tile_id = tl.program_id(2)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = dim_tile_id * DIM_TILE + tl.arange(0, DIM_TILE)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    split_rows = split_offsets.to(tl.int64) * num_rows + row
    partial_output = tl.load(
        partial_output_ptr
        + (split_rows[:, None] * NUM_QUERY_HEADS + head) * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


def _validate_sparse_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None,
) -> None:
    if not q.is_cuda:
        raise ValueError("QSA operators require CUDA/HIP tensors.")
    device = q.device
    if (
        k_cache.device != device
        or v_cache.device != device
        or logical_indices.device != device
        or block_table.device != device
        or token_to_req.device != device
        or (out is not None and out.device != device)
    ):
        raise ValueError("QSA tensors must all use the same device.")
    if q.ndim != 3 or q.shape[1] <= 0:
        raise ValueError("QSA sparse query must be [rows, heads, head_dim].")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse K/V caches must have matching rank-4 shapes.")
    if (
        not all(k_cache.shape[:3])
        or block_table.ndim != 2
        or not all(block_table.shape)
    ):
        raise ValueError("QSA sparse cache and block table must be nonempty.")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA sparse indices must have one row per query.")
    if logical_indices.shape[1] <= 0:
        raise ValueError("QSA sparse attention requires a positive selection width.")
    if token_to_req.shape != (q.shape[0],):
        raise ValueError("QSA sparse request mapping must match query rows.")
    if q.shape[2] != k_cache.shape[3]:
        raise ValueError("QSA sparse query and cache head dimensions must match.")
    if q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads.")
    if q.shape[2] < 16 or q.shape[2] & (q.shape[2] - 1):
        raise ValueError(
            "QSA sparse head dimension must be a power of two at least 16."
        )
    if q.dtype is not torch.bfloat16:
        raise ValueError("QSA sparse query must be bfloat16.")
    if k_cache.dtype is not torch.bfloat16 or v_cache.dtype is not torch.bfloat16:
        raise ValueError("QSA sparse K/V caches must be bfloat16.")
    for name, tensor in (
        ("indices", logical_indices),
        ("block table", block_table),
        ("request mapping", token_to_req),
    ):
        if tensor.dtype is not torch.int32:
            raise ValueError(f"QSA {name} must be int32.")
        if tensor.stride(-1) != 1:
            raise ValueError(f"QSA {name} must have a contiguous last dimension.")
    for name, tensor in (
        ("query", q),
        ("key cache", k_cache),
        ("value cache", v_cache),
    ):
        if tensor.stride(-1) != 1:
            raise ValueError(f"QSA {name} must have a contiguous last dimension.")
    if out is not None:
        if out.shape != q.shape:
            raise ValueError("QSA sparse output must match its query.")
        if out.dtype is not torch.bfloat16:
            raise ValueError("QSA sparse output must be bfloat16.")
        if out.stride(-1) != 1:
            raise ValueError("QSA sparse output must have a contiguous last dimension.")


def _select_qsa_sparse_block_m(group_size: int, head_dim: int) -> int:
    """Choose a matrix-core-compatible M tile for the measured HCU GQA shape."""
    # HCU exposes 16x16 BF16 MMAC.  GROUP_SIZE=6 otherwise rounds to 8,
    # which makes both QK (8xN) and PV (8xD) dots fail the MMAC legality check.
    if torch.version.hip is not None and group_size == 6 and head_dim == 256:
        return 16
    return triton.next_power_of_2(group_size)


def _select_sparse_launch_config(
    rows: int,
    kv_heads: int,
    topk: int,
    head_dim: int,
    group_size: int,
) -> dict[str, int]:
    """Select the launch policy used by the public sparse attention API."""
    block_m = _select_qsa_sparse_block_m(group_size, head_dim)
    return _select_qsa_sparse_config(rows, kv_heads, topk, head_dim, block_m)


def _launch_qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    config: dict[str, int],
    out: torch.Tensor | None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch validated sparse QSA with an explicit static configuration."""
    if out is None:
        out = torch.empty_like(q)
    if not q.shape[0]:
        return out

    is_fp8 = k_scale is not None
    scale_k = k_scale if is_fp8 else q
    scale_v = v_scale if is_fp8 else q
    group_size = q.shape[1] // k_cache.shape[2]
    block_m = _select_qsa_sparse_block_m(group_size, q.shape[2])
    block_n = config["block_n"]
    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(max_useful_splits, config["target_splits"])
    partial_warps = config["num_warps"]
    # The static table is calibrated for HIP.  CUDA retains the upstream
    # source stage so an unmeasured non-target path stays source-equivalent.
    partial_stages = config["num_stages"] if torch.version.hip is not None else 2

    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        # Split-K partials remain FP32 until the final normalized merge.
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    if q.shape[0] >= 3:
        # From three rows on, skipping fully-masked tiles outweighs their
        # in-kernel extent scan; single/double-token decode keeps the
        # constant-bound kernel.
        _qsa_sparse_paged_gqa_splitk_clamped_kernel[
            (q.shape[0], k_cache.shape[2], num_splits)
        ](
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            partial_output,
            partial_lse,
            out,
            scale_k,
            scale_v,
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            logical_indices.stride(0),
            block_table.stride(0),
            out.stride(0),
            out.stride(1),
            q.shape[0],
            k_cache.shape[0],
            block_table.shape[0],
            TOPK=logical_indices.shape[1],
            PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1],
            GROUP_SIZE=group_size,
            HEAD_DIM=q.shape[2],
            NUM_QUERY_HEADS=q.shape[1],
            NUM_SPLITS=num_splits,
            NUM_TILES=num_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            IS_FP8=is_fp8,
            SCAN_BLOCK=min(
                1024,
                triton.next_power_of_2(triton.cdiv(num_tiles, num_splits) * block_n),
            ),
            num_warps=partial_warps,
            num_stages=partial_stages,
        )
    else:
        _qsa_sparse_paged_gqa_splitk_kernel[(q.shape[0], k_cache.shape[2], num_splits)](
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            partial_output,
            partial_lse,
            out,
            scale_k,
            scale_v,
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            logical_indices.stride(0),
            block_table.stride(0),
            out.stride(0),
            out.stride(1),
            q.shape[0],
            k_cache.shape[0],
            block_table.shape[0],
            TOPK=logical_indices.shape[1],
            PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1],
            GROUP_SIZE=group_size,
            HEAD_DIM=q.shape[2],
            NUM_QUERY_HEADS=q.shape[1],
            NUM_SPLITS=num_splits,
            NUM_TILES=num_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            IS_FP8=is_fp8,
            num_warps=partial_warps,
            num_stages=partial_stages,
        )
    if num_splits == 1:
        return out

    merge_dim_tile = config["merge_dim_tile"]
    if (
        merge_dim_tile <= 0
        or merge_dim_tile > q.shape[2]
        or q.shape[2] % merge_dim_tile
    ):
        # Non-divisor tiles keep the full-head source layout.
        merge_dim_tile = q.shape[2]
    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1], q.shape[2] // merge_dim_tile)](
        partial_output,
        partial_lse,
        out,
        out.stride(0),
        out.stride(1),
        q.shape[0],
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        DIM_TILE=merge_dim_tile,
        num_warps=config["merge_warps"],
        num_stages=1,
    )
    return out


def _qsa_sparse_paged_attention_with_config(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    config: dict[str, int],
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sparse QSA using a caller-supplied config for benchmark dispatch."""
    return _launch_qsa_sparse_paged_attention(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req, config, out
    )


def _native_qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sparse grouped-query attention over paged BF16 K/V caches."""
    _validate_sparse_inputs(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req, out
    )
    if not q.shape[0]:
        return out if out is not None else torch.empty_like(q)

    config = _select_recorded_launch_config(q, k_cache, logical_indices)
    return _launch_qsa_sparse_paged_attention(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req, config, out
    )


def _native_qsa_sparse_paged_attention_fp8(
    q,
    k_cache,
    v_cache,
    logical_indices,
    block_table,
    token_to_req,
    out=None,
    k_scale=None,
    v_scale=None,
    *,
    kv_cache_dtype=None,
):
    """Native FP8 KV; full Hq6 E5M2 result in experiment 940; E4M3FN not full-validated."""
    if (
        k_cache.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
        or k_cache.dtype != v_cache.dtype
    ):
        raise ValueError("matching E4M3FN/E5M2 K/V required")
    for scale in (k_scale, v_scale):
        if (
            scale is None
            or scale.device != q.device
            or scale.dtype != torch.float32
            or scale.numel() != 1
        ):
            raise ValueError("device FP32 scalar K/V scale required")
    # Reuse existing metadata validation without materializing a BF16 cache.
    # The stride-preserving dtype views are only validation proxies.
    if q.dtype != torch.bfloat16 or not q.is_cuda or q.ndim != 3:
        raise ValueError("BF16 GPU Q with three dimensions required")
    if (
        k_cache.ndim != 4
        or k_cache.shape != v_cache.shape
        or q.shape[2] != k_cache.shape[3]
        or q.shape[1] % k_cache.shape[2]
    ):
        raise ValueError("invalid Q/K/V shapes")
    if (
        logical_indices.ndim != 2
        or logical_indices.shape[0] != q.shape[0]
        or logical_indices.shape[1] <= 0
        or block_table.ndim != 2
        or not all(block_table.shape)
        or token_to_req.shape != (q.shape[0],)
    ):
        raise ValueError("invalid metadata shapes")
    if any(
        t.device != q.device
        for t in (k_cache, v_cache, logical_indices, block_table, token_to_req)
    ):
        raise ValueError("device mismatch")
    if any(
        t.dtype != torch.int32 or t.stride(-1) != 1
        for t in (logical_indices, block_table, token_to_req)
    ):
        raise ValueError("metadata must be contiguous int32")
    if any(t.stride(-1) != 1 for t in (q, k_cache, v_cache)):
        raise ValueError("last dimensions must be contiguous")
    if out is not None and (
        out.shape != q.shape
        or out.dtype != q.dtype
        or out.device != q.device
        or out.stride(-1) != 1
    ):
        raise ValueError("invalid output")
    if not q.shape[0]:
        return out if out is not None else torch.empty_like(q)
    config = _select_recorded_launch_config(q, k_cache, logical_indices)
    return _launch_qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        config,
        out,
        k_scale,
        v_scale,
    )


def cfg_for(t, dtype):
    cfg = dict(_select_sparse_launch_config(t, 1, 2051, 256, 6))
    # Frozen before this sweep; non-listed token counts keep source defaults.
    if t == 16:
        cfg.update(block_n=32, target_splits=16 if dtype == "bf16" else 32, num_warps=2)
    elif t == 32:
        cfg.update(block_n=32, target_splits=8, num_warps=2)
    elif t == 64:
        cfg.update(block_n=32, target_splits=4 if dtype == "bf16" else 8, num_warps=2)
    elif t == 128:
        cfg.update(
            block_n=32,
            target_splits=4 if dtype == "bf16" else 8,
            num_warps=4 if dtype == "bf16" else 2,
        )
    elif t == 512 and dtype == "bf16":
        cfg.update(block_n=16, target_splits=1)
    elif t == 512:
        cfg.update(block_n=32, target_splits=4, num_warps=2)
    return cfg


def new_cfg_for(t, dtype):
    cfg = cfg_for(t, dtype)
    # Frozen from experiment 923 before this validation; the 32, 64, and
    # 128-token measured settings remain unchanged. This is a hypothesis,
    # not a claim of optimality for unmeasured intermediate token counts.
    if 24 <= t <= 160 and t not in (32, 64, 128):
        splits = (8 if t <= 96 else 2) if dtype == "bf16" else 16
        cfg.update(block_n=32, target_splits=splits, num_warps=2)
    return cfg


def experiment_cfg_for(t, dtype):
    # Frozen before representative pilot and 64-record validation.
    # Interval extension beyond measured T192/T384 is a hypothesis.
    cfg = new_cfg_for(t, dtype)
    if dtype == "bf16" and t == 128:
        cfg.update(block_n=32, target_splits=2, num_warps=2)
    elif dtype == "bf16" and t == 192:
        cfg.update(block_n=32, target_splits=8, num_warps=2)
    elif dtype == "fp8_e5m2" and t == 2:
        cfg.update(block_n=16, target_splits=64, num_warps=4)
    elif dtype == "fp8_e5m2" and 160 < t <= 256:
        cfg.update(block_n=32, target_splits=8, num_warps=2)
    elif dtype == "fp8_e5m2" and 256 < t < 512:
        cfg.update(block_n=32, target_splits=4, num_warps=2)
    return cfg


def candidate_cfg_for(t, dtype):
    # Freeze before pilot and 64-record validation. Extending T320/T448
    # choices to the whole 257..511 interval remains a hypothesis.
    cfg = experiment_cfg_for(t, dtype)
    if t == 4:
        cfg.update(block_n=32, target_splits=32, num_warps=4)
    elif dtype == "bf16" and t == 8:
        cfg.update(block_n=32, target_splits=16, num_warps=4)
    elif dtype == "fp8_e5m2" and 256 < t < 512:
        cfg.update(block_n=32, target_splits=8, num_warps=2)
    return cfg


def candidate940_cfg_for(t, dtype):
    # Frozen ahead of pilot and 64-record validation; 2369 and the full
    # 513..2368 interval are unmeasured extrapolations from experiment 938.
    cfg = candidate_cfg_for(t, dtype)
    if t >= 2369:
        cfg.update(block_n=32, target_splits=1, num_warps=2)
    elif dtype == "fp8_e5m2" and 512 < t < 2369:
        cfg.update(block_n=32, target_splits=4, num_warps=2)
    return cfg


def _select_recorded_launch_config(q, k_cache, logical_indices):
    rows, heads, dim = q.shape
    kv_heads = k_cache.shape[2]
    group = heads // kv_heads
    if (kv_heads, logical_indices.shape[1], dim, group) == (1, 2051, 256, 6):
        dtype = "bf16" if k_cache.dtype == torch.bfloat16 else "fp8_e5m2"
        return candidate940_cfg_for(rows, dtype)
    return _select_sparse_launch_config(
        rows, kv_heads, logical_indices.shape[1], dim, group
    )


@triton.jit
def _unpack_qsa_indices_kernel(
    packed_ptr,
    indices_ptr,
    scales_ptr,
    packed_row_stride,
    k_scale,
    v_scale,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    FP8: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    count = tl.load(packed_ptr + row * packed_row_stride + WIDTH)
    count = tl.minimum(tl.maximum(count, 0), WIDTH)
    indices = tl.load(
        packed_ptr + row * packed_row_stride + columns,
        mask=(columns < WIDTH) & (columns < count),
        other=-1,
    )
    tl.store(indices_ptr + row * WIDTH + columns, indices, mask=columns < WIDTH)
    if FP8:
        if row == 0:
            tl.store(scales_ptr, k_scale)
            tl.store(scales_ptr + 1, v_scale)


@triton.jit
def _qsa_output_gate_kernel(
    raw_ptr,
    gate_ptr,
    out_ptr,
    raw_row_stride,
    raw_head_stride,
    out_row_stride,
    out_head_stride,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    head, dim = offsets // DIM, offsets % DIM
    valid = offsets < HEADS * DIM
    raw = tl.load(
        raw_ptr + row * raw_row_stride + head * raw_head_stride + dim,
        mask=valid,
        other=0,
    ).to(tl.float32)
    gate = tl.load(gate_ptr + row * HEADS * DIM + offsets, mask=valid, other=0).to(
        tl.float32
    )
    tl.store(
        out_ptr + row * out_row_stride + head * out_head_stride + dim,
        raw * tl.sigmoid(gate),
        mask=valid,
    )


def _validate_complete_inputs(
    q, k_cache, v_cache, logical_indices, block_table, token_to_req, out, output_gate
):
    if runtime.device.vendor_name != "hygon":
        raise NotImplementedError("This frozen QSA implementation targets Hygon HIP")
    if q.ndim != 3 or q.shape[1] <= 0 or k_cache.ndim != 4:
        raise ValueError("QSA requires rank-3 Q and rank-4 K/V")
    if v_cache.shape != k_cache.shape or not all(k_cache.shape[:3]):
        raise ValueError("QSA requires matching, nonempty K/V cache dimensions")
    dim = q.shape[2]
    if dim < 16 or dim & (dim - 1) or dim != k_cache.shape[3]:
        raise ValueError("QSA head dimension must match and be a power of two >= 16")
    if q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA requires grouped-query heads divisible by KV heads")
    if q.dtype != torch.bfloat16 or k_cache.dtype != v_cache.dtype:
        raise ValueError("QSA requires BF16 Q and matching K/V dtypes")
    if k_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        raise NotImplementedError("QSA supports BF16, E4M3FN and E5M2 KV only")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if logical_indices.shape[1] < 2:
        raise ValueError("QSA packed indices need selection columns plus a count")
    if block_table.ndim != 2 or not all(block_table.shape):
        raise ValueError("QSA block table must be a nonempty matrix")
    if token_to_req.shape != (q.shape[0],):
        raise ValueError("QSA request mapping must match query rows")
    tensors = (
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        output_gate,
    )
    if out is not None:
        tensors += (out,)
    if any(t.device != q.device or not t.is_cuda for t in tensors):
        raise ValueError("All QSA tensors must reside on the same HCU")
    for tensor in (logical_indices, block_table, token_to_req):
        if tensor.dtype != torch.int32 or tensor.stride(-1) != 1:
            raise ValueError(
                "QSA metadata must be int32 with contiguous last dimension"
            )
    if any(t.stride(-1) != 1 for t in (q, k_cache, v_cache)):
        raise ValueError("QSA Q/K/V last dimensions must be contiguous")
    if not output_gate.is_contiguous() or output_gate.numel() != q.numel():
        raise ValueError("QSA output gate must be contiguous and reshapeable to Q")
    if out is not None and (
        out.shape != q.shape or out.dtype != q.dtype or out.stride(-1) != 1
    ):
        raise ValueError("QSA output must match Q with contiguous last dimension")


def _with_native_backend(
    q,
    k_cache,
    v_cache,
    logical_indices,
    block_table,
    token_to_req,
    use_prefill_config,
    out=None,
    k_scale=None,
    v_scale=None,
    *,
    output_gate,
    native_bf16,
    native_fp8,
):
    """Shared adapter for candidate and unchanged vendor benchmark functions."""
    _validate_complete_inputs(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        out,
        output_gate,
    )
    # This is a launch hint, not a semantic choice. Freeze both to experiment 940.
    del use_prefill_config
    is_fp8 = k_cache.dtype != torch.bfloat16
    if is_fp8:
        if not all(isinstance(s, (float, int)) for s in (k_scale, v_scale)):
            raise TypeError("FP8 k_scale/v_scale must be host scalar floats")
        ks, vs = float(k_scale), float(v_scale)
    else:
        ks, vs = 1.0, 1.0
    if out is None:
        out = torch.empty_like(q)
    if q.shape[0] == 0:
        return out
    width = logical_indices.shape[1] - 1
    indices = torch.empty((q.shape[0], width), device=q.device, dtype=torch.int32)
    scales = torch.empty((2,), device=q.device, dtype=torch.float32) if is_fp8 else q
    _unpack_qsa_indices_kernel[(q.shape[0],)](
        logical_indices,
        indices,
        scales,
        logical_indices.stride(0),
        ks,
        vs,
        WIDTH=width,
        BLOCK=triton.next_power_of_2(width),
        FP8=is_fp8,
    )
    # Separate storage preserves gate contents when out aliases the gate and
    # avoids overwriting Q before native attention has consumed it.
    raw = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    args = (q, k_cache, v_cache, indices, block_table, token_to_req)
    if is_fp8:
        native_fp8(*args, out=raw, k_scale=scales[:1], v_scale=scales[1:])
    else:
        native_bf16(*args, out=raw)
    _qsa_output_gate_kernel[(q.shape[0], triton.cdiv(q.shape[1] * q.shape[2], 256))](
        raw,
        output_gate,
        out,
        raw.stride(0),
        raw.stride(1),
        out.stride(0),
        out.stride(1),
        HEADS=q.shape[1],
        DIM=q.shape[2],
        BLOCK=256,
    )
    return out


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    use_prefill_config: bool,
    out: torch.Tensor | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    *,
    output_gate: torch.Tensor,
) -> torch.Tensor:
    """vLLM v0.32.0 call contract; E5M2 KV is a Hygon-specific extension.

    Count conversion, native attention and sigmoid gate are all part of this
    entry. Complete-entry HCU correctness/performance remain unmeasured.
    """
    return _with_native_backend(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        use_prefill_config,
        out,
        k_scale,
        v_scale,
        output_gate=output_gate,
        native_bf16=_native_qsa_sparse_paged_attention,
        native_fp8=_native_qsa_sparse_paged_attention_fp8,
    )
