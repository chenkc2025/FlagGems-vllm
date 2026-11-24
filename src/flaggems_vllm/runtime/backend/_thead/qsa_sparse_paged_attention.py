# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors
#
# T-Head adaptation of vLLM QSA (upstream commit 8b10bda57ceab3a741d2fb3718af48cef691023e,
# BF16 precursor f47792af7f) and configuration derived from vLLM-SAIL commit
# 72cbab3b7bc4db60793098eec9bde4edc5f4ec25. Modified: PPU launch policy,
# split/merge scheduling, delayed V loads, BF16/FP8 kernels and dispatch.
"""T-Head Triton implementation of vLLM sparse paged attention."""
from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

_BF16_CONTIGUOUS_K = True
_BF16_STAGES = 2


def _sail_config(num_rows, num_kv_heads, use_prefill_config, num_columns, is_fp8=False):
    programs = num_rows * num_kv_heads
    if programs <= 4:
        block_n, target_splits, num_warps = 16, 64, 4
    elif programs < 32:
        block_n, target_splits, num_warps = 16, 32, 4
    elif programs <= 256:
        block_n, target_splits, num_warps = 64, 8, 2
    elif programs <= 512:
        block_n, target_splits, num_warps = 64, 4, 2
    else:
        block_n, target_splits, num_warps = 64, 1, 2
    tiles = (num_columns + block_n - 1) // block_n
    return block_n, num_warps, tiles, min(target_splits, tiles)


def _bf16_select_config(
    num_rows, num_kv_heads, use_prefill_config, num_columns, is_fp8=False
):
    if is_fp8:
        raise NotImplementedError("BF16 configuration only")
    splits = 8 if num_rows <= 128 else (4 if num_rows <= 512 else 1)
    tiles = (num_columns + 31) // 32
    return 32, (4 if num_rows <= 512 else 2), tiles, min(splits, tiles)


def _bf16_select_n16(rows, num_kv_heads, use_prefill_config, num_columns, is_fp8=False):
    if is_fp8:
        raise NotImplementedError("BF16 configuration only")
    splits = 4 if rows == 512 else 1
    tiles = (num_columns + 15) // 16
    return 16, 2, tiles, min(splits, tiles)


def _selected_shape(rows, blocks, requests):
    return (rows, blocks, requests) in ((512, 512, 512), (6177, 2712, 57)) or (
        rows == 16384 and blocks == 2712 and 1 <= requests <= 57
    )


def _eligible_bf16_n16(q, k, v, indices, table, request):
    # Return False for other shapes/layouts so 342.py retains its own validation.
    if q.ndim != 3 or q.shape[0] not in (512, 6177, 16384) or q.shape[1:] != (6, 256):
        return False
    if (
        k.ndim != 4
        or k.shape[1:] != (784, 1, 256)
        or v.shape != k.shape
        or indices.shape != (q.shape[0], 2052)
        or table.ndim != 2
        or table.shape[1] != 88
        or request.shape != (q.shape[0],)
    ):
        return False
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if (
        tuple(q.stride()) != (1536, 256, 1)
        or tuple(k.stride()) != (401408, 512, 512, 1)
        or tuple(v.stride()) != tuple(k.stride())
        or tuple(indices.stride()) != (2052, 1)
        or tuple(table.stride()) != (88, 1)
        or tuple(request.stride()) != (1,)
    ):
        return False
    return _selected_shape(q.shape[0], k.shape[0], table.shape[0])


@triton.jit
def _bf16_small_split(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    output_ptr,
    stats_max_ptr,
    stats_sum_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_v_dim,
    stride_indices_row,
    stride_indices_column,
    stride_table_req,
    stride_table_page,
    stride_output_row,
    stride_output_head,
    stride_output_dim,
    num_rows,
    num_cache_blocks,
    num_requests,
    softmax_scale,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    valid_count = tl.maximum(
        0,
        tl.minimum(
            TOPK,
            tl.load(
                indices_ptr + row * stride_indices_row + TOPK * stride_indices_column
            ),
        ),
    )
    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :] * stride_q_dim,
        mask=(head_offsets[:, None] < GROUP_SIZE) & (dim_offsets[None, :] < HEAD_DIM),
        other=0.0,
    )
    query = (query * softmax_scale * 1.4426950408889634).to(query.dtype)

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    column_offsets = tl.arange(0, BLOCK_N)

    for start in tl.range(
        split * CHUNK,
        tl.minimum(tl.minimum((split + 1) * CHUNK, TOPK), valid_count),
        BLOCK_N,
    ):
        columns = start + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns * stride_indices_column,
            mask=columns < TOPK,
            other=-1,
        )
        logical_page = tl.maximum(logical_token, 0) // PAGE_SIZE
        page_offset = tl.maximum(logical_token, 0) % PAGE_SIZE
        valid = (
            (row < num_rows)
            & (request >= 0)
            & (request < num_requests)
            & (columns < valid_count)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + tl.minimum(tl.maximum(request, 0), num_requests - 1) * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None] * stride_k_dim,
            mask=(dim_offsets[:, None] < HEAD_DIM) & valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :] * stride_v_dim,
            mask=valid[:, None] & (dim_offsets[None, :] < HEAD_DIM),
            other=0.0,
        )
        scores = tl.dot(query, keys)
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    base = ((row * NUM_KV_HEADS + kv_head) * SPLITS + split) * BLOCK_M
    tl.store(
        stats_max_ptr + base + head_offsets, max_value, mask=head_offsets < GROUP_SIZE
    )
    tl.store(
        stats_sum_ptr + base + head_offsets, normalizer, mask=head_offsets < GROUP_SIZE
    )
    tl.store(
        output_ptr + (base + head_offsets[:, None]) * HEAD_DIM + dim_offsets[None, :],
        accumulator,
        mask=(head_offsets[:, None] < GROUP_SIZE) & (dim_offsets[None, :] < HEAD_DIM),
    )


@triton.jit
def _bf16_small_merge(
    partial,
    maxima,
    norms,
    out,
    gate,
    stride_row,
    stride_head,
    stride_dim,
    GROUP: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // GROUP
    head_in_group = head % GROUP
    s = tl.arange(0, BLOCK_S)
    d = tl.arange(0, BLOCK_D)
    base = ((row * KV_HEADS + kv_head) * SPLITS + s) * BLOCK_M + head_in_group
    norm = tl.load(norms + base, mask=s < SPLITS, other=0.0)
    maximum = tl.load(maxima + base, mask=s < SPLITS, other=-1.0e20)
    maximum = tl.where(norm > 0.0, maximum, -1.0e20)
    global_max = tl.max(maximum, axis=0)
    weight = tl.where(norm > 0.0, tl.math.exp2(maximum - global_max), 0.0)
    denominator = tl.sum(norm * weight, axis=0)
    values = tl.load(
        partial + base[:, None] * HEAD_DIM + d[None, :],
        mask=(s[:, None] < SPLITS) & (d[None, :] < HEAD_DIM),
        other=0.0,
    )
    numerator = tl.sum(values * weight[:, None], axis=0)
    result = tl.where(
        denominator > 0.0, numerator / tl.maximum(denominator, 1.0e-20), 0.0
    )
    gate_value = tl.load(
        gate + (row * (GROUP * KV_HEADS) + head) * HEAD_DIM + d,
        mask=d < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    result = result.to(out.dtype.element_ty).to(tl.float32) * tl.sigmoid(gate_value)
    tl.store(
        out + row * stride_row + head * stride_head + d * stride_dim,
        result,
        mask=d < HEAD_DIM,
    )


def _bf16_small_attention(
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
):
    if not k_cache.shape[0] or not block_table.shape[0]:
        raise ValueError("Cache and request table must be nonempty")
    tensors = (
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        output_gate,
    )
    if any(x.device != q.device for x in tensors):
        raise ValueError("All tensors must be on the query device")
    if any(
        x.dtype != torch.int32 for x in (logical_indices, block_table, token_to_req)
    ):
        raise ValueError("Indices, block table and request mapping must be int32")
    if not output_gate.is_contiguous() or output_gate.numel() != q.numel():
        raise ValueError(
            "Gate must be contiguous and contain one value per query element"
        )
    gate = output_gate.view_as(q)
    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
        raise ValueError("Output shape, dtype and device must match query")
    t = q.shape[0]
    if t == 0:
        return out
    split = 8
    partial = torch.empty((t, 1, split, 16, 256), device=q.device, dtype=torch.float32)
    maxima = torch.empty((t, 1, split, 16), device=q.device, dtype=torch.float32)
    norms = torch.empty_like(maxima)
    common = (
        *q.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *logical_indices.stride(),
        *block_table.stride(),
        *out.stride(),
        t,
        k_cache.shape[0],
        block_table.shape[0],
        1.0 / 16.0,
    )
    _bf16_small_split[(t, 1, split)](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        partial,
        maxima,
        norms,
        *common,
        TOPK=2051,
        PAGE_SIZE=784,
        PAGE_TABLE_WIDTH=88,
        NUM_KV_HEADS=1,
        GROUP_SIZE=6,
        HEAD_DIM=256,
        BLOCK_M=16,
        BLOCK_N=32,
        BLOCK_D=256,
        SPLITS=split,
        CHUNK=288,
        num_warps=4,
        num_stages=2,
    )
    _bf16_small_merge[(t, 6)](
        partial,
        maxima,
        norms,
        out,
        gate,
        *out.stride(),
        GROUP=6,
        KV_HEADS=1,
        HEAD_DIM=256,
        BLOCK_M=16,
        SPLITS=split,
        BLOCK_S=split,
        BLOCK_D=256,
        num_warps=4,
    )
    return out


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _bf16_qsa_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    softmax_scale,
    output_scale,
    output_gate_ptr,
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
    stride_output_gate_row,
    stride_output_gate_head,
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
    CONTIGUOUS_K: tl.constexpr = False,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    # The packed selection buffer carries one TRAILING COUNT COLUMN per row
    # (column TOPK of a TOPK+1-wide buffer): the row's valid-entry count,
    # written by the expand kernel. It is never a token index — the tile loop
    # and the index load below only ever cover columns [0, TOPK).
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
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
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    # softmax_scale is the host-side attention scale (1/sqrt(head_dim), with the
    # fp8 K dequant scale already pre-multiplied in); convert to log2 units once
    # here for the exp2-based online softmax.
    score_scale = softmax_scale * 1.4426950408889634

    tile_end = tl.minimum(NUM_TILES, tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N))

    for tile in range(split_id, tile_end, NUM_SPLITS):
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
        )
        if (tile + 1) * BLOCK_N > valid_count:
            valid &= columns < valid_count
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        if CONTIGUOUS_K:
            # Load contiguous head dimensions before forming the QK operand.
            keys = tl.load(
                k_cache_ptr
                + safe_page[:, None] * stride_k_block
                + page_offset[:, None] * stride_k_token
                + kv_head * stride_k_head
                + dim_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            keys = tl.trans(keys)
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
            # e4m3 -> Q dtype is exact; keep the QK dot in Q's dtype (fp8 QK
            # measured slower here and less accurate).
            keys = keys.to(query.dtype)
        scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16; for fp8
        # caches the K dequant scale is already folded into softmax_scale on the
        # host.
        scores *= score_scale
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
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
            # Dequant V to fp16 (not bf16) for the PV dot: P <= 1 (online
            # softmax) so fp16 has the range, its wider mantissa is more
            # accurate, and the fp8->fp16 upcast with an fp16 PV dot is faster.
            values = values.to(tl.float16)
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    # Fold the fp8 V dequant scale (output_scale, 1.0 for bf16) into a per-row
    # reciprocal normalizer, so the output is a per-row multiply rather than a
    # HEAD_DIM-wide scale. The split-K LSE below keeps the unscaled normalizer,
    # so the merge stays correct.
    inv_normalizer = output_scale / tl.maximum(normalizer, 1.0e-20)
    normalized_output = tl.where(
        has_values[:, None],
        accumulator * inv_normalizer[:, None],
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        # Preserve the unfused path's BF16 attention-output rounding before
        # applying the gate in FP32.
        normalized_output = normalized_output.to(output_ptr.dtype.element_ty)
        output_gate = tl.load(
            output_gate_ptr
            + row * stride_output_gate_row
            + (first_head + head_offsets[:, None]) * stride_output_gate_head
            + dim_offsets[None, :],
            mask=output_mask,
            other=0.0,
        ).to(tl.float32)
        normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(output_gate)
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


@triton.jit(do_not_specialize=["num_rows"])
def _bf16_merge_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    output_gate_ptr,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
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
    # Preserve the unfused path's BF16 attention-output rounding before
    # applying the gate in FP32.
    merged = merged.to(output_ptr.dtype.element_ty)
    output_gate = tl.load(
        output_gate_ptr
        + row * stride_output_gate_row
        + head * stride_output_gate_head
        + dim_offsets
    ).to(tl.float32)
    merged = merged.to(tl.float32) * tl.sigmoid(output_gate)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


def _bf16_attention(
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
    """Run sparse GQA directly over paged BF16 or FP8-e4m3 K/V caches.

    With fp8 caches, k_scale/v_scale are the layer's per-tensor dequant scales
    as host floats (e.g. layer._k_scale_float/_v_scale_float): k_scale is
    pre-multiplied into the softmax scale and v_scale becomes the kernel's
    output scale, so the kernel needs no device scale buffers.

    logical_indices is the PACKED selection buffer: [rows, selection_width + 1]
    with the trailing column holding each row's valid-entry count (written by
    the expand kernel; never a token index). The kernel reads it as the
    tile-loop bound. use_prefill_config only steers the top of the config table; see
    _select_config.
    """
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA sparse attention cache and block table must be nonempty")
    if logical_indices.shape[1] < 2:
        raise ValueError(
            "QSA packed indices need selection columns plus the count column"
        )
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    assert q.dtype == torch.bfloat16
    assert k_cache.dtype == v_cache.dtype
    is_fp8 = k_cache.dtype == torch.float8_e4m3fn
    if is_fp8:
        assert k_scale is not None and v_scale is not None
        # Host pre-multiply: fold the K dequant scale into the attention scale
        # and pass V's dequant scale as the kernel's output scale.
        softmax_scale = (head_dim**-0.5) * float(k_scale)
        output_scale = float(v_scale)
    else:
        assert k_cache.dtype == torch.bfloat16
        softmax_scale = head_dim**-0.5
        output_scale = 1.0
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    assert q.device == k_cache.device == v_cache.device
    assert q.device == logical_indices.device == block_table.device
    assert q.device == token_to_req.device
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1

    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA sparse output must match its query")
    assert out.dtype == q.dtype and out.device == q.device
    assert out.stride(2) == 1
    assert output_gate.is_contiguous()
    output_gate_view = output_gate.view_as(q)
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = max(16, triton.next_power_of_2(group_size))
    selection_width = logical_indices.shape[1] - 1  # trailing column is the count
    block_n, partial_warps, num_tiles, num_splits = (
        _bf16_select_n16
        if _eligible_bf16_n16(
            q, k_cache, v_cache, logical_indices, block_table, token_to_req
        )
        else _bf16_select_config
    )(q.shape[0], k_cache.shape[2], use_prefill_config, selection_width, is_fp8)

    # Split=1 writes output directly and compiles out all workspace accesses.
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        # FP32 partials preserve accuracy when merging independently normalized
        # splits.
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
    _bf16_qsa_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        softmax_scale,
        output_scale,
        output_gate_view,
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
        output_gate_view.stride(0),
        output_gate_view.stride(1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        TOPK=selection_width,
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
        CONTIGUOUS_K=_BF16_CONTIGUOUS_K and not is_fp8,
        num_warps=partial_warps,
        num_stages=_BF16_STAGES,
    )
    if num_splits == 1:
        return out

    _bf16_merge_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        output_gate_view,
        out.stride(0),
        out.stride(1),
        output_gate_view.stride(0),
        output_gate_view.stride(1),
        q.shape[0],
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=2,
        num_stages=1,
    )
    return out


@triton.jit
def _maybe_dequant(tile):
    if tile.dtype == tl.uint8:
        # e4m3 bytes below sm_89 (no Triton fp8e4nv pointer type): sign to bit
        # 15, exp+mantissa to bits 13..7 gives the fp16 of x * 2^-8. Exact for
        # every finite code; the two NaN codes decode to +-480 after the host
        # scale instead of NaN. Decodes four bytes per 32-bit register: the
        # rotate moves each byte's bits 6..1 to 5..0 and its bit 0 to bit 7 of
        # the byte below, lop3 restores the sign bits, prmt pairs high and low
        # bytes, and the mask clears the sign copy at bit 14 and bits 6..0.
        # pack=4 puts element k in byte k of $2 and half k of ($0, $1), the
        # same lane mapping as Triton's own sm<89 fp8 -> fp16 asm.
        return tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b32 rot, high;
            shf.r.wrap.b32 rot, $2, $2, 1;
            lop3.b32 high, $2, rot, 0x80808080, 0xE4;
            prmt.b32 $0, rot, high, 0x5043;
            prmt.b32 $1, rot, high, 0x7261;
            and.b32 $0, $0, 0xBF80BF80;
            and.b32 $1, $1, 0xBF80BF80;
            }
            """,
            constraints="=r,=r,r",
            args=[tile],
            dtype=tl.float16,
            is_pure=True,
            pack=4,
        )
    elif tile.dtype == tl.float8e4nv:
        return tile.to(tl.float16)
    else:
        return tile


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _fp8_qsa_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    softmax_scale,
    output_scale,
    output_gate_ptr,
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
    stride_output_gate_row,
    stride_output_gate_head,
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
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    # The packed selection buffer carries one TRAILING COUNT COLUMN per row
    # (column TOPK of a TOPK+1-wide buffer): the row's valid-entry count,
    # written by the expand kernel. It is never a token index — the tile loop
    # and the index load below only ever cover columns [0, TOPK).
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    if HEAD_DIM == 256:
        half_dim = tl.arange(0, 128)
        query0 = tl.load(
            q_ptr
            + row * stride_q_row
            + (first_head + head_offsets[:, None]) * stride_q_head
            + half_dim[None, :],
            mask=head_offsets[:, None] < GROUP_SIZE,
            other=0.0,
        )
        query1 = tl.load(
            q_ptr
            + row * stride_q_row
            + (first_head + head_offsets[:, None]) * stride_q_head
            + half_dim[None, :]
            + 128,
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
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    # softmax_scale is the host-side attention scale (1/sqrt(head_dim), with the
    # fp8 K dequant scale already pre-multiplied in); convert to log2 units once
    # here for the exp2-based online softmax.
    score_scale = softmax_scale * 1.4426950408889634

    tile_end = tl.minimum(NUM_TILES, tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N))

    for tile in range(split_id, tile_end, NUM_SPLITS):
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
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        if HEAD_DIM == 256:
            key_ptr = (
                k_cache_ptr
                + safe_page[None, :] * stride_k_block
                + page_offset[None, :] * stride_k_token
                + kv_head * stride_k_head
                + half_dim[:, None]
            )
            keys0 = _maybe_dequant(tl.load(key_ptr, mask=valid[None, :], other=0.0)).to(
                query0.dtype
            )
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
            # e4m3 -> Q dtype is exact; keep the QK dot in Q's dtype (fp8 QK
            # measured slower here and less accurate).
            keys = _maybe_dequant(keys).to(query.dtype)
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        # Dequant V to fp16 (not bf16) for the PV dot: P <= 1 (online
        # softmax) so fp16 has the range, its wider mantissa is more
        # accurate, and the fp8->fp16 upcast with an fp16 PV dot is faster.
        values = _maybe_dequant(values)
        if HEAD_DIM == 256:
            scores = tl.dot(query0, keys0)
            keys1 = _maybe_dequant(
                tl.load(key_ptr + 128, mask=valid[None, :], other=0.0)
            ).to(query1.dtype)
            scores = tl.dot(query1, keys1, acc=scores)
        else:
            scores = tl.dot(query, keys)
            # Scaling scores avoids re-quantizing a scaled query to BF16; for fp8
            # caches the K dequant scale is already folded into softmax_scale on the
            # host.
        scores *= score_scale
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    # Fold the fp8 V dequant scale (output_scale, 1.0 for bf16) into a per-row
    # reciprocal normalizer, so the output is a per-row multiply rather than a
    # HEAD_DIM-wide scale. The split-K LSE below keeps the unscaled normalizer,
    # so the merge stays correct.
    inv_normalizer = output_scale / tl.maximum(normalizer, 1.0e-20)
    normalized_output = tl.where(
        has_values[:, None],
        accumulator * inv_normalizer[:, None],
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        # Preserve the unfused path's BF16 attention-output rounding before
        # applying the gate in FP32.
        normalized_output = normalized_output.to(output_ptr.dtype.element_ty)
        output_gate = tl.load(
            output_gate_ptr
            + row * stride_output_gate_row
            + (first_head + head_offsets[:, None]) * stride_output_gate_head
            + dim_offsets[None, :],
            mask=output_mask,
            other=0.0,
        ).to(tl.float32)
        normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(output_gate)
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


@triton.jit(do_not_specialize=["num_rows"])
def _fp8_merge_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    output_gate_ptr,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
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
    # Preserve the unfused path's BF16 attention-output rounding before
    # applying the gate in FP32.
    merged = merged.to(output_ptr.dtype.element_ty)
    output_gate = tl.load(
        output_gate_ptr
        + row * stride_output_gate_row
        + head * stride_output_gate_head
        + dim_offsets
    ).to(tl.float32)
    merged = merged.to(tl.float32) * tl.sigmoid(output_gate)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


def _fp8_attention(
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
    """Run sparse GQA directly over paged BF16 or FP8-e4m3 K/V caches.

    An fp8 cache is a native e4m3 view on sm_89+ and the raw uint8 storage
    below sm_89 (no Triton fp8e4nv pointer type there); the kernel dequantizes
    either on load. k_scale/v_scale are the layer's per-tensor dequant scales
    as host floats (e.g. layer._k_scale_float/_v_scale_float): k_scale is
    pre-multiplied into the softmax scale and v_scale becomes the kernel's
    output scale, so the kernel needs no device scale buffers.

    logical_indices is the PACKED selection buffer: [rows, selection_width + 1]
    with the trailing column holding each row's valid-entry count (written by
    the expand kernel; never a token index). The kernel reads it as the
    tile-loop bound. use_prefill_config only steers the top of the config table; see
    _select_config.
    """
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA sparse attention cache and block table must be nonempty")
    if logical_indices.shape[1] < 2:
        raise ValueError(
            "QSA packed indices need selection columns plus the count column"
        )
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    assert q.dtype == torch.bfloat16
    assert k_cache.dtype == v_cache.dtype
    is_fp8 = k_cache.dtype != torch.bfloat16
    if is_fp8:
        assert k_scale is not None and v_scale is not None
        # Host pre-multiply: fold the K dequant scale into the attention scale
        # and pass V's dequant scale as the kernel's output scale.
        softmax_scale = (head_dim**-0.5) * float(k_scale)
        output_scale = float(v_scale)
        if k_cache.dtype == torch.uint8:
            # _maybe_dequant's byte decode yields x * 2^-8; fold 2^8 into the
            # scales so the decode needs no multiply.
            softmax_scale *= 256.0
            output_scale *= 256.0
        else:
            assert k_cache.dtype == torch.float8_e4m3fn
    else:
        softmax_scale = head_dim**-0.5
        output_scale = 1.0
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    assert q.device == k_cache.device == v_cache.device
    assert q.device == logical_indices.device == block_table.device
    assert q.device == token_to_req.device
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1

    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA sparse output must match its query")
    assert out.dtype == q.dtype and out.device == q.device
    assert out.stride(2) == 1
    assert output_gate.is_contiguous()
    output_gate_view = output_gate.view_as(q)
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    selection_width = logical_indices.shape[1] - 1  # trailing column is the count
    block_n, partial_warps, num_tiles, num_splits = _sail_config(
        q.shape[0], k_cache.shape[2], use_prefill_config, selection_width, is_fp8
    )

    # Split=1 writes output directly and compiles out all workspace accesses.
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        # FP32 partials preserve accuracy when merging independently normalized
        # splits.
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
    _fp8_qsa_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        softmax_scale,
        output_scale,
        output_gate_view,
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
        output_gate_view.stride(0),
        output_gate_view.stride(1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        TOPK=selection_width,
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=partial_warps,
        num_stages=2,
    )
    if num_splits == 1:
        return out

    _fp8_merge_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        output_gate_view,
        out.stride(0),
        out.stride(1),
        output_gate_view.stride(0),
        output_gate_view.stride(1),
        q.shape[0],
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=2,
        num_stages=1,
    )
    return out


def qsa_sparse_paged_attention(
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
):
    if k_cache.dtype == torch.bfloat16 and v_cache.dtype == torch.bfloat16:
        if (
            q.ndim != 3
            or q.shape[1:] != (6, 256)
            or k_cache.ndim != 4
            or k_cache.shape[1:] != (784, 1, 256)
            or v_cache.shape != k_cache.shape
            or logical_indices.shape != (q.shape[0], 2052)
            or block_table.ndim != 2
            or block_table.shape[1] != 88
        ):
            raise NotImplementedError(
                "BF16 QSA is limited to the validated shape family"
            )
        implementation = _bf16_small_attention if q.shape[0] <= 8 else _bf16_attention
    elif k_cache.dtype == v_cache.dtype and k_cache.dtype in (
        torch.uint8,
        torch.float8_e4m3fn,
    ):
        if k_cache.dtype == torch.float8_e4m3fn:
            # Same storage and shape; the PPU path decodes the bytes in Triton.
            k_cache = k_cache.view(torch.uint8)
            v_cache = v_cache.view(torch.uint8)
        implementation = _fp8_attention
    else:
        raise NotImplementedError("QSA supports BF16 or E4M3FN KV")
    return implementation(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        use_prefill_config,
        out=out,
        k_scale=k_scale,
        v_scale=v_scale,
        output_gate=output_gate,
    )
