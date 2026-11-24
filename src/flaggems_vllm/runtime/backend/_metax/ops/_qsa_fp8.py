# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Experimental three-pass sparse attention, independently tested on C550.

QK is computed once per selection tile. Global softmax writes low-precision
probabilities. PV consumes wider tiles without keeping Q/K and online-softmax
state live alongside its accumulator. Scratch allocation is part of every call.
This trial supports BF16 queries, one KV head, six query heads and D=256 only.
"""
# Source contract: MetaX-MACA/vLLM-metax, commit
# 5e1a0bd66493e3845f3c3fe354f87667b3fbfb8d:
# https://github.com/MetaX-MACA/vLLM-metax/blob/5e1a0bd/vllm_metax/models/qwen4_exp/metax/ops/qsa.py
# Changes: independent three-pass QK/softmax/PV rewrite, K64 dot chunks,
# optional split-PV and fused output gate. Preserved from measured experiment 164.

import torch
import triton
import triton.language as tl

RESOURCE_SINK = None


@triton.jit
def _resolve(
    indices,
    table,
    mapping,
    row,
    columns,
    stride_indices,
    stride_table,
    num_requests,
    num_blocks,
    TOPK: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
):
    count = tl.load(indices + row * stride_indices + TOPK)
    active = (columns < TOPK) & (columns < tl.maximum(count, 0))
    logical = tl.load(indices + row * stride_indices + columns, mask=active, other=-1)
    request = tl.load(mapping + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    token = tl.maximum(logical, 0)
    logical_page = token // PAGE
    page_offset = token % PAGE
    valid = active & (logical >= 0) & (logical_page < TABLE_WIDTH)
    valid &= (request >= 0) & (request < num_requests)
    physical = tl.load(
        table + safe_request * stride_table + tl.minimum(logical_page, TABLE_WIDTH - 1),
        mask=valid,
        other=-1,
    )
    valid &= (physical >= 0) & (physical < num_blocks)
    return tl.maximum(physical, 0).to(tl.int64), page_offset, valid


@triton.jit
def _qk(
    q,
    k,
    indices,
    table,
    mapping,
    scores,
    q_row,
    q_head,
    k_block,
    k_token,
    index_row,
    table_row,
    num_requests,
    num_blocks,
    score_scale,
    TOPK: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    N: tl.constexpr,
    K_DN: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * N + tl.arange(0, N)
    physical, offset, valid = _resolve(
        indices,
        table,
        mapping,
        row,
        columns,
        index_row,
        table_row,
        num_requests,
        num_blocks,
        TOPK,
        PAGE,
        TABLE_WIDTH,
    )
    heads = tl.arange(0, 16)
    dims = tl.arange(0, DIM)
    query = tl.load(
        q + row * q_row + heads[:, None] * q_head + dims[None, :],
        mask=heads[:, None] < HEADS,
        other=0.0,
    )
    if K_DN:
        keys = tl.load(
            k + physical[None, :] * k_block + offset[None, :] * k_token + dims[:, None],
            mask=valid[None, :],
            other=0.0,
        ).to(query.dtype)
    else:
        keys_nd = tl.load(
            k + physical[:, None] * k_block + offset[:, None] * k_token + dims[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(query.dtype)
        keys = tl.permute(keys_nd, (1, 0))
    logits = tl.dot(query, keys) * score_scale
    logits = tl.where(valid[None, :], logits, -float("inf"))
    address = (row.to(tl.int64) * HEADS + heads[:, None]) * TOPK + columns[None, :]
    tl.store(
        scores + address,
        logits,
        mask=(heads[:, None] < HEADS) & (columns[None, :] < TOPK),
    )


@triton.jit
def _qk_reduce(
    q,
    k,
    indices,
    table,
    mapping,
    scores,
    q_row,
    q_head,
    k_block,
    k_token,
    index_row,
    table_row,
    num_requests,
    num_blocks,
    score_scale,
    TOPK: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    N: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * N + tl.arange(0, N)
    physical, offset, valid = _resolve(
        indices,
        table,
        mapping,
        row,
        columns,
        index_row,
        table_row,
        num_requests,
        num_blocks,
        TOPK,
        PAGE,
        TABLE_WIDTH,
    )
    dims = tl.arange(0, DIM)
    # Contiguous channel loads; one key tile feeds all six real query heads.
    keys = tl.load(
        k + physical[:, None] * k_block + offset[:, None] * k_token + dims[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    for head in tl.static_range(HEADS):
        query = tl.load(q + row * q_row + head * q_head + dims).to(tl.float32)
        logits = tl.sum(keys * query[None, :], 1) * score_scale
        logits = tl.where(valid, logits, -float("inf"))
        address = (row.to(tl.int64) * HEADS + head) * TOPK + columns
        tl.store(scores + address, logits, mask=columns < TOPK)


@triton.jit
def _qk_chunked(
    q,
    k,
    indices,
    table,
    mapping,
    scores,
    q_row,
    q_head,
    k_block,
    k_token,
    index_row,
    table_row,
    num_requests,
    num_blocks,
    score_scale,
    TOPK: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    N: tl.constexpr,
    K_CHUNK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * N + tl.arange(0, N)
    physical, offset, valid = _resolve(
        indices,
        table,
        mapping,
        row,
        columns,
        index_row,
        table_row,
        num_requests,
        num_blocks,
        TOPK,
        PAGE,
        TABLE_WIDTH,
    )
    heads = tl.arange(0, 16)
    dims = tl.arange(0, K_CHUNK)
    accumulator = tl.zeros((16, N), tl.float32)
    # A smaller reduction tile reduces operand/shared-memory pressure, allowing
    # wider selection tiles without loading a full 256-channel operand at once.
    for part in range(DIM // K_CHUNK):
        channels = part * K_CHUNK + dims
        query = tl.load(
            q + row * q_row + heads[:, None] * q_head + channels[None, :],
            mask=heads[:, None] < HEADS,
            other=0.0,
        )
        keys_nd = tl.load(
            k
            + physical[:, None] * k_block
            + offset[:, None] * k_token
            + channels[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(query.dtype)
        accumulator = tl.dot(query, tl.permute(keys_nd, (1, 0)), acc=accumulator)
    logits = tl.where(valid[None, :], accumulator * score_scale, -float("inf"))
    address = (row.to(tl.int64) * HEADS + heads[:, None]) * TOPK + columns[None, :]
    tl.store(
        scores + address,
        logits,
        mask=(heads[:, None] < HEADS) & (columns[None, :] < TOPK),
    )


@triton.jit
def _softmax(scores, probabilities, TOPK: tl.constexpr, WIDTH: tl.constexpr):
    item = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, WIDTH)
    logits = tl.load(
        scores + item * TOPK + columns, mask=columns < TOPK, other=-float("inf")
    )
    maximum = tl.max(logits, 0)
    # An all-invalid selection must yield exact zeros rather than NaNs.
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp2(logits - safe_maximum)
    total = tl.sum(weights, 0)
    weights /= tl.maximum(total, 1.0e-20)
    tl.store(probabilities + item * TOPK + columns, weights, mask=columns < TOPK)


@triton.jit
def _pv(
    v,
    indices,
    table,
    mapping,
    probabilities,
    out,
    gate,
    v_block,
    v_token,
    index_row,
    table_row,
    out_row,
    out_head,
    num_requests,
    num_blocks,
    value_scale,
    TOPK: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    N: tl.constexpr,
    OUT_DIM: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    row = tl.program_id(0)
    dims = tl.program_id(1) * OUT_DIM + tl.arange(0, OUT_DIM)
    heads = tl.arange(0, 16)
    column_offsets = tl.arange(0, N)
    count = tl.load(indices + row * index_row + TOPK)
    tile_end = tl.cdiv(tl.minimum(tl.maximum(count, 0), TOPK), N)
    accumulator = tl.zeros((16, OUT_DIM), tl.float32)
    for tile in range(tile_end):
        columns = tile * N + column_offsets
        physical, offset, valid = _resolve(
            indices,
            table,
            mapping,
            row,
            columns,
            index_row,
            table_row,
            num_requests,
            num_blocks,
            TOPK,
            PAGE,
            TABLE_WIDTH,
        )
        p_address = (row.to(tl.int64) * HEADS + heads[:, None]) * TOPK + columns[
            None, :
        ]
        weights = tl.load(
            probabilities + p_address,
            mask=(heads[:, None] < HEADS) & (columns[None, :] < TOPK),
            other=0.0,
        )
        values = tl.load(
            v + physical[:, None] * v_block + offset[:, None] * v_token + dims[None, :],
            mask=valid[:, None] & (dims[None, :] < DIM),
            other=0.0,
        )
        if IS_FP8:
            values = values.to(tl.float16)
        # The FP8 route keeps both weights and values in FP16, as the prior
        # best does. BF16 caches keep BF16 PV operands.
        accumulator = tl.dot(weights.to(values.dtype), values, acc=accumulator)
    attention = (accumulator * value_scale).to(out.dtype.element_ty)
    mask = (heads[:, None] < HEADS) & (dims[None, :] < DIM)
    gates = tl.load(
        gate + (row * HEADS + heads[:, None]) * DIM + dims[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    result = attention.to(tl.float32) * tl.sigmoid(gates)
    tl.store(
        out + row * out_row + heads[:, None] * out_head + dims[None, :],
        result,
        mask=mask,
    )


@triton.jit
def _pv_split(
    v,
    indices,
    table,
    mapping,
    probabilities,
    partial,
    v_block,
    v_token,
    index_row,
    table_row,
    num_rows,
    num_requests,
    num_blocks,
    TOPK: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    N: tl.constexpr,
    OUT_DIM: tl.constexpr,
    IS_FP8: tl.constexpr,
    SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    dims = tl.program_id(1) * OUT_DIM + tl.arange(0, OUT_DIM)
    split = tl.program_id(2)
    heads = tl.arange(0, 16)
    column_offsets = tl.arange(0, N)
    count = tl.load(indices + row * index_row + TOPK)
    count_tiles = tl.cdiv(tl.minimum(tl.maximum(count, 0), TOPK), N)
    tiles_per_split: tl.constexpr = tl.cdiv(tl.cdiv(TOPK, N), SPLITS)
    first = split * tiles_per_split
    end = tl.minimum(count_tiles, first + tiles_per_split)
    accumulator = tl.zeros((16, OUT_DIM), tl.float32)
    for tile in range(first, end):
        columns = tile * N + column_offsets
        physical, offset, valid = _resolve(
            indices,
            table,
            mapping,
            row,
            columns,
            index_row,
            table_row,
            num_requests,
            num_blocks,
            TOPK,
            PAGE,
            TABLE_WIDTH,
        )
        p_address = (row.to(tl.int64) * HEADS + heads[:, None]) * TOPK + columns[
            None, :
        ]
        weights = tl.load(
            probabilities + p_address,
            mask=(heads[:, None] < HEADS) & (columns[None, :] < TOPK),
            other=0.0,
        )
        values = tl.load(
            v + physical[:, None] * v_block + offset[:, None] * v_token + dims[None, :],
            mask=valid[:, None] & (dims[None, :] < DIM),
            other=0.0,
        )
        if IS_FP8:
            values = values.to(tl.float16)
        accumulator = tl.dot(weights.to(values.dtype), values, acc=accumulator)
    address = (
        (split.to(tl.int64) * num_rows + row) * HEADS + heads[:, None]
    ) * DIM + dims[None, :]
    tl.store(
        partial + address,
        accumulator,
        mask=(heads[:, None] < HEADS) & (dims[None, :] < DIM),
    )


@triton.jit
def _pv_merge(
    partial,
    out,
    gate,
    num_rows,
    out_row,
    out_head,
    value_scale,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    splits = tl.arange(0, SPLITS)
    dims = tl.arange(0, DIM)
    address = (
        (splits[:, None].to(tl.int64) * num_rows + row) * HEADS + head
    ) * DIM + dims[None, :]
    partials = tl.load(partial + address)
    attention = (tl.sum(partials, 0) * value_scale).to(out.dtype.element_ty)
    gates = tl.load(gate + (row * HEADS + head) * DIM + dims).to(tl.float32)
    result = attention.to(tl.float32) * tl.sigmoid(gates)
    tl.store(out + row * out_row + head * out_head + dims, result)


@triton.jit
def _prepare_offsets(
    indices,
    table,
    mapping,
    precomputed,
    index_row,
    table_row,
    k_block,
    k_token,
    num_requests,
    num_blocks,
    TOPK: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    PRE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * PRE_BLOCK + tl.arange(0, PRE_BLOCK)
    physical, offset, valid = _resolve(
        indices,
        table,
        mapping,
        row,
        columns,
        index_row,
        table_row,
        num_requests,
        num_blocks,
        TOPK,
        PAGE,
        TABLE_WIDTH,
    )
    address = tl.where(
        valid, physical.to(tl.int32) * k_block + offset.to(tl.int32) * k_token, -1
    ).to(tl.int32)
    tl.store(
        precomputed + row.to(tl.int64) * TOPK + columns, address, mask=columns < TOPK
    )


@triton.jit
def _qk_offsets(
    q,
    k,
    indices,
    table,
    mapping,
    precomputed,
    scores,
    q_row,
    q_head,
    k_block,
    k_token,
    index_row,
    table_row,
    num_requests,
    num_blocks,
    score_scale,
    TOPK: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    N: tl.constexpr,
    K_CHUNK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * N + tl.arange(0, N)
    physical = tl.load(
        precomputed + row.to(tl.int64) * TOPK + columns, mask=columns < TOPK, other=-1
    ).to(tl.int32)
    valid = physical >= 0
    physical = tl.maximum(physical, 0)
    heads = tl.arange(0, 16)
    dims = tl.arange(0, K_CHUNK)
    accumulator = tl.zeros((16, N), tl.float32)
    # A smaller reduction tile reduces operand/shared-memory pressure, allowing
    # wider selection tiles without loading a full 256-channel operand at once.
    for part in range(DIM // K_CHUNK):
        channels = part * K_CHUNK + dims
        query = tl.load(
            q + row * q_row + heads[:, None] * q_head + channels[None, :],
            mask=heads[:, None] < HEADS,
            other=0.0,
        )
        keys_nd = tl.load(
            k + physical[:, None] + channels[None, :], mask=valid[:, None], other=0.0
        ).to(query.dtype)
        accumulator = tl.dot(query, tl.permute(keys_nd, (1, 0)), acc=accumulator)
    logits = tl.where(valid[None, :], accumulator * score_scale, -float("inf"))
    address = (row.to(tl.int64) * HEADS + heads[:, None]) * TOPK + columns[None, :]
    tl.store(
        scores + address,
        logits,
        mask=(heads[:, None] < HEADS) & (columns[None, :] < TOPK),
    )


@triton.jit
def _pv_offsets(
    v,
    indices,
    table,
    mapping,
    precomputed,
    probabilities,
    out,
    gate,
    v_block,
    v_token,
    index_row,
    table_row,
    out_row,
    out_head,
    num_requests,
    num_blocks,
    value_scale,
    TOPK: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    N: tl.constexpr,
    OUT_DIM: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    row = tl.program_id(0)
    dims = tl.program_id(1) * OUT_DIM + tl.arange(0, OUT_DIM)
    heads = tl.arange(0, 16)
    column_offsets = tl.arange(0, N)
    count = tl.load(indices + row * index_row + TOPK)
    tile_end = tl.cdiv(tl.minimum(tl.maximum(count, 0), TOPK), N)
    accumulator = tl.zeros((16, OUT_DIM), tl.float32)
    for tile in range(tile_end):
        columns = tile * N + column_offsets
        physical = tl.load(
            precomputed + row.to(tl.int64) * TOPK + columns,
            mask=columns < TOPK,
            other=-1,
        ).to(tl.int32)
        valid = physical >= 0
        physical = tl.maximum(physical, 0)
        p_address = (row.to(tl.int64) * HEADS + heads[:, None]) * TOPK + columns[
            None, :
        ]
        weights = tl.load(
            probabilities + p_address,
            mask=(heads[:, None] < HEADS) & (columns[None, :] < TOPK),
            other=0.0,
        )
        values = tl.load(
            v + physical[:, None] + dims[None, :],
            mask=valid[:, None] & (dims[None, :] < DIM),
            other=0.0,
        )
        if IS_FP8:
            values = values.to(tl.float16)
        # The FP8 route keeps both weights and values in FP16, as the prior
        # best does. BF16 caches keep BF16 PV operands.
        accumulator = tl.dot(weights.to(values.dtype), values, acc=accumulator)
    attention = (accumulator * value_scale).to(out.dtype.element_ty)
    mask = (heads[:, None] < HEADS) & (dims[None, :] < DIM)
    gates = tl.load(
        gate + (row * HEADS + heads[:, None]) * DIM + dims[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    result = attention.to(tl.float32) * tl.sigmoid(gates)
    tl.store(
        out + row * out_row + heads[:, None] * out_head + dims[None, :],
        result,
        mask=mask,
    )


@triton.jit
def _pv_split_offsets(
    v,
    indices,
    table,
    mapping,
    precomputed,
    probabilities,
    partial,
    v_block,
    v_token,
    index_row,
    table_row,
    num_rows,
    num_requests,
    num_blocks,
    TOPK: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    N: tl.constexpr,
    OUT_DIM: tl.constexpr,
    IS_FP8: tl.constexpr,
    SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    dims = tl.program_id(1) * OUT_DIM + tl.arange(0, OUT_DIM)
    split = tl.program_id(2)
    heads = tl.arange(0, 16)
    column_offsets = tl.arange(0, N)
    count = tl.load(indices + row * index_row + TOPK)
    count_tiles = tl.cdiv(tl.minimum(tl.maximum(count, 0), TOPK), N)
    tiles_per_split: tl.constexpr = tl.cdiv(tl.cdiv(TOPK, N), SPLITS)
    first = split * tiles_per_split
    end = tl.minimum(count_tiles, first + tiles_per_split)
    accumulator = tl.zeros((16, OUT_DIM), tl.float32)
    for tile in range(first, end):
        columns = tile * N + column_offsets
        physical = tl.load(
            precomputed + row.to(tl.int64) * TOPK + columns,
            mask=columns < TOPK,
            other=-1,
        ).to(tl.int32)
        valid = physical >= 0
        physical = tl.maximum(physical, 0)
        p_address = (row.to(tl.int64) * HEADS + heads[:, None]) * TOPK + columns[
            None, :
        ]
        weights = tl.load(
            probabilities + p_address,
            mask=(heads[:, None] < HEADS) & (columns[None, :] < TOPK),
            other=0.0,
        )
        values = tl.load(
            v + physical[:, None] + dims[None, :],
            mask=valid[:, None] & (dims[None, :] < DIM),
            other=0.0,
        )
        if IS_FP8:
            values = values.to(tl.float16)
        accumulator = tl.dot(weights.to(values.dtype), values, acc=accumulator)
    address = (
        (split.to(tl.int64) * num_rows + row) * HEADS + heads[:, None]
    ) * DIM + dims[None, :]
    tl.store(
        partial + address,
        accumulator,
        mask=(heads[:, None] < HEADS) & (dims[None, :] < DIM),
    )


def _resource(name, compiled, grid, kwargs):
    if RESOURCE_SINK is None:
        return
    metadata = getattr(compiled, "metadata", None)
    RESOURCE_SINK.append(
        {
            "kernel": name,
            "grid": list(grid),
            "launch": kwargs,
            "shared_bytes": getattr(metadata, "shared", None),
            "n_regs_reported": getattr(compiled, "n_regs", None),
            "n_spills_reported": getattr(compiled, "n_spills", None),
            "compiled_hash": str(getattr(compiled, "hash", "")),
        }
    )


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
    qk_n=64,
    pv_n=64,
    qk_warps=4,
    pv_warps=4,
    out_dim=128,
    k_dn=True,
    qk_mode="dot",
    qk_chunk=64,
    pv_splits=1,
    use_offsets=False,
    pre_block=256,
    offset_mode="both",
):
    """Trial contract; all scratch, launches, scaling and gate are timed."""
    if q.ndim != 3 or q.shape[1:] != (6, 256) or q.dtype != torch.bfloat16:
        raise ValueError("Trial requires BF16 Q[T,6,256]")
    if (
        k_cache.ndim != 4
        or k_cache.shape[2:] != (1, 256)
        or v_cache.shape != k_cache.shape
    ):
        raise ValueError("Trial requires K/V[B,page,1,256]")
    if k_cache.dtype != v_cache.dtype or k_cache.dtype not in (
        torch.bfloat16,
        torch.float8_e4m3fn,
    ):
        raise ValueError("Trial requires BF16 or FP8 e4m3fn KV")
    if (
        qk_n not in (16, 32, 64, 128, 256, 512)
        or pv_n not in (32, 64, 128)
        or out_dim not in (64, 128, 256)
    ):
        raise ValueError("Unexpected experimental tile sizes")
    if qk_mode not in ("dot", "reduce", "chunked") or qk_chunk not in (32, 64, 128):
        raise ValueError("Unexpected QK algorithm")
    if pv_splits not in (1, 2, 4, 8):
        raise ValueError("Unexpected PV split count")
    if use_offsets and (
        qk_mode != "chunked"
        or k_cache.stride(0) != v_cache.stride(0)
        or k_cache.stride(1) != v_cache.stride(1)
    ):
        raise ValueError(
            "Offsets trial requires chunked QK and equal K/V block/token strides"
        )
    if pre_block not in (256, 1024):
        raise ValueError("Unexpected offset preprocessing block")
    if offset_mode not in ("both", "qk", "pv"):
        raise ValueError("Unexpected offset mode")
    rows, heads, dim = q.shape
    if (
        logical_indices.ndim != 2
        or logical_indices.shape[0] != rows
        or logical_indices.shape[1] < 2
    ):
        raise ValueError("Invalid packed selections")
    if (
        block_table.ndim != 2
        or not all(block_table.shape)
        or not all(k_cache.shape[:2])
    ):
        raise ValueError("Nonempty cache and table required")
    if token_to_req.shape != (rows,):
        raise ValueError("Invalid token mapping")
    if any(
        x.dtype != torch.int32 for x in (logical_indices, block_table, token_to_req)
    ):
        raise ValueError("Metadata must be int32")
    if any(
        x.device != q.device
        for x in (
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            output_gate,
        )
    ):
        raise ValueError("All tensors must share a device")
    if any(
        x.stride(-1) != 1
        for x in (q, k_cache, v_cache, logical_indices, block_table, token_to_req)
    ):
        raise ValueError("Unsupported inner stride")
    if (
        output_gate.dtype not in (torch.bfloat16, torch.float32)
        or output_gate.numel() != q.numel()
        or not output_gate.is_contiguous()
    ):
        raise ValueError("Invalid gate")
    fp8 = k_cache.dtype == torch.float8_e4m3fn
    if fp8 and (k_scale is None or v_scale is None):
        raise ValueError("FP8 requires K and V scales")
    score_scale = (dim**-0.5) * (float(k_scale) if fp8 else 1.0) * 1.4426950408889634
    value_scale = float(v_scale) if fp8 else 1.0
    if out is None:
        out = torch.empty_like(q)
    elif (
        out.shape != q.shape
        or out.dtype != q.dtype
        or out.device != q.device
        or out.stride(-1) != 1
    ):
        raise ValueError("Invalid output")
    else:
        for tensor in (
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            output_gate,
        ):
            if tensor.untyped_storage().data_ptr() == out.untyped_storage().data_ptr():
                raise ValueError("Output may not share storage with inputs")
        # Trial accepts strided rows/heads when their address ranges do not overlap.
        if out.stride(1) < dim or out.stride(0) < (heads - 1) * out.stride(1) + dim:
            raise ValueError("Output rows/heads overlap")
    if rows == 0:
        return out
    topk = logical_indices.shape[1] - 1
    precomputed = None
    if use_offsets:
        block_stride, token_stride = k_cache.stride(0), k_cache.stride(1)
        max_offset = (
            (k_cache.shape[0] - 1) * block_stride
            + (k_cache.shape[1] - 1) * token_stride
            + dim
            - 1
        )
        if (
            min(block_stride, token_stride) < 0
            or max(block_stride, token_stride, max_offset) > 2147483647
        ):
            raise ValueError("K/V element offsets do not fit in signed int32")
        precomputed = torch.empty((rows, topk), dtype=torch.int32, device=q.device)
        grid = (rows, triton.cdiv(topk, pre_block))
        launch = dict(
            TOPK=topk,
            PAGE=k_cache.shape[1],
            TABLE_WIDTH=block_table.shape[1],
            PRE_BLOCK=pre_block,
            num_warps=4,
            num_stages=1,
        )
        compiled = _prepare_offsets[grid](
            logical_indices,
            block_table,
            token_to_req,
            precomputed,
            logical_indices.stride(0),
            block_table.stride(0),
            k_cache.stride(0),
            k_cache.stride(1),
            block_table.shape[0],
            k_cache.shape[0],
            **launch,
        )
        if RESOURCE_SINK is not None:
            _resource("prepare_offsets", compiled, grid, launch)
    scores = torch.empty((rows, heads, topk), dtype=torch.float32, device=q.device)
    probabilities = torch.empty(
        (rows, heads, topk),
        dtype=torch.float16 if fp8 else torch.bfloat16,
        device=q.device,
    )
    meta = dict(
        TOPK=topk,
        PAGE=k_cache.shape[1],
        TABLE_WIDTH=block_table.shape[1],
        HEADS=heads,
        DIM=dim,
    )
    grid = (rows, triton.cdiv(topk, qk_n))
    if qk_mode == "dot":
        qk_kernel = _qk
        launch = dict(N=qk_n, K_DN=k_dn, num_warps=qk_warps, num_stages=1)
    elif qk_mode == "reduce":
        qk_kernel = _qk_reduce
        launch = dict(N=qk_n, num_warps=qk_warps, num_stages=1)
    else:
        qk_kernel = _qk_chunked
        launch = dict(N=qk_n, K_CHUNK=qk_chunk, num_warps=qk_warps, num_stages=1)
    if use_offsets and offset_mode != "pv":
        qk_kernel = _qk_offsets
    qk_args = (q, k_cache, logical_indices, block_table, token_to_req)
    if use_offsets and offset_mode != "pv":
        qk_args += (precomputed,)
    compiled = qk_kernel[grid](
        *qk_args,
        scores,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        logical_indices.stride(0),
        block_table.stride(0),
        block_table.shape[0],
        k_cache.shape[0],
        score_scale,
        **meta,
        **launch,
    )
    if RESOURCE_SINK is not None:
        _resource("qk_" + qk_mode, compiled, grid, launch)
    grid = (rows * heads,)
    launch = dict(WIDTH=triton.next_power_of_2(topk), num_warps=4, num_stages=1)
    compiled = _softmax[grid](scores, probabilities, TOPK=topk, **launch)
    if RESOURCE_SINK is not None:
        _resource("softmax", compiled, grid, launch)
    if use_offsets and offset_mode != "qk" and pv_splits == 1:
        grid = (rows, triton.cdiv(dim, out_dim))
        launch = dict(
            N=pv_n, OUT_DIM=out_dim, IS_FP8=fp8, num_warps=pv_warps, num_stages=1
        )
        compiled = _pv_offsets[grid](
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            precomputed,
            probabilities,
            out,
            output_gate,
            v_cache.stride(0),
            v_cache.stride(1),
            logical_indices.stride(0),
            block_table.stride(0),
            out.stride(0),
            out.stride(1),
            block_table.shape[0],
            k_cache.shape[0],
            value_scale,
            **meta,
            **launch,
        )
        if RESOURCE_SINK is not None:
            _resource("pv_offsets", compiled, grid, launch)
    elif use_offsets and offset_mode != "qk":
        partial = torch.empty(
            (pv_splits, rows, heads, dim), dtype=torch.float32, device=q.device
        )
        grid = (rows, triton.cdiv(dim, out_dim), pv_splits)
        launch = dict(
            N=pv_n,
            OUT_DIM=out_dim,
            IS_FP8=fp8,
            SPLITS=pv_splits,
            num_warps=pv_warps,
            num_stages=1,
        )
        compiled = _pv_split_offsets[grid](
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            precomputed,
            probabilities,
            partial,
            v_cache.stride(0),
            v_cache.stride(1),
            logical_indices.stride(0),
            block_table.stride(0),
            rows,
            block_table.shape[0],
            k_cache.shape[0],
            **meta,
            **launch,
        )
        if RESOURCE_SINK is not None:
            _resource("pv_split_offsets", compiled, grid, launch)
        grid = (rows, heads)
        launch = dict(HEADS=heads, DIM=dim, SPLITS=pv_splits, num_warps=4, num_stages=1)
        compiled = _pv_merge[grid](
            partial,
            out,
            output_gate,
            rows,
            out.stride(0),
            out.stride(1),
            value_scale,
            **launch,
        )
        if RESOURCE_SINK is not None:
            _resource("pv_merge", compiled, grid, launch)
    elif pv_splits == 1:
        grid = (rows, triton.cdiv(dim, out_dim))
        launch = dict(
            N=pv_n, OUT_DIM=out_dim, IS_FP8=fp8, num_warps=pv_warps, num_stages=1
        )
        compiled = _pv[grid](
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            probabilities,
            out,
            output_gate,
            v_cache.stride(0),
            v_cache.stride(1),
            logical_indices.stride(0),
            block_table.stride(0),
            out.stride(0),
            out.stride(1),
            block_table.shape[0],
            k_cache.shape[0],
            value_scale,
            **meta,
            **launch,
        )
        if RESOURCE_SINK is not None:
            _resource("pv", compiled, grid, launch)
    else:
        partial = torch.empty(
            (pv_splits, rows, heads, dim), dtype=torch.float32, device=q.device
        )
        grid = (rows, triton.cdiv(dim, out_dim), pv_splits)
        launch = dict(
            N=pv_n,
            OUT_DIM=out_dim,
            IS_FP8=fp8,
            SPLITS=pv_splits,
            num_warps=pv_warps,
            num_stages=1,
        )
        compiled = _pv_split[grid](
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            probabilities,
            partial,
            v_cache.stride(0),
            v_cache.stride(1),
            logical_indices.stride(0),
            block_table.stride(0),
            rows,
            block_table.shape[0],
            k_cache.shape[0],
            **meta,
            **launch,
        )
        if RESOURCE_SINK is not None:
            _resource("pv_split", compiled, grid, launch)
        grid = (rows, heads)
        launch = dict(HEADS=heads, DIM=dim, SPLITS=pv_splits, num_warps=4, num_stages=1)
        compiled = _pv_merge[grid](
            partial,
            out,
            output_gate,
            rows,
            out.stride(0),
            out.stride(1),
            value_scale,
            **launch,
        )
        if RESOURCE_SINK is not None:
            _resource("pv_merge", compiled, grid, launch)
    return out
