# PPU development QSA candidate: vendor-independent, adapted from 300.py and 303.py.
# BF16 packed-index count and BF16-rounded output gate are included.
from vllm.triton_utils import tl, triton


@triton.jit
def split_kernel(
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
def merge_kernel(
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


@triton.jit
def direct_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    output_ptr,
    gate_ptr,
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
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    K_TRANS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
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

    for start in tl.range(0, valid_count, BLOCK_N):
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
        if K_TRANS:
            keys = tl.load(
                k_cache_ptr
                + safe_page[:, None] * stride_k_block
                + page_offset[:, None] * stride_k_token
                + kv_head * stride_k_head
                + dim_offsets[None, :] * stride_k_dim,
                mask=(dim_offsets[None, :] < HEAD_DIM) & valid[:, None],
                other=0.0,
            )
            keys = tl.trans(keys)
        else:
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

    output = tl.where(
        normalizer[:, None] > 0,
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    gate_value = tl.load(
        gate_ptr
        + (row * (NUM_KV_HEADS * GROUP_SIZE) + first_head + head_offsets[:, None])
        * HEAD_DIM
        + dim_offsets[None, :],
        mask=(head_offsets[:, None] < GROUP_SIZE) & (dim_offsets[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    output = output.to(output_ptr.dtype.element_ty).to(tl.float32) * tl.sigmoid(
        gate_value
    )
    tl.store(
        output_ptr
        + row * stride_output_row
        + (first_head + head_offsets[:, None]) * stride_output_head
        + dim_offsets[None, :] * stride_output_dim,
        output,
        mask=(row < num_rows)
        & (head_offsets[:, None] < GROUP_SIZE)
        & (dim_offsets[None, :] < HEAD_DIM),
    )
