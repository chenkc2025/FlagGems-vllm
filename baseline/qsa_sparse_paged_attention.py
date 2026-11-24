# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Native NVIDIA capture and shared QSA benchmark inputs.

The recorded dimensions are paired with synthetic tensors and metadata.
DTYPES specifies KV storage; query and output use BF16.
"""

import importlib
import inspect
import logging

import torch

OP_NAME = "qsa_sparse_paged_attention"
DTYPES = [torch.bfloat16, torch.float8_e4m3fn]
IS_INPLACE = False
RECORDS = (
    (1, 512, 1, 96),
    (1, 2712, 1, 240),
    (2, 512, 2, 96),
    (2, 2712, 2, 240),
    (4, 512, 4, 96),
    (4, 2712, 4, 192),
    (8, 512, 8, 96),
    (8, 2712, 8, 192),
    (16, 512, 16, 96),
    (16, 2712, 16, 192),
    (24, 512, 24, 96),
    (24, 2712, 24, 192),
    (32, 512, 32, 96),
    (32, 2712, 32, 192),
    (40, 512, 40, 96),
    (40, 2712, 40, 192),
    (48, 512, 48, 96),
    (48, 2712, 48, 192),
    (56, 512, 56, 96),
    (56, 2712, 56, 192),
    (64, 512, 64, 96),
    (64, 2712, 64, 192),
    (72, 512, 72, 96),
    (72, 2712, 72, 192),
    (80, 512, 80, 96),
    (80, 2712, 80, 192),
    (88, 512, 88, 96),
    (88, 2712, 88, 192),
    (96, 512, 96, 96),
    (96, 2712, 96, 192),
    (104, 512, 104, 96),
    (104, 2712, 104, 192),
    (112, 512, 112, 96),
    (112, 2712, 112, 192),
    (120, 512, 120, 96),
    (120, 2712, 120, 192),
    (128, 512, 128, 96),
    (128, 2712, 128, 192),
    (136, 512, 136, 96),
    (136, 2712, 136, 192),
    (144, 512, 144, 96),
    (144, 2712, 144, 192),
    (152, 512, 152, 96),
    (152, 2712, 152, 192),
    (160, 512, 160, 96),
    (160, 2712, 160, 192),
    (168, 512, 168, 96),
    (168, 2712, 168, 192),
    (176, 512, 176, 96),
    (176, 2712, 176, 192),
    (184, 512, 184, 96),
    (184, 2712, 184, 192),
    (192, 512, 192, 96),
    (192, 2712, 192, 192),
    (200, 512, 200, 96),
    (200, 2712, 200, 192),
    (208, 512, 208, 96),
    (208, 2712, 208, 192),
    (216, 512, 216, 96),
    (216, 2712, 216, 192),
    (224, 512, 224, 96),
    (224, 2712, 224, 192),
    (232, 512, 232, 96),
    (232, 2712, 232, 192),
    (240, 512, 240, 96),
    (240, 2712, 240, 192),
    (248, 512, 248, 96),
    (248, 2712, 248, 192),
    (256, 512, 256, 96),
    (256, 2712, 256, 192),
    (272, 512, 272, 96),
    (272, 2712, 272, 192),
    (288, 512, 288, 96),
    (288, 2712, 288, 192),
    (304, 512, 304, 96),
    (304, 2712, 304, 192),
    (320, 512, 320, 96),
    (320, 2712, 320, 192),
    (336, 512, 336, 96),
    (336, 2712, 336, 192),
    (352, 512, 352, 96),
    (352, 2712, 352, 192),
    (368, 512, 368, 96),
    (368, 2712, 368, 192),
    (384, 512, 384, 96),
    (384, 2712, 384, 192),
    (400, 512, 400, 96),
    (400, 2712, 400, 192),
    (416, 512, 416, 96),
    (416, 2712, 416, 192),
    (432, 512, 432, 96),
    (432, 2712, 432, 192),
    (448, 512, 448, 96),
    (448, 2712, 448, 192),
    (451, 2712, 451, 48),
    (464, 512, 464, 96),
    (464, 2712, 464, 192),
    (480, 512, 480, 96),
    (480, 2712, 480, 192),
    (496, 512, 496, 192),
    (496, 2712, 496, 192),
    (512, 512, 512, 192),
    (512, 2712, 512, 192),
    (902, 2712, 451, 48),
    (2369, 2712, 54, 48),
    (6177, 2712, 57, 48),
    (16384, 2712, 1, 96),
    (16384, 2712, 2, 96),
    (16384, 2712, 3, 96),
    (16384, 2712, 4, 96),
    (16384, 2712, 5, 96),
    (16384, 2712, 6, 96),
    (16384, 2712, 7, 96),
    (16384, 2712, 8, 96),
    (16384, 2712, 9, 96),
    (16384, 2712, 10, 96),
    (16384, 2712, 11, 96),
    (16384, 2712, 12, 96),
    (16384, 2712, 13, 96),
    (16384, 2712, 14, 96),
    (16384, 2712, 15, 96),
    (16384, 2712, 16, 96),
    (16384, 2712, 17, 96),
    (16384, 2712, 18, 96),
    (16384, 2712, 19, 96),
    (16384, 2712, 20, 96),
    (16384, 2712, 21, 96),
    (16384, 2712, 22, 96),
    (16384, 2712, 23, 96),
    (16384, 2712, 24, 96),
    (16384, 2712, 25, 96),
    (16384, 2712, 26, 96),
    (16384, 2712, 27, 96),
    (16384, 2712, 28, 96),
    (16384, 2712, 29, 96),
    (16384, 2712, 30, 96),
    (16384, 2712, 31, 96),
    (16384, 2712, 32, 96),
    (16384, 2712, 33, 96),
    (16384, 2712, 34, 96),
    (16384, 2712, 35, 96),
    (16384, 2712, 36, 96),
    (16384, 2712, 37, 96),
    (16384, 2712, 38, 96),
    (16384, 2712, 39, 96),
    (16384, 2712, 40, 96),
    (16384, 2712, 41, 96),
    (16384, 2712, 42, 96),
    (16384, 2712, 43, 96),
    (16384, 2712, 44, 96),
    (16384, 2712, 45, 96),
    (16384, 2712, 46, 96),
    (16384, 2712, 47, 96),
    (16384, 2712, 48, 96),
    (16384, 2712, 49, 96),
    (16384, 2712, 50, 96),
    (16384, 2712, 51, 96),
    (16384, 2712, 52, 96),
    (16384, 2712, 53, 96),
    (16384, 2712, 54, 528),
    (16384, 2712, 55, 528),
    (16384, 2712, 56, 96),
    (16384, 2712, 57, 96),
    (16384, 2712, 2048, 48),
)
PARAMETERS = (
    "q",
    "k_cache",
    "v_cache",
    "logical_indices",
    "block_table",
    "token_to_req",
    "use_prefill_config",
    "out",
    "k_scale",
    "v_scale",
    "output_gate",
)


def resolve(qualified_name):
    try:
        op = importlib.import_module(qualified_name).qsa_sparse_paged_attention
        params = inspect.signature(op).parameters
    except (ImportError, AttributeError, TypeError, ValueError) as err:
        logging.getLogger(__name__).warning(
            "QSA unavailable in %s: %s", qualified_name, err
        )
        return None
    if (
        tuple(params) == PARAMETERS
        and params["output_gate"].kind == inspect.Parameter.KEYWORD_ONLY
    ):
        return op
    logging.getLogger(__name__).warning(
        "Unexpected QSA interface in %s", qualified_name
    )
    return None


def native():
    return resolve("vllm.models.qwen4_exp.nvidia.ops.qsa")


def grid():
    return [
        dict(
            id=shape_id,
            tokens=t,
            pages=p,
            requests=r,
            calls=calls,
            use_prefill_config=prefill,
        )
        for prefill in (True, False)
        for shape_id, (t, p, r, calls) in enumerate(RECORDS, 1)
    ]


def make_inputs(
    record,
    device,
    *,
    heads=6,
    dim=256,
    page=784,
    table_width=88,
    packed_width=2052,
    valid_count=2048,
):
    """Reuse the handoff's seed, page permutation, request mapping and KV layout."""
    tokens, blocks, requests, _ = record
    assert tokens >= requests > 0 and blocks >= table_width
    assert 1 <= valid_count < packed_width and valid_count <= page * table_width
    torch.manual_seed(2)
    q = torch.randn((tokens, heads, dim), dtype=torch.bfloat16, device=device)
    gate = torch.randn_like(q)
    kv = torch.randn((blocks, page, 1, 2 * dim), dtype=q.dtype, device=device)
    k, v = kv.split(dim, dim=-1)
    rows = torch.arange(tokens, device=device)
    permutation = torch.randperm(blocks, device=device)
    positions = (
        torch.arange(requests, device=device)[:, None] * table_width
        + torch.arange(table_width, device=device)[None, :]
    ) % blocks
    table = permutation[positions].to(torch.int32)
    token_to_req = (rows * requests // tokens).to(torch.int32)
    capacity = page * table_width
    selected = torch.randperm(capacity, device=device)[:valid_count]
    indices = torch.full((tokens, packed_width), -1, dtype=torch.int32, device=device)
    indices[:, :valid_count] = (selected[None, :] + rows[:, None]) % capacity
    indices[:, -1] = valid_count
    return (q, k, v, indices, table, token_to_req), {"output_gate": gate}


def build_inputs(binding, dtype, device):
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported KV dtype: {dtype}")
    record = (
        binding["tokens"],
        binding["pages"],
        binding["requests"],
        binding["calls"],
    )
    args, kwargs = make_inputs(record, device)
    if dtype == torch.float8_e4m3fn:
        q, k, v, indices, table, mapping = args
        k.div_(0.5)
        v.div_(2.0)
        fused = k.as_strided((*k.shape[:-1], 2 * k.shape[-1]), k.stride()).to(dtype)
        k, v = fused.split(q.shape[-1], dim=-1)
        args = (q, k, v, indices, table, mapping)
        kwargs.update(k_scale=0.5, v_scale=2.0)
    else:
        kwargs.update(k_scale=1.0, v_scale=1.0)
    kwargs["use_prefill_config"] = binding["use_prefill_config"]
    return args, kwargs


def key_shape(binding):
    return (binding["tokens"], 6, 256)


def config(binding, dtype):
    tokens, pages, requests = (binding[key] for key in ("tokens", "pages", "requests"))
    return dict(
        binding,
        q_shape=key_shape(binding),
        k_cache_shape=(pages, 784, 1, 256),
        v_cache_shape=(pages, 784, 1, 256),
        logical_indices_shape=(tokens, 2052),
        block_table_shape=(requests, 88),
        token_to_req_shape=(tokens,),
        output_gate_shape=key_shape(binding),
        q_dtype=str(torch.bfloat16),
        kv_dtype=str(dtype),
        k_scale=1.0 if dtype == torch.bfloat16 else 0.5,
        v_scale=1.0 if dtype == torch.bfloat16 else 2.0,
        out=None,
    )
