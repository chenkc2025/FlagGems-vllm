# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA collection interface and shared synthetic QSA input construction.

The three (tokens, cache pages, requests) bindings are recorded dimensions;
tensor values, cache layout, sparse indices and request metadata are synthetic.
DTYPES describes KV storage: Q/output are always BF16. E5M2 is intentionally
absent from NVIDIA collection because it is a Hygon-specific extension.
No torch attention reference is substituted when the native op is unavailable.
"""

import importlib
import inspect

import torch

OP_NAME = "qsa_sparse_paged_attention"
DTYPES = [torch.bfloat16, torch.float8_e4m3fn]
IS_INPLACE = False


def native():
    """Return the complete NVIDIA API, or None for an incompatible installation.

    Reference contract: vLLM v0.32.0, commit
    5df63a7203cdb9328d824a27b682e77c7b9928be,
    vllm/models/qwen4_exp/nvidia/ops/qsa.py.
    """
    if torch.version.hip is not None:
        return None
    try:
        module = importlib.import_module("vllm.models.qwen4_exp.nvidia.ops.qsa")
        op = module.qsa_sparse_paged_attention
        params = inspect.signature(op).parameters
    except (ImportError, AttributeError, TypeError, ValueError):
        return None
    expected = (
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
    if tuple(params) != expected:
        return None
    if params["output_gate"].kind != inspect.Parameter.KEYWORD_ONLY:
        return None
    return op


def grid():
    """Use exactly these bindings in NVIDIA collection and Hygon benchmark."""
    return [
        dict(
            tokens=t,
            pages=p,
            requests=r,
            heads=6,
            head_dim=256,
            topk=2051,
            page_size=784,
            table_width=88,
            use_prefill_config=False,
        )
        for t, p, r in ((8, 2712, 8), (902, 2712, 451), (16384, 2712, 54))
    ]


def build_inputs(binding, dtype, device):
    """Return (args, kwargs); allocation/quantization stays outside timing."""
    t, p, r = (binding[k] for k in ("tokens", "pages", "requests"))
    h, d, width, page, table_width = (
        binding[k] for k in ("heads", "head_dim", "topk", "page_size", "table_width")
    )
    if dtype not in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("Unsupported QSA KV storage dtype")
    gen = torch.Generator(device=device).manual_seed(856 + t + p)
    q = torch.randn((t, h, d), device=device, dtype=torch.bfloat16, generator=gen)
    cache = torch.randn(
        (p, page, 1, d * 2), device=device, dtype=torch.bfloat16, generator=gen
    )
    k, v = cache.split(d, -1)
    ks, vs = (1.0, 1.0) if dtype == torch.bfloat16 else (0.5, 2.0)
    if dtype != torch.bfloat16:
        k, v = (k.float() / ks).to(dtype), (v.float() / vs).to(dtype)
    permutation = torch.randperm(p, device=device, generator=gen)
    positions = (
        torch.arange(r, device=device)[:, None] * table_width
        + torch.arange(table_width, device=device)[None, :]
    ) % p
    table = permutation[positions].to(torch.int32)
    rows = torch.arange(t, device=device, dtype=torch.int32)
    req = (rows * r // t).to(torch.int32)
    count = min(width - 3, page * table_width)
    chosen = torch.randperm(page * table_width, device=device, generator=gen)[:count]
    packed = torch.full((t, width + 1), -1, device=device, dtype=torch.int32)
    packed[:, :count] = ((chosen[None, :] + rows[:, None]) % (page * table_width)).to(
        torch.int32
    )
    packed[:, -1] = count
    gate = torch.randn(q.shape, device=device, dtype=q.dtype, generator=gen)
    args = (q, k, v, packed, table, req)
    kwargs = dict(
        use_prefill_config=binding["use_prefill_config"],
        out=torch.empty_like(q),
        k_scale=ks,
        v_scale=vs,
        output_gate=gate,
    )
    return args, kwargs


def key_shape(binding):
    return tuple(
        binding[k] for k in ("tokens", "pages", "requests", "heads", "head_dim", "topk")
    )


def config(binding, dtype):
    return dict(
        binding,
        kv_dtype=str(dtype),
        q_dtype="torch.bfloat16",
        input_origin="recorded dimensions; all data and metadata synthetic",
        scope="complete: packed count conversion + attention + sigmoid gate",
        timing="non-Graph; input generation/quantization excluded; preallocated out",
    )
