"""Frozen BF16 development dispatch for the 164 recorded PPU QSA shapes.

Dependencies: 312.py (short queries), 336.py (count-correct larger queries).
Both are pinned below. This module does not change installed operator sources.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(filename, digest):
    path = HERE / filename
    actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual != digest:
        raise RuntimeError(f"{filename} SHA mismatch")
    name = "_qsa342_" + filename[:3]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_small = _load(
    "312.py", "7eba530eb2cbddf96eec9ae36ba06639cd1e8c7fcba557beb9116c9b99551ae2"
)
_large = _load(
    "336.py", "f047046828ba0cdf78fc7d2e0a1f6c6c0861143ef1c74e87a447d90b2d0a9e22"
)


def _select_config(
    num_rows, num_kv_heads, use_prefill_config, num_columns, is_fp8=False
):
    if is_fp8:
        raise NotImplementedError(
            "This frozen candidate has only been developed for BF16 KV"
        )
    splits = 8 if num_rows <= 128 else (4 if num_rows <= 512 else 1)
    tiles = (num_columns + 31) // 32
    return 32, (4 if num_rows <= 512 else 2), tiles, min(splits, tiles)


_large._QSA_CONTIGUOUS_K = True
_large._QSA_STAGES = 2
_large._select_config = _select_config


def describe(rows):
    if rows <= 8:
        return {"path": "312", "split": 8, "bn": 32, "warps": 4, "stages": 2}
    return {
        "path": "336",
        "split": 8 if rows <= 128 else (4 if rows <= 512 else 1),
        "bn": 32,
        "warps": 4 if rows <= 512 else 2,
        "stages": 2,
    }


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
    if (
        q.ndim != 3
        or q.shape[1:] != (6, 256)
        or k_cache.ndim != 4
        or k_cache.shape[1:] != (784, 1, 256)
        or v_cache.shape != k_cache.shape
        or logical_indices.shape != (q.shape[0], 2052)
        or block_table.ndim != 2
        or block_table.shape[1] != 88
        or token_to_req.shape != (q.shape[0],)
    ):
        raise ValueError(
            "342.py is restricted to the recorded Hq6/Hkv1/D256/page784/selection2051 family"
        )
    if (
        q.dtype != torch.bfloat16
        or k_cache.dtype != q.dtype
        or v_cache.dtype != q.dtype
    ):
        raise NotImplementedError("342.py currently requires BF16 Q/K/V")
    if q.shape[0] > 8:
        return _large.qsa_sparse_paged_attention(
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
    _small.split_kernel[(t, 1, split)](
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
    _small.merge_kernel[(t, 6)](
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
