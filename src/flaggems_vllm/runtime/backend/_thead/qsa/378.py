"""Experimental BF16 QSA dispatch for the 164 recorded PPU shape family.

342.py remains the development reference. This wrapper only enables the
independently tested BN16/W2 path for selected recorded shapes and the captured
strides. Every other input is delegated to 342.py.
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
    name = "_qsa378_" + filename[:3]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_reference = _load(
    "342.py", "4059cf1a12dd328d4d7bc55423f4e04bef8a0e780154bbcd2e0f4503460f691e"
)
_n16 = _load(
    "360.py", "2d02ba17a4f62db2e823c5b9f250c2fd08257a493f9cad78654507ccc2e2f28c"
)


def _select_n16(rows, num_kv_heads, use_prefill_config, num_columns, is_fp8=False):
    if is_fp8:
        raise NotImplementedError("This experiment is BF16 only")
    splits = 4 if rows == 512 else 1
    tiles = (num_columns + 15) // 16
    return 16, 2, tiles, min(splits, tiles)


_n16._QSA_BLOCK_M = 16
_n16._QSA_CONTIGUOUS_K = True
_n16._QSA_STAGES = 2
_n16._select_config = _select_n16


def selected_shape(rows, blocks, requests):
    return (rows, blocks, requests) in ((512, 512, 512), (6177, 2712, 57)) or (
        rows == 16384 and blocks == 2712 and 1 <= requests <= 57
    )


def _eligible(q, k, v, indices, table, request):
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
    return selected_shape(q.shape[0], k.shape[0], table.shape[0])


def describe(rows, blocks, requests):
    plan = dict(_reference.describe(rows))
    if selected_shape(rows, blocks, requests):
        plan.update(path="360", bn=16, warps=2, stages=2)
    plan["selected_shape"] = selected_shape(rows, blocks, requests)
    return plan


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
    if _eligible(q, k_cache, v_cache, logical_indices, block_table, token_to_req):
        return _n16.qsa_sparse_paged_attention(
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
    return _reference.qsa_sparse_paged_attention(
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
