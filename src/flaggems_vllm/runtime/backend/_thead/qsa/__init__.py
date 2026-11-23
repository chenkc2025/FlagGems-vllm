# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""T-Head QSA dispatch for the currently validated input family."""

import importlib.util
import sys
from pathlib import Path

import torch


def _load(filename):
    name = "_flaggems_thead_qsa_" + filename[:-3]
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parent / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sail_config(num_rows, num_kv_heads, use_prefill_config, num_columns, is_fp8=False):
    # vLLM-SAIL qsa_select_config, commit 72cbab3b7bc4db60793098eec9bde4edc5f4ec25.
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


_bf16 = _load("403.py")
_fp8 = _load("632.py")
_fp8._select_config = _sail_config


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
        implementation = _bf16
    elif k_cache.dtype == v_cache.dtype and k_cache.dtype in (
        torch.uint8,
        torch.float8_e4m3fn,
    ):
        if k_cache.dtype == torch.float8_e4m3fn:
            # Same storage and shape; the PPU path decodes the bytes in Triton.
            k_cache = k_cache.view(torch.uint8)
            v_cache = v_cache.view(torch.uint8)
        implementation = _fp8
    else:
        raise NotImplementedError("QSA supports BF16 or E4M3FN KV")
    return implementation.qsa_sparse_paged_attention(
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
