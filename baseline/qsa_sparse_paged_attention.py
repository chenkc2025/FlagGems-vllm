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

"""NVIDIA vLLM QSA baseline for the repository's benchmark shape list.

Inputs match benchmark/test_qsa_sparse_paged_attention.py. Query and output are
BF16; fp8 in a shape describes the KV cache, not the query. The input data and
configuration flags are synthetic and are not a model-request replay.
"""

import importlib
import os
from pathlib import Path

import torch
import yaml

OP_NAME = "qsa_sparse_paged_attention"
DTYPES = [torch.bfloat16]
IS_INPLACE = False
_DEFAULT_SHAPES = Path(__file__).resolve().parents[1] / "benchmark" / "core_shapes.yaml"


def native():
    try:
        module = importlib.import_module("vllm.models.qwen4_exp.nvidia.ops.qsa")
    except ImportError:
        return None
    candidate = getattr(module, OP_NAME, None)
    return candidate if callable(candidate) else None


def grid():
    shape_file = Path(os.environ.get("FLAGGEMS_QSA_SHAPES_FILE", _DEFAULT_SHAPES))
    with shape_file.open(encoding="utf-8") as stream:
        shapes = yaml.safe_load(stream)[OP_NAME]["shapes"]
    return [{"shape": list(shape)} for shape in shapes]


def build_inputs(binding, dtype, device):
    rows, q_heads, kv_heads, page, prefill, requests, fp8 = binding["shape"]
    if dtype != torch.bfloat16:
        raise ValueError("QSA baseline requires BF16 query")
    torch.manual_seed(2)
    dim, pages_per_request, selection_width = 256, 65, 2051
    blocks = requests * pages_per_request
    q = torch.randn(rows, q_heads, dim, device=device, dtype=dtype)
    gate = torch.randn_like(q)
    kv = torch.randn(blocks, page, kv_heads, 2 * dim, device=device, dtype=dtype)
    k, v = kv.split(dim, dim=-1)
    table = torch.randperm(blocks, device=device).reshape(requests, pages_per_request)
    table = table.to(torch.int32)
    row_ids = torch.arange(rows, device=device, dtype=torch.int32)
    token_to_req = row_ids // ((rows + requests - 1) // requests)
    capacity = pages_per_request * page
    selected = torch.randperm(capacity, device=device)[:selection_width]
    selected = (selected[None, :] + row_ids[:, None]) % capacity
    counts = torch.where(row_ids % 3 == 1, 17, selection_width - row_ids % 4)
    ranks = torch.arange(selection_width, device=device)
    indices = torch.full(
        (rows, selection_width + 1), -1, device=device, dtype=torch.int32
    )
    indices[:, :-1] = torch.where(ranks < counts[:, None], selected, -1)
    indices[:, -1] = counts
    if fp8:
        k_scale, v_scale = 0.5, 2.0
        k = (k / k_scale).to(torch.float8_e4m3fn)
        v = (v / v_scale).to(torch.float8_e4m3fn)
    else:
        k_scale, v_scale = 1.0, 1.0
    return (q, k, v, indices, table, token_to_req), {
        "use_prefill_config": prefill,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "output_gate": gate,
    }


def key_shape(binding):
    # Include page size, request count, mode and cache dtype to avoid collisions.
    return list(binding["shape"])


def config(binding, dtype):
    rows, q_heads, kv_heads, page, prefill, requests, fp8 = binding["shape"]
    query_shape = [rows, q_heads, 256]
    kv_shape = [requests * 65, page, kv_heads, 256]
    query_dtype = str(dtype)
    kv_dtype = "torch.float8_e4m3fn" if fp8 else query_dtype
    return {
        "inputs": {
            "q": {"shape": query_shape, "dtype": query_dtype},
            "k_cache": {"shape": kv_shape, "dtype": kv_dtype},
            "v_cache": {"shape": kv_shape, "dtype": kv_dtype},
            "logical_indices": {"shape": [rows, 2052], "dtype": "torch.int32"},
            "block_table": {"shape": [requests, 65], "dtype": "torch.int32"},
            "token_to_req": {"shape": [rows], "dtype": "torch.int32"},
            "output_gate": {"shape": query_shape, "dtype": query_dtype},
            "use_prefill_config": {"scalar": prefill},
            "k_scale": {"scalar": 0.5 if fp8 else 1.0},
            "v_scale": {"scalar": 2.0 if fp8 else 1.0},
        },
        "outputs": {"out": {"shape": query_shape, "dtype": query_dtype}},
        "dims": {
            "shape": key_shape(binding),
            "input_source": "benchmark shape, synthetic values and metadata",
        },
    }
