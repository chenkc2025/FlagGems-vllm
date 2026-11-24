# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""MetaX's full vLLM-style QSA API and independent FP32 reference."""

import inspect

import pytest
import torch

import flaggems_vllm
from baseline import qsa_sparse_paged_attention as inputs

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "metax", reason="MetaX only"
)


def _gather_float(cache, physical, offset):
    if cache.dtype == torch.float8_e4m3fn:
        return cache.view(torch.uint8)[physical, offset, 0, :].view(cache.dtype).float()
    return cache[physical, offset, 0, :].float()


def _reference(args, kwargs):
    q, k, v, packed, table, requests = args
    result = torch.empty_like(q)
    columns = torch.arange(packed.shape[1] - 1, device=q.device)
    page_size = k.shape[1]
    capacity = table.shape[1] * page_size
    for start in range(0, q.shape[0], 8):
        end = min(start + 8, q.shape[0])
        logical = packed[start:end, :-1].long()
        req = requests[start:end].long()
        valid = (
            (columns[None, :] < packed[start:end, -1, None])
            & (logical >= 0)
            & (logical < capacity)
            & (req[:, None] >= 0)
            & (req[:, None] < table.shape[0])
        )
        safe = logical.clamp(0, capacity - 1)
        physical = table[
            req.clamp(0, table.shape[0] - 1)[:, None], safe // page_size
        ].long()
        valid &= (physical >= 0) & (physical < k.shape[0])
        physical = physical.clamp(0, k.shape[0] - 1)
        keys = _gather_float(k, physical, safe % page_size) * kwargs["k_scale"]
        values = _gather_float(v, physical, safe % page_size) * kwargs["v_scale"]
        scores = (
            torch.bmm(q[start:end].float(), keys.transpose(1, 2)) * q.shape[2] ** -0.5
        )
        scores.masked_fill_(~valid[:, None, :], -float("inf"))
        scores = torch.where(valid.any(1)[:, None, None], scores, 0.0)
        probabilities = torch.softmax(scores, -1).masked_fill(~valid[:, None, :], 0.0)
        attention = torch.bmm(probabilities, values).to(q.dtype)
        result[start:end] = (
            attention.float() * torch.sigmoid(kwargs["output_gate"][start:end].float())
        ).to(q.dtype)
    return result


@pytest.mark.parametrize("dtype", inputs.DTYPES)
@pytest.mark.parametrize("prefill", (True, False), ids=("prefill", "decode"))
def test_qsa_sparse_paged_attention(dtype, prefill):
    op = flaggems_vllm.qsa_sparse_paged_attention
    params = inspect.signature(op).parameters
    assert tuple(params) == inputs.PARAMETERS
    assert params["output_gate"].kind == inspect.Parameter.KEYWORD_ONLY
    binding = dict(tokens=33, pages=88, requests=1, calls=1, use_prefill_config=prefill)
    args, kwargs = inputs.build_inputs(binding, dtype, flaggems_vllm.device)
    q, _, _, indices, table, mapping = args
    indices[0, -1] = 0
    indices[1, -1] = 1
    indices[2, 0] = -1
    mapping[3] = -1
    table[0, 0] = -1
    if dtype == torch.float8_e4m3fn:
        kwargs.update(k_scale=0.37, v_scale=1.13)
    expected = _reference(args, kwargs)
    out = torch.empty((*q.shape[:-1], q.shape[-1] * 2), device=q.device, dtype=q.dtype)[
        ..., : q.shape[-1]
    ]
    checked = op(*args, **kwargs)
    written = op(*args, **kwargs, out=out)
    assert written is out
    torch.testing.assert_close(checked, expected, atol=0.02, rtol=0.016)
    torch.testing.assert_close(written, expected, atol=0.02, rtol=0.016)
