# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Complete Hygon QSA contract; numerical execution requires an HCU."""

import math

import pytest
import torch

import flaggems_vllm


def make_case(rows=4, heads=6, dim=256, kv_dtype=torch.bfloat16):
    gen = torch.Generator().manual_seed(20261123 + rows + heads + dim)
    q = torch.randn((rows, heads, dim), generator=gen).to(torch.bfloat16)
    kv = torch.randn((8, 32, 1, dim * 2), generator=gen).to(torch.bfloat16)
    k, v = kv.split(dim, dim=-1)
    ks, vs = (1.0, 1.0) if kv_dtype == torch.bfloat16 else (0.5, 2.0)
    if kv_dtype != torch.bfloat16:
        k, v = (k.float() / ks).to(kv_dtype), (v.float() / vs).to(kv_dtype)
    gate = torch.randn(q.shape, generator=gen).to(torch.bfloat16)
    idx = torch.full((rows, 68), -1, dtype=torch.int32)
    for row in range(rows):
        count = (0, 1, 17, 67)[row % 4]
        idx[row, :count] = torch.arange(count) % 128
        if count > 2:
            idx[row, 1] = idx[row, 0]
            idx[row, 2] = 128  # Invalid logical page, masked by native reader.
        idx[row, -1] = count
    req = torch.arange(rows, dtype=torch.int32) % 4
    table = torch.arange(16, dtype=torch.int32).reshape(4, 4) % 8
    if rows >= 4:
        req[3] = -1
    return (q, k, v, idx, table, req), gate, ks, vs


def reference(q, k, v, packed, table, req, gate, ks=1.0, vs=1.0):
    output = torch.zeros(q.shape, dtype=torch.float32)
    k, v = k.float(), v.float()
    for row in range(q.shape[0]):
        request = int(req[row])
        if not 0 <= request < table.shape[0]:
            continue
        count = min(max(int(packed[row, -1]), 0), packed.shape[1] - 1)
        idx = packed[row, :count].long()
        idx = idx[(idx >= 0) & (idx // k.shape[1] < table.shape[1])]
        pages = table[request, idx // k.shape[1]].long()
        valid = (pages >= 0) & (pages < k.shape[0])
        pages, idx = pages[valid], idx[valid]
        if idx.numel() == 0:
            continue
        for head in range(q.shape[1]):
            kv_head = head // (q.shape[1] // k.shape[2])
            keys = k[pages, idx % k.shape[1], kv_head] * ks
            vals = v[pages, idx % v.shape[1], kv_head] * vs
            scores = (q[row, head].float() @ keys.T) / math.sqrt(q.shape[2])
            output[row, head] = torch.softmax(scores, 0) @ vals
    return output * torch.sigmoid(gate.view_as(q).float())


pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "hygon", reason="Hygon-specific frozen QSA branch"
)


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize(
    "kv_dtype", [torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn]
)
@pytest.mark.parametrize("rows,heads,dim", [(0, 6, 64), (4, 6, 256), (16, 3, 64)])
def test_qsa_sparse_paged_attention_complete(rows, heads, dim, kv_dtype, prefill):
    args, gate, ks, vs = make_case(rows, heads, dim, kv_dtype)
    expected = reference(*args, gate, ks, vs)
    args = tuple(t.to("cuda") for t in args)
    gate = gate.to("cuda").reshape(rows, heads * dim)
    out = torch.full(
        (rows, heads, dim * 2), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    out = out[..., :dim]
    returned = flaggems_vllm.qsa_sparse_paged_attention(
        *args, prefill, out=out, k_scale=ks, v_scale=vs, output_gate=gate
    )
    torch.cuda.synchronize()
    assert returned is out
    torch.testing.assert_close(out.float().cpu(), expected, atol=0.03, rtol=0.03)


@pytest.mark.parametrize("prefill", [False, True])
def test_qsa_sparse_paged_attention_gate_alias(prefill):
    args, gate, ks, vs = make_case()
    expected = reference(*args, gate, ks, vs)
    args = tuple(t.to("cuda") for t in args)
    gate = gate.to("cuda")
    returned = flaggems_vllm.qsa_sparse_paged_attention(
        *args, prefill, out=gate, k_scale=ks, v_scale=vs, output_gate=gate
    )
    torch.cuda.synchronize()
    assert returned is gate
    torch.testing.assert_close(gate.float().cpu(), expected, atol=0.03, rtol=0.03)


def test_qsa_sparse_paged_attention_requires_host_scales():
    args, gate, _, _ = make_case(kv_dtype=torch.float8_e5m2)
    args = tuple(t.to("cuda") for t in args)
    with pytest.raises(TypeError, match="host scalar"):
        flaggems_vllm.qsa_sparse_paged_attention(
            *args,
            False,
            k_scale=torch.ones(1, device="cuda"),
            v_scale=2.0,
            output_gate=gate.to("cuda"),
        )


def test_qsa_sparse_paged_attention_backend_registration():
    fn = flaggems_vllm.qsa_sparse_paged_attention
    assert fn.__module__.endswith(
        "runtime.backend._hygon.ops.qsa_sparse_paged_attention"
    )
    assert any(
        name == "qsa_sparse_paged_attention" and op is fn
        for name, op in flaggems_vllm._FULL_CONFIG
    )


@pytest.mark.parametrize("count", [-5, 0, 1, 67, 100])
def test_qsa_sparse_paged_attention_count_boundaries(count):
    args, gate, ks, vs = make_case()
    args[3][0, :-1] = torch.arange(67, dtype=torch.int32)
    args[3][0, -1] = count
    expected = reference(*args, gate, ks, vs)
    args = tuple(t.to("cuda") for t in args)
    out = flaggems_vllm.qsa_sparse_paged_attention(
        *args, False, k_scale=ks, v_scale=vs, output_gate=gate.to("cuda")
    )
    torch.testing.assert_close(out.float().cpu(), expected, atol=0.03, rtol=0.03)
