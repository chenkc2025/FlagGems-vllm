# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License").

import pytest
import torch

import flaggems_vllm


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("prefill", [False, True])
def test_thead_qsa_against_cpu(fp8, prefill):
    if flaggems_vllm.vendor_name != "thead":
        pytest.skip("T-Head QSA test")
    torch.manual_seed(17)
    rows, pages, width = 4, 4, 88
    q = torch.randn(rows, 6, 256, dtype=torch.bfloat16)
    gate = torch.randn_like(q)
    kv = torch.randn(pages, 784, 1, 512, dtype=torch.bfloat16)
    if fp8:
        kv = kv.to(torch.float8_e4m3fn)
    table = torch.randint(pages, (rows, width), dtype=torch.int32)
    req = torch.arange(rows, dtype=torch.int32)
    ids = torch.randint(width * 784, (rows, 2051), dtype=torch.int32)
    counts = [0, 17, 33, 2048]
    for row, count in enumerate(counts):
        ids[row, count:] = -1
    packed = torch.cat([ids, torch.tensor(counts, dtype=torch.int32)[:, None]], dim=1)
    expected = torch.zeros_like(q)
    for row, count in enumerate(counts):
        if not count:
            continue
        logical = ids[row, :count].long()
        physical = table[row, logical // 784].long() * 784 + logical % 784
        if fp8:
            values = kv.view(torch.uint8).reshape(-1, 512)[physical]
            values = values.view(torch.float8_e4m3fn).float()
        else:
            values = kv.reshape(-1, 512)[physical].float()
        keys, values = values[:, :256], values[:, 256:]
        k_scale, v_scale = (0.5, 1.25) if fp8 else (1.0, 1.0)
        attention = torch.softmax(q[row].float() @ (keys * k_scale).T / 16, dim=-1)
        attention = (attention @ (values * v_scale)).to(torch.bfloat16)
        expected[row] = (attention.float() * torch.sigmoid(gate[row].float())).to(
            torch.bfloat16
        )
    q, gate, kv, packed, table, req = [
        value.to(flaggems_vllm.device) for value in (q, gate, kv, packed, table, req)
    ]
    out = torch.empty_like(q)
    actual = flaggems_vllm.qsa_sparse_paged_attention(
        q,
        kv[..., :256],
        kv[..., 256:],
        packed,
        table,
        req,
        prefill,
        out=out,
        k_scale=0.5 if fp8 else None,
        v_scale=1.25 if fp8 else None,
        output_gate=gate,
    )
    assert actual is out
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.016, atol=0.02)
