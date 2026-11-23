# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License").
"""Same-input graph benchmark against upstream QSA with the vLLM-SAIL patch.

Run on a T-Head system with the matching vLLM, vLLM-SAIL, and FlagTree installed:
    python benchmark/qsa_sparse_paged_attention_thead.py
"""

import statistics

import torch

import flaggems_vllm


def main():
    if flaggems_vllm.vendor_name != "thead":
        raise RuntimeError("This benchmark requires a T-Head PPU")
    from vllm.models.qwen4_exp.nvidia.ops import qsa
    from vllm_sail.patch.performance import qsa_select_config

    qsa_select_config.install()
    for fp8 in (False, True):
        for rows in (1, 8, 128):
            torch.manual_seed(rows)
            device = flaggems_vllm.device
            q = torch.randn(rows, 6, 256, dtype=torch.bfloat16, device=device)
            gate = torch.randn_like(q)
            kv = torch.randn(512, 784, 1, 512, dtype=torch.bfloat16, device=device)
            if fp8:
                # Generate on CPU instead if the local runtime cannot cast to FP8.
                kv = kv.cpu().to(torch.float8_e4m3fn).view(torch.uint8).to(device)
            table = torch.randint(512, (rows, 88), dtype=torch.int32, device=device)
            req = torch.arange(rows, dtype=torch.int32, device=device)
            indices = torch.randint(
                88 * 784, (rows, 2052), dtype=torch.int32, device=device
            )
            indices[:, 2048:2051] = -1
            indices[:, 2051] = 2048
            args = (q, kv[..., :256], kv[..., 256:], indices, table, req, True)
            kwargs = {
                "k_scale": 0.5 if fp8 else None,
                "v_scale": 1.25 if fp8 else None,
                "output_gate": gate,
            }
            funcs = {
                "baseline": lambda: qsa.qsa_sparse_paged_attention(*args, **kwargs),
                "candidate": lambda: flaggems_vllm.qsa_sparse_paged_attention(
                    *args, **kwargs
                ),
            }
            reference, candidate = funcs["baseline"](), funcs["candidate"]()
            torch.testing.assert_close(candidate, reference, rtol=0.016, atol=0.02)
            for _ in range(3):
                for function in funcs.values():
                    function()
            torch.cuda.synchronize()
            graphs = {}
            outputs = {}
            for name, function in funcs.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs[name] = function()
                graphs[name] = graph
                graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(outputs["baseline"], reference)
            torch.testing.assert_close(outputs["candidate"], candidate)

            def elapsed(graph):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                for _ in range(10):
                    graph.replay()
                end.record()
                end.synchronize()
                return start.elapsed_time(end) / 10

            ratios = []
            for iteration in range(8):
                order = ("baseline", "candidate")
                if iteration % 2:
                    order = order[::-1]
                times = {name: elapsed(graphs[name]) for name in order}
                ratios.append(times["baseline"] / times["candidate"])
            print(f"fp8_kv={fp8} rows={rows} speedup={statistics.median(ratios):.5f}")


if __name__ == "__main__":
    main()
