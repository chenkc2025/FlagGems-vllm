# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Complete-entry comparison using Benchmark's non-Graph --mode kernel.

Both Hygon paths include the SAME packed-count/host-scale/gate adapters around
their native attention. This is an adapted vendor full entry, not a claim that
the vendor natively implements the vLLM v0.32.0 signature. Input construction
and preallocated output setup are outside timing; split/merge stays inside.
Historical native-only Graph speedups are not results of this benchmark.
"""

import importlib
from functools import lru_cache, partial

import pytest
import torch

import flaggems_vllm
from baseline import qsa_sparse_paged_attention as inputs

from . import base


def native():
    """NVIDIA original full API, or None; never torch-compute fallback."""
    return inputs.native()


@lru_cache(maxsize=1)
def vendor_native():
    """Resolve the unchanged Hygon BF16 and FP8 FlashAttention entry points."""
    if flaggems_vllm.vendor_name != "hygon":
        return None
    try:
        module = importlib.import_module("flash_attn")
        return (
            module.sparse_gqa_paged_attn_func,
            module.sparse_gqa_paged_attn_fp8_func,
        )
    except (ImportError, AttributeError):
        return None


@lru_cache(maxsize=1)
def vendor():
    """Bind native readers once, outside the timed function."""
    readers = vendor_native()
    if readers is None:
        return None
    module = importlib.import_module(
        "flaggems_vllm.runtime.backend._hygon.ops.qsa_sparse_paged_attention"
    )
    return partial(
        module._with_native_backend,
        native_bf16=readers[0],
        native_fp8=readers[1],
    )


class QSASparsePagedAttentionBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "qsa_sparse_paged_attention",
            vendor(),
            [torch.bfloat16, torch.float8_e5m2],
            gems_op=flaggems_vllm.qsa_sparse_paged_attention,
        )

    def get_input_iter(self, dtype):
        for binding in inputs.grid():
            args, kwargs = inputs.build_inputs(binding, dtype, self.device)
            yield (*args, kwargs)


@pytest.mark.skipif(flaggems_vllm.vendor_name != "hygon", reason="Hygon only")
def test_qsa_sparse_paged_attention_benchmark():
    if vendor_native() is None:
        pytest.skip("Native vendor QSA unavailable; no substitute baseline")
    QSASparsePagedAttentionBenchmark().run()
