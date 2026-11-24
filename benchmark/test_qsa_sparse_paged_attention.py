# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""MetaX versus the vendor's complete QSA entry using the benchmark framework."""

import importlib.util
import os
from functools import lru_cache

import pytest

import flaggems_vllm
from baseline import qsa_sparse_paged_attention as inputs

from . import base, consts
from .conftest import Config


def native():
    return inputs.native()


@lru_cache(maxsize=1)
def vendor():
    # A fixed snapshot may be supplied for standalone development comparisons.
    path = os.environ.get("FLAGGEMS_QSA_VENDOR_SOURCE")
    if path:
        spec = importlib.util.spec_from_file_location("qsa_metax_vendor", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Invalid vendor QSA source: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.qsa_sparse_paged_attention
    return inputs.resolve("vllm_metax.models.qwen4_exp.metax.ops.qsa")


class QSABenchmark(base.Benchmark):
    def __init__(self, vendor_op, prefill):
        super().__init__(
            inputs.OP_NAME,
            vendor_op,
            inputs.DTYPES,
            is_inplace=inputs.IS_INPLACE,
            gems_op=flaggems_vllm.qsa_sparse_paged_attention,
        )
        self.prefill = prefill

    def get_input_iter(self, dtype):
        for binding in inputs.grid():
            if binding["use_prefill_config"] != self.prefill:
                continue
            if Config.bench_level == consts.BenchLevel.CORE and binding["id"] not in (
                1,
                14,
                38,
            ):
                continue
            args, kwargs = inputs.build_inputs(binding, dtype, self.device)
            yield (*args, kwargs)


@pytest.mark.skipif(flaggems_vllm.vendor_name != "metax", reason="MetaX only")
@pytest.mark.parametrize("prefill", (True, False), ids=("prefill", "decode"))
def test_qsa_sparse_paged_attention_benchmark(prefill):
    vendor_op = vendor()
    if vendor_op is None:
        pytest.skip("Vendor QSA unavailable; benchmark requires the native baseline")
    QSABenchmark(vendor_op, prefill).run()
