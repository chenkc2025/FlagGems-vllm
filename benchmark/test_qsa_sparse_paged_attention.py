# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""QSA benchmark using the repository benchmark framework, without graphs."""

import importlib

import pytest
import torch

import flaggems_vllm
from baseline import qsa_sparse_paged_attention as reference

from . import base


def native_op():
    """The upstream vLLM reference used to collect NVIDIA data."""
    return reference.native()


def vendor_op():
    """Use upstream QSA with the SAIL configuration patch on T-Head."""
    if flaggems_vllm.vendor_name != "thead":
        return None
    op = native_op()
    if op is None:
        return None
    try:
        patch = importlib.import_module("vllm_sail.patch.performance.qsa_select_config")
    except ImportError:
        return None
    patch.install()
    # Do not silently benchmark the unpatched upstream implementation.
    mod = importlib.import_module("vllm.models.qwen4_exp.nvidia.ops.qsa")
    for rows in (1, 8, 128, 512, 2369):
        if mod._select_config(rows, 1, True, 2051, False) != patch.ppu_select_config(
            rows, 1, True, 2051, False
        ):
            raise RuntimeError("vLLM-SAIL QSA configuration patch is not active")
    return op


class QsaBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [tuple(binding["shape"]) for binding in reference.grid()]
        self.shape_desc = "query rows, physical pages, requests"

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            args, kwargs = reference.build_inputs({"shape": shape}, dtype, self.device)
            yield (*args, kwargs)


@pytest.mark.qsa_sparse_paged_attention
def test_qsa_sparse_paged_attention():
    if flaggems_vllm.vendor_name != "thead":
        pytest.skip("T-Head backend benchmark")
    baseline = vendor_op()
    if baseline is None:
        pytest.skip("vLLM QSA or vLLM-SAIL configuration patch unavailable")
    QsaBenchmark(
        op_name="qsa_sparse_paged_attention",
        torch_op=baseline,
        gems_op=flaggems_vllm.qsa_sparse_paged_attention,
        dtypes=[torch.bfloat16, torch.float8_e4m3fn],
    ).run()
