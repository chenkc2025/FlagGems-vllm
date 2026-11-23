"""BF16 experiment: delay V loads; preserve 378's dispatch and outputs."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


control = load("_qsa403_control", "378.py")
large = load("_qsa403_336_delayed", "401.py")
n16 = load("_qsa403_360_delayed", "402.py")

large._QSA_CONTIGUOUS_K = True
large._QSA_STAGES = 2
large._select_config = control._reference._select_config
n16._QSA_BLOCK_M = 16
n16._QSA_CONTIGUOUS_K = True
n16._QSA_STAGES = 2
n16._select_config = control._select_n16


def qsa_sparse_paged_attention(
    q,
    k,
    v,
    indices,
    table,
    request,
    use_prefill_config,
    out=None,
    k_scale=None,
    v_scale=None,
    *,
    output_gate,
):
    if control._eligible(q, k, v, indices, table, request):
        mod = n16
    elif q.shape[0] > 8:
        mod = large
    else:
        mod = control
    return mod.qsa_sparse_paged_attention(
        q,
        k,
        v,
        indices,
        table,
        request,
        use_prefill_config,
        out=out,
        k_scale=k_scale,
        v_scale=v_scale,
        output_gate=output_gate,
    )
