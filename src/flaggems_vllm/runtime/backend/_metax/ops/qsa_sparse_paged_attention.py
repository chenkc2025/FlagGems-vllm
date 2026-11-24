# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import torch

from . import _qsa_bf16, _qsa_fp8


def _select_launch_config(rows, is_fp8):
    # Frozen experiment 164 policy, not runtime autotuning.
    cfg = dict(
        qk_n=128,
        pv_n=64,
        qk_warps=4,
        pv_warps=4,
        out_dim=128,
        k_dn=True,
        qk_mode="chunked",
        qk_chunk=64,
        pv_splits=1,
    )
    if rows <= 8:
        cfg.update(pv_splits=8)
    elif rows <= 256:
        cfg.update(pv_splits=4)
        if rows == 32 and is_fp8:
            cfg.update(qk_n=64, pv_n=128)
    elif rows <= 512:
        if not is_fp8:
            cfg.update(qk_n=64, pv_splits=4)
    else:
        cfg.update(qk_n=256)
    return cfg


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    use_prefill_config: bool,
    out: torch.Tensor | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    *,
    output_gate: torch.Tensor,
) -> torch.Tensor:
    """Complete QSA with the native vLLM signature.

    BF16 Q[T,6,256], one KV head, BF16 or E4M3FN KV. Private kernel
    parameters use the same measured policy for prefill and decode.
    """
    fp8 = k_cache.dtype == torch.float8_e4m3fn
    implementation = _qsa_fp8 if fp8 else _qsa_bf16
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
        **_select_launch_config(q.shape[0], fp8),
    )
