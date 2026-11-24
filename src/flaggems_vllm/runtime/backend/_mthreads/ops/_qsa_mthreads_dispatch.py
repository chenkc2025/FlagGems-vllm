# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Mthreads dispatch for the complete QSA sparse paged attention interface."""

import torch

from flaggems_vllm.runtime.backend._mthreads.ops import (
    _qsa_mthreads_prefill,
    _qsa_mthreads_split,
)


def qsa_sparse_paged_attention(
    q,
    k_cache,
    v_cache,
    logical_indices,
    block_table,
    token_to_req,
    use_prefill_config,
    out=None,
    k_scale=None,
    v_scale=None,
    *,
    output_gate,
):
    if not (
        q.ndim == 3
        and q.shape[1:] == (6, 256)
        and q.dtype == torch.bfloat16
        and k_cache.ndim == 4
        and k_cache.shape[2:] == (1, 256)
        and k_cache.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and v_cache.dtype == k_cache.dtype
    ):
        from flaggems_vllm.ops.qsa_sparse_paged_attention import (
            qsa_sparse_paged_attention as generic_qsa,
        )

        return generic_qsa(
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            use_prefill_config,
            out,
            k_scale,
            v_scale,
            output_gate=output_gate,
        )
    if (
        use_prefill_config
        and q.ndim == 3
        and q.shape[0] > 32
        and q.shape[1:] == (6, 256)
        and q.dtype == torch.bfloat16
        and k_cache.ndim == 4
        and k_cache.shape[2:] == (1, 256)
        and k_cache.dtype == torch.bfloat16
        and v_cache.dtype == torch.bfloat16
    ):
        fn = _qsa_mthreads_prefill.qsa_sparse_paged_attention
    else:
        fn = _qsa_mthreads_split.qsa_sparse_paged_attention
    return fn(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        use_prefill_config,
        out,
        k_scale,
        v_scale,
        output_gate=output_gate,
    )
