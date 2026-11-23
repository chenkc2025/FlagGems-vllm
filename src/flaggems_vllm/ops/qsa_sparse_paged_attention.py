# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vLLM QSA-compatible entry point, currently implemented for T-Head PPU."""

from flaggems_vllm import runtime


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
    if runtime.device.vendor_name != "thead":
        raise NotImplementedError("QSA currently has only a T-Head implementation")
    from flaggems_vllm.runtime.backend._thead.qsa import qsa_sparse_paged_attention

    return qsa_sparse_paged_attention(
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
    )
