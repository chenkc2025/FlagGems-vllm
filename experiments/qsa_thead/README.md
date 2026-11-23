# T-Head QSA experiment

The independently benchmarked Triton QSA candidates are kept under
`src/flaggems_vllm/runtime/backend/_thead/qsa/`. The new
`flaggems_vllm.qsa_sparse_paged_attention` entry point dispatches BF16 and
FP8 KV through those candidates with the upstream QSA signature. Other
backends explicitly raise `NotImplementedError`.

- BF16 Q and BF16 KV: `403.py` is the measured entry point. It loads the
  neighboring numbered files, whose dispatch and constants must remain intact.
- BF16 Q and E4M3FN FP8 KV stored as `uint8`: `632.py` is the measured
  candidate. The caller supplies the same PPU launch-config selector as the
  baseline. Its inline byte decoder is inherited from the pinned upstream QSA.

The PPU baseline used the upstream vLLM QSA at commit
`8b10bda57ceab3a741d2fb3718af48cef691023e`, with vLLM-SAIL's
`qsa_select_config` patch at commit `72cbab3b7bc4db60793098eec9bde4edc5f4ec25`.
The comparison loaded those components in a standalone process with import
shims; it did not execute the complete model or plugin initialization.

On PPU-ZW810E, a private FlagTree build at commit
`149392a652c0e0e1313a642d4c2057b39e990571` was used for both sides. The
164 recorded BF16 shape tuples were tested with synthetic inputs and metadata;
FP8 reused those same tuples and generated synthetic FP8 KV, not FP8 traces.
The measurement included attention and output gate in graph replay, with
allocations during graph capture. Both sweeps set `use_prefill_config=True`.
Each passed sampled CPU reference, complete output comparison, and graph output
checks. Arithmetic mean baseline/candidate ratios were **1.256364x BF16** and
**1.051691x FP8 KV**; these are not full-model or separate decode results.

The integrated entry point was additionally tested on PPU-ZW810E against the
old candidate and the baseline. Twenty checks covered BF16/FP8 KV, both
prefill-config values, an eight-row count-boundary case, and four representative
recorded shapes. Sampled CPU reference, complete output comparison, output
buffer identity, and FP8 byte/float8 views passed. For the 16 timed cases,
the paired old/integrated Graph-replay ratios ranged from 0.99579 to 1.00310
for BF16 and 0.99827 to 1.00236 for FP8. The PPU integration test used the
real FlagGems-vLLM package with standalone upstream/SAIL shims; it did not run
the complete plugin or model. The new repository test and benchmark have not
themselves been executed on PPU.

The 164-shape performance measurements above describe the original standalone
entry points; the newly integrated path has not been retested on all 164
shapes. BF16 supports only the recorded input family and raises
`NotImplementedError` otherwise. Invalid metadata, scale variants, nonstandard
layouts, and all native vLLM interface edge cases require additional testing.

The experiments have limited shape/layout and metadata coverage and are not
production-ready. The FP8 byte decoder inherits the upstream behavior mapping
the two E4M3FN NaN encodings to finite values. Tests used finite inputs and do
not establish complete NaN or packed-index edge semantics. Do not use these
results as proof of model-level routing or compatibility with other backends.
