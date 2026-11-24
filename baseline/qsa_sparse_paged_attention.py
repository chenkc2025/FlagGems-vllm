# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""NVIDIA reference collection for QSA; inputs are synthetic recorded shapes."""
import importlib

import torch

OP_NAME = "qsa_sparse_paged_attention"
DTYPES = [torch.bfloat16, torch.float8_e4m3fn]
IS_INPLACE = False

# Recorded shape tuples: query rows, physical pages, request count.
_SHAPES = [
    (1, 512, 1),
    (1, 2712, 1),
    (2, 512, 2),
    (2, 2712, 2),
    (4, 512, 4),
    (4, 2712, 4),
    (8, 512, 8),
    (8, 2712, 8),
    (16, 512, 16),
    (16, 2712, 16),
    (24, 512, 24),
    (24, 2712, 24),
    (32, 512, 32),
    (32, 2712, 32),
    (40, 512, 40),
    (40, 2712, 40),
    (48, 512, 48),
    (48, 2712, 48),
    (56, 512, 56),
    (56, 2712, 56),
    (64, 512, 64),
    (64, 2712, 64),
    (72, 512, 72),
    (72, 2712, 72),
    (80, 512, 80),
    (80, 2712, 80),
    (88, 512, 88),
    (88, 2712, 88),
    (96, 512, 96),
    (96, 2712, 96),
    (104, 512, 104),
    (104, 2712, 104),
    (112, 512, 112),
    (112, 2712, 112),
    (120, 512, 120),
    (120, 2712, 120),
    (128, 512, 128),
    (128, 2712, 128),
    (136, 512, 136),
    (136, 2712, 136),
    (144, 512, 144),
    (144, 2712, 144),
    (152, 512, 152),
    (152, 2712, 152),
    (160, 512, 160),
    (160, 2712, 160),
    (168, 512, 168),
    (168, 2712, 168),
    (176, 512, 176),
    (176, 2712, 176),
    (184, 512, 184),
    (184, 2712, 184),
    (192, 512, 192),
    (192, 2712, 192),
    (200, 512, 200),
    (200, 2712, 200),
    (208, 512, 208),
    (208, 2712, 208),
    (216, 512, 216),
    (216, 2712, 216),
    (224, 512, 224),
    (224, 2712, 224),
    (232, 512, 232),
    (232, 2712, 232),
    (240, 512, 240),
    (240, 2712, 240),
    (248, 512, 248),
    (248, 2712, 248),
    (256, 512, 256),
    (256, 2712, 256),
    (272, 512, 272),
    (272, 2712, 272),
    (288, 512, 288),
    (288, 2712, 288),
    (304, 512, 304),
    (304, 2712, 304),
    (320, 512, 320),
    (320, 2712, 320),
    (336, 512, 336),
    (336, 2712, 336),
    (352, 512, 352),
    (352, 2712, 352),
    (368, 512, 368),
    (368, 2712, 368),
    (384, 512, 384),
    (384, 2712, 384),
    (400, 512, 400),
    (400, 2712, 400),
    (416, 512, 416),
    (416, 2712, 416),
    (432, 512, 432),
    (432, 2712, 432),
    (448, 512, 448),
    (448, 2712, 448),
    (451, 2712, 451),
    (464, 512, 464),
    (464, 2712, 464),
    (480, 512, 480),
    (480, 2712, 480),
    (496, 512, 496),
    (496, 2712, 496),
    (512, 512, 512),
    (512, 2712, 512),
    (902, 2712, 451),
    (2369, 2712, 54),
    (6177, 2712, 57),
    (16384, 2712, 1),
    (16384, 2712, 2),
    (16384, 2712, 3),
    (16384, 2712, 4),
    (16384, 2712, 5),
    (16384, 2712, 6),
    (16384, 2712, 7),
    (16384, 2712, 8),
    (16384, 2712, 9),
    (16384, 2712, 10),
    (16384, 2712, 11),
    (16384, 2712, 12),
    (16384, 2712, 13),
    (16384, 2712, 14),
    (16384, 2712, 15),
    (16384, 2712, 16),
    (16384, 2712, 17),
    (16384, 2712, 18),
    (16384, 2712, 19),
    (16384, 2712, 20),
    (16384, 2712, 21),
    (16384, 2712, 22),
    (16384, 2712, 23),
    (16384, 2712, 24),
    (16384, 2712, 25),
    (16384, 2712, 26),
    (16384, 2712, 27),
    (16384, 2712, 28),
    (16384, 2712, 29),
    (16384, 2712, 30),
    (16384, 2712, 31),
    (16384, 2712, 32),
    (16384, 2712, 33),
    (16384, 2712, 34),
    (16384, 2712, 35),
    (16384, 2712, 36),
    (16384, 2712, 37),
    (16384, 2712, 38),
    (16384, 2712, 39),
    (16384, 2712, 40),
    (16384, 2712, 41),
    (16384, 2712, 42),
    (16384, 2712, 43),
    (16384, 2712, 44),
    (16384, 2712, 45),
    (16384, 2712, 46),
    (16384, 2712, 47),
    (16384, 2712, 48),
    (16384, 2712, 49),
    (16384, 2712, 50),
    (16384, 2712, 51),
    (16384, 2712, 52),
    (16384, 2712, 53),
    (16384, 2712, 54),
    (16384, 2712, 55),
    (16384, 2712, 56),
    (16384, 2712, 57),
    (16384, 2712, 2048),
]


def native():
    try:
        module = importlib.import_module("vllm.models.qwen4_exp.nvidia.ops.qsa")
    except ImportError:
        return None
    return getattr(module, OP_NAME, None)


def grid():
    return [{"shape": list(shape)} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    rows, pages, requests = binding["shape"]
    q = torch.randn((rows, 6, 256), dtype=torch.bfloat16, device=device)
    gate = torch.randn_like(q)
    # Generate FP8 data on CPU: not every backend supports a device cast.
    if dtype == torch.float8_e4m3fn:
        kv = torch.randn((pages, 784, 1, 512), dtype=torch.bfloat16)
        kv = kv.to(torch.float8_e4m3fn).view(torch.uint8).to(device)
    elif dtype == torch.bfloat16:
        kv = torch.randn((pages, 784, 1, 512), dtype=dtype, device=device)
    else:
        raise ValueError("KV dtype must be BF16 or E4M3FN")
    table = torch.randint(pages, (requests, 88), dtype=torch.int32, device=device)
    req = torch.arange(rows, dtype=torch.int32, device=device).remainder(requests)
    ids = torch.randint(88 * 784, (rows, 2052), dtype=torch.int32, device=device)
    ids[:, 2048:2051] = -1
    ids[:, 2051] = 2048
    args = (q, kv[..., :256], kv[..., 256:], ids, table, req, True)
    kwargs = {
        "k_scale": 0.5 if dtype == torch.float8_e4m3fn else None,
        "v_scale": 1.25 if dtype == torch.float8_e4m3fn else None,
        "output_gate": gate,
    }
    return args, kwargs


def key_shape(binding):
    return [binding["shape"][0], 6, 256]


def config(binding, dtype):
    rows, pages, requests = binding["shape"]
    return {
        "inputs": {
            "q": {"shape": [rows, 6, 256], "dtype": str(torch.bfloat16)},
            "k_cache": {"shape": [pages, 784, 1, 256], "dtype": str(dtype)},
            "v_cache": {"shape": [pages, 784, 1, 256], "dtype": str(dtype)},
            "logical_indices": {"shape": [rows, 2052], "dtype": str(torch.int32)},
            "block_table": {"shape": [requests, 88], "dtype": str(torch.int32)},
            "token_to_req": {"shape": [rows], "dtype": str(torch.int32)},
            "output_gate": {"shape": [rows, 6, 256], "dtype": str(torch.bfloat16)},
            "use_prefill_config": {"scalar": True},
            "k_scale": {"scalar": 0.5 if dtype == torch.float8_e4m3fn else None},
            "v_scale": {"scalar": 1.25 if dtype == torch.float8_e4m3fn else None},
        },
        "outputs": {"out": {"shape": [rows, 6, 256], "dtype": str(torch.bfloat16)}},
        "dims": {"rows": rows, "pages": pages, "requests": requests},
    }
