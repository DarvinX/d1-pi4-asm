#!/usr/bin/env python3
"""Prepare checkpoint-specific vision assets; never read or cache images."""

import argparse
import hashlib
from pathlib import Path
import sys
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from prepare import Reader

PROJECTOR_SHA256 = "62cc6d74fd1e5b1d6b48518566bf4c0cfd2468079a0c096ea323b74eed01e0d1"
CONV_BLOCKS = (0, 1, 3, 5, 7, 9, 11, 13)
BLOCK = (
    ("ln1", 1024, 1), ("ln2", 1024, 1), ("ffn_gate", 1024, 2560),
    ("ffn_up", 1024, 2560), ("ffn_down", 2560, 1024), ("conv2d_in_proj", 1024, 3072),
    ("conv2d_out_proj", 1024, 1024), ("attn_q", 1024, 1024), ("attn_k", 1024, 512),
    ("attn_v", 1024, 512), ("attn_out", 1024, 1024), ("attn_q_norm", 64, 1),
    ("attn_k_norm", 64, 1),
)


def tensor(reader, name, shape, kind):
    offset, actual, type_id = reader.tensors[name]
    if tuple(actual) != shape or type_id != kind:
        raise ValueError(f"unexpected {name}: {actual}, type {type_id}")
    dtype = {0: "<f4", 1: "<f2"}[kind]
    return np.ndarray(shape[::-1], dtype, reader.data, reader.base + offset)


def descriptor(reader, name, width, rows):
    offset, shape, kind = reader.tensors[name]
    expected_shape = [width] if rows == 1 else [width, rows]
    expected_kind = 0 if rows == 1 else 8
    if shape != expected_shape or kind != expected_kind:
        raise ValueError(f"unexpected {name}: {shape}, type {kind}")
    row_size = width * 4 if kind == 0 else (width // 32) * 34
    return [f".quad {reader.base + offset}", f".word {width},{rows},{kind},{row_size}",
            f".quad {row_size * rows}"]


def emit_index(reader, out):
    asm = [
        f".equ D1_VISION_BYTES,{len(reader.data)}",
        f".equ D1_VISION_CRC,{zlib.crc32(reader.data)}",
        ".section .rodata", ".balign 64",
        "d1_vis_stem_asset:", f'.incbin "{out / "stem.bin"}"',
        ".balign 64", "d1_vis_dw_asset:", f'.incbin "{out / "depthwise.bin"}"',
        ".balign 64", "d1_vis_rope_asset:", f'.incbin "{out / "rope.bin"}"',
        ".balign 32", "d1_vis_stem_desc:", ".quad 0", ".word 768,1024,0,3072",
        ".quad 3145728", "d1_vis_block_desc:",
    ]
    for index in range(14):
        for slot, (field, width, rows) in enumerate(BLOCK):
            used = slot < 5 or (slot in (5, 6) if index in CONV_BLOCKS else slot > 6)
            name = f"v.blk.{index}.{field}.weight"
            asm.extend(descriptor(reader, name, width, rows) if used else [".zero 32"])
    for label, name, width, rows in (
        ("gamma", "mm.input_norm.weight", 1024, 1), ("beta", "mm.input_norm.bias", 1024, 1),
        ("up", "mm.0.weight", 4096, 4096), ("up_bias", "mm.0.bias", 4096, 1),
        ("down", "mm.2.weight", 4096, 2048), ("down_bias", "mm.2.bias", 2048, 1),
    ):
        asm.extend([f"d1_vis_{label}_desc:", *descriptor(reader, name, width, rows)])
    (out / "vision.inc").write_text("\n".join(asm) + "\n")


def emit_rope(out):
    step = np.float32(1000000.0 ** (-2.0 / 32))
    table = np.empty((1200, 2, 32), dtype="<f4")
    for i in range(1200):
        y, x = divmod(i, 40)
        for section, coord in enumerate((y, x)):
            angle = np.float32(coord)
            for lane in range(16 * section, 16 * (section + 1)):
                table[i, 0, lane] = np.cos(angle)
                table[i, 1, lane] = np.sin(angle)
                angle *= step
    table.tofile(out / "rope.bin")


def prepare(path, out):
    reader = Reader(path)
    if hashlib.sha256(reader.data).hexdigest() != PROJECTOR_SHA256:
        raise ValueError("expected the pinned d1-2B Q8_0 projector")
    out.mkdir(parents=True, exist_ok=True)
    shape = (16, 16, 3, 1024)
    first = tensor(reader, "v.patch_embd.weight", shape, 1)
    second = tensor(reader, "v.patch_embd.weight.1", shape, 1)
    norm = tensor(reader, "v.embedding_norm.weight", (1024,), 0)
    with (out / "stem.bin").open("wb") as asset:
        asset.write(np.add(first, second, dtype=np.float32).astype("<f4").tobytes())
        asset.write(norm.tobytes())
    with (out / "depthwise.bin").open("wb") as asset:
        for index in CONV_BLOCKS:
            name = f"v.blk.{index}.conv2d_spatial.weight"
            taps = tensor(reader, name, (3, 3, 1, 1024), 0)
            asset.write(taps.reshape(1024, 9).T.tobytes(order="C"))
    emit_rope(out)
    emit_index(reader, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projector", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    prepare(args.projector, args.out)
