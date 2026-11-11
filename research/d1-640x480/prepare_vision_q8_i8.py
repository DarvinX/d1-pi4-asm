#!/usr/bin/env python3
"""Transpose vision Q8_0 blocks into 16-output panels."""

import argparse
import hashlib
import os
from pathlib import Path
import struct
import tempfile
import zlib

import numpy as np

from prepare_vision import BLOCK, CONV_BLOCKS, PROJECTOR_SHA256, Reader


HEADER_BYTES = 4096
FFN_BYTES = 2785280
CONV_PAYLOAD_BYTES = 22282240
PANEL = np.dtype([("scales", "<u2", (16,)), ("pairs", "u1", (16, 16, 2))])
HALF_PANEL = np.dtype([("scales", "<u2", (8,)), ("pairs", "u1", (16, 8, 2))])
MERGER = (("mm.0.weight", 4096, 4096), ("mm.2.weight", 4096, 2048))


def pack(source, half_major=False):
    rows, groups, _ = source.shape
    blocks = source.reshape(rows // 16, 16, groups, 34).transpose(0, 2, 1, 3)
    panels = np.empty((rows // 16, groups), dtype=PANEL)
    panels["scales"] = blocks[..., :2].copy().view("<u2")[..., 0]
    panels["pairs"] = blocks[..., 2:].reshape(rows // 16, groups, 16, 16, 2).swapaxes(-3, -2)
    if half_major:
        halves = np.empty((rows // 16, 2, groups), dtype=HALF_PANEL)
        for half in range(2):
            lanes = slice(half * 8, (half + 1) * 8)
            halves["scales"][:, half] = panels["scales"][..., lanes]
            halves["pairs"][:, half] = panels["pairs"][..., lanes, :]
        return halves
    return panels


def tensor_names(early, compact=False, full=False):
    if early:
        for layer in range(8):
            if layer in CONV_BLOCKS:
                for name, width, rows in BLOCK[5:7]:
                    yield f"v.blk.{layer}.{name}.weight", width, rows
        if compact:
            return
        for layer in range(14 if full else 8):
            slots = BLOCK[2:5] if layer in CONV_BLOCKS else (*BLOCK[2:5], *BLOCK[7:11])
            if full and layer >= 8 and layer in CONV_BLOCKS:
                slots = (*slots, *BLOCK[5:7])
            for name, width, rows in slots:
                yield f"v.blk.{layer}.{name}.weight", width, rows
    else:
        for layer in range(8, 14):
            for name, width, rows in BLOCK[2:5]:
                yield f"v.blk.{layer}.{name}.weight", width, rows


def certify_halves(panels):
    if np.any(panels["scales"] & 0x8000):
        raise ValueError("Certificate bit requires all original scale signs to be clear")
    absolute_sums = np.abs(panels["pairs"].view(np.int8)).sum(axis=-3, dtype=np.int16)
    certified = np.all(absolute_sums <= 255, axis=(-1, -2))
    panels["scales"][..., 0] |= np.where(certified, np.uint16(0x8000), np.uint16(0))


def prepare(projector, output, early=False, compact=False, working_view=None, full=False,
            half_major=False, certified=False, merger_only=False):
    reader = Reader(projector)
    if hashlib.sha256(reader.data).hexdigest() != PROJECTOR_SHA256:
        raise ValueError("Expected the pinned Q8_0 projector")
    if (compact or working_view is not None) and not early:
        raise ValueError("Compact panels and working views require early resident weights")
    if full and (not early or compact or working_view is None):
        raise ValueError("Full resident panels require a working view and non-compact early mode")
    if half_major and not full:
        raise ValueError("Half-major panels require full resident mode")
    if certified and not half_major:
        raise ValueError("Weight certificates require the half-major layout")
    if merger_only and not certified:
        raise ValueError("The merger sidecar requires certified half-major six-bit panels")
    weights = Reader(working_view) if working_view is not None else reader
    if weights.tensors != reader.tensors:
        raise ValueError("Working view must preserve the pinned tensor index")
    if output.exists():
        raise FileExistsError(output)
    payload_crc = 0
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(bytes(HEADER_BYTES))
            payload_bytes = CONV_PAYLOAD_BYTES if compact else 99155968 if early else 18 * FFN_BYTES
            if full:
                payload_bytes = 172687360
            tensors = tensor_names(early, compact, full)
            if merger_only:
                tensors = MERGER
                payload_bytes = 26738688
            count = 0
            for name, width, rows in tensors:
                offset, shape, kind = weights.tensors[name]
                if shape != [width, rows] or kind != 8:
                    raise ValueError(f"Unexpected Q8_0 layout in {name}")
                source = np.ndarray((rows, width // 32, 34), dtype="u1",
                                    buffer=weights.data, offset=weights.base + offset)
                if working_view is not None:
                    codes = source[..., 2:].view(np.int8)
                    if codes.min() < -31 or codes.max() > 31:
                        raise ValueError(f"Working codes exceed the six-bit accumulation bound in {name}")
                if early and np.any(source[..., 2:] == 128):
                    raise ValueError(f"Weight code exceeds the resident tile's pair bound in {name}")
                panels = pack(source, half_major)
                if certified:
                    certify_halves(panels)
                payload_crc = zlib.crc32(panels, payload_crc)
                stream.write(panels)
                count += 1
            if stream.tell() != HEADER_BYTES + payload_bytes:
                raise ValueError("Unexpected Q8_0 panel size")
            magic = b"D1Q8RS\x01\0" if early else b"D1Q8I8\x01\0"
            if compact:
                magic = b"D1Q8C6\x01\0" if working_view is not None else b"D1Q8C8\x01\0"
            elif merger_only:
                magic = b"D1Q8M6\x01\0"
            elif certified:
                magic = b"D1Q8S6\x01\0"
            elif full:
                magic = b"D1Q8H6\x01\0" if half_major else b"D1Q8A6\x01\0"
            elif working_view is not None:
                magic = b"D1Q8E6\x01\0"
            stream.seek(0)
            stream.write(struct.pack("<8sII", magic, zlib.crc32(reader.data), payload_crc))
            stream.flush()
            os.link(temporary, output)
        finally:
            temporary.unlink()
    print(f"Packed {count} Q8_0 tensors, {HEADER_BYTES + payload_bytes} bytes, CRC {payload_crc:08x}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projector", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--early-resident", action="store_true")
    parser.add_argument("--compact-conv", action="store_true")
    parser.add_argument("--working-view", type=Path)
    parser.add_argument("--full-resident", action="store_true")
    parser.add_argument("--half-major", action="store_true")
    parser.add_argument("--certified", action="store_true")
    parser.add_argument("--merger-only", action="store_true")
    args = parser.parse_args()
    prepare(args.projector, args.output, args.early_resident, args.compact_conv,
            args.working_view, args.full_resident, args.half_major, args.certified, args.merger_only)
