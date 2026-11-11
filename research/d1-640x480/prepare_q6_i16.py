#!/usr/bin/env python3
"""Write the checkpoint-bound signed-16 FFN-down panels for the Pi."""

import argparse
import hashlib
import os
from pathlib import Path
import struct
import sys
import tempfile
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from prepare import EXPECTED_D1, Reader


ROWS, WIDTH, BLOCK, STRIPE, PANEL, TILE = 2048, 7680, 256, 512, 96, 8
BLOCKS = WIDTH // BLOCK
PACKED = np.dtype([("base", "<u2", (TILE,)), ("coefficient", "<i2", (BLOCK, TILE))])
MAGIC = b"D1Q6I16\x01"
HEADER_BYTES = 4096


def pack_panel(source):
    k = np.arange(BLOCK)
    half, group, lane = k // 128, k // 32 % 4, k % 32
    low = (source[..., half * 64 + group % 2 * 32 + lane] >> (group // 2 * 4)) & 15
    high = (source[..., 128 + half * 32 + lane] >> (group * 2)) & 3
    codes = (low | (high << 4)).astype(np.int16) - 32
    scales = source[..., 192:208].view(np.int8)
    coeff = codes * scales[..., k // 16]
    base = source[..., 208:210].copy().view("<u2").reshape(len(source), BLOCKS)
    tiles = len(source) // TILE
    packed = np.empty((BLOCKS, tiles), dtype=PACKED)
    packed["base"] = base.reshape(tiles, TILE, BLOCKS).transpose(2, 0, 1)
    packed["coefficient"] = coeff.reshape(tiles, TILE, BLOCKS, BLOCK).transpose(2, 0, 3, 1)
    return packed


def prepare(model, output):
    reader = Reader(model)
    digest = hashlib.sha256(reader.data).hexdigest()
    if digest != EXPECTED_D1:
        raise ValueError("Expected the pinned d1 Q4_K_M checkpoint")
    tensors = []
    for layer in range(23):
        offset, shape, kind = reader.tensors[f"blk.{layer}.ffn_down.weight"]
        if kind == 14:
            if shape != [WIDTH, ROWS]:
                raise ValueError(f"Unexpected Q6_K shape in layer {layer}")
            tensors.append((layer, reader.base + offset))
    if len(tensors) != 11 or PACKED.itemsize != 4112:
        raise ValueError("Unexpected d1 compact layer count or tile size")
    if output.exists():
        raise FileExistsError(output)
    model_crc = zlib.crc32(reader.data)
    payload_crc = 0
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(bytes(HEADER_BYTES))
            for _, offset in tensors:
                source = np.ndarray((ROWS, BLOCKS, 210), dtype=np.uint8, buffer=reader.data, offset=offset)
                for stripe in range(0, ROWS, STRIPE):
                    for first in range(stripe, stripe + STRIPE, PANEL):
                        end = min(first + PANEL, stripe + STRIPE)
                        panel = pack_panel(source[first:end])
                        payload_crc = zlib.crc32(panel, payload_crc)
                        stream.write(panel)
            stream.seek(0)
            stream.write(struct.pack("<8sII", MAGIC, model_crc, payload_crc))
            stream.flush()
            expected = HEADER_BYTES + len(tensors) * ROWS // TILE * BLOCKS * PACKED.itemsize
            if os.fstat(stream.fileno()).st_size != expected:
                raise ValueError("Invalid compact sidecar size")
            os.link(temporary, output)
        finally:
            temporary.unlink()
    print(f"Packed {len(tensors)} d1 layers, {expected} bytes, CRC {payload_crc:08x}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.model, args.output)
