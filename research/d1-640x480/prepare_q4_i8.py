#!/usr/bin/env python3
"""Build exact packed-nibble Q4_K panels for the pinned backbone."""

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
from prepare import EXPECTED_D1, Reader, SLOTS


PACKED = np.dtype([
    ("base", "<u2", (16,)), ("min_base", "<u2", (16,)),
    ("scale", "u1", (8, 16)), ("minima", "u1", (8, 16)), ("codes", "u1", (256, 8)),
])
MAGIC = b"D1Q4I8\x01\0"
HEADER_BYTES = 4096
FILE_BYTES = HEADER_BYTES + 735330304
CHANNELS = np.arange(256)


def pack_rows(source):
    rows, blocks, _ = source.shape
    lanes = source.reshape(rows // 16, 16, blocks, 144).transpose(0, 2, 1, 3)
    bases = lanes[..., :4].copy().view("<u2")
    metadata = lanes[..., 4:16]
    codes = (lanes[..., 16 + CHANNELS // 64 * 32 + CHANNELS % 32] >> (CHANNELS // 32 % 2 * 4)) & 15
    codes = codes.astype(np.uint8).swapaxes(-1, -2)
    packed = np.empty((rows // 16, blocks), dtype=PACKED)
    packed["base"] = bases[..., 0]
    packed["min_base"] = bases[..., 1]
    packed["scale"][..., :4, :] = (metadata[..., :4] & 63).swapaxes(-1, -2)
    packed["scale"][..., 4:, :] = ((metadata[..., 8:] & 15) | ((metadata[..., :4] >> 6) << 4)).swapaxes(-1, -2)
    packed["minima"][..., :4, :] = (metadata[..., 4:8] & 63).swapaxes(-1, -2)
    packed["minima"][..., 4:, :] = ((metadata[..., 8:] >> 4) | ((metadata[..., 4:8] >> 6) << 4)).swapaxes(-1, -2)
    packed["codes"] = codes[..., :8] | (codes[..., 8:] << 4)
    return packed


def prepare(model, output):
    reader = Reader(model)
    if hashlib.sha256(reader.data).hexdigest() != EXPECTED_D1:
        raise ValueError("Expected the pinned d1 Q4_K_M checkpoint")
    tensors = []
    for layer in range(24):
        for slot in SLOTS:
            descriptor = reader.tensors.get(f"blk.{layer}.{slot}.weight")
            if descriptor is None:
                continue
            offset, shape, kind = descriptor
            if kind == 12 and len(shape) == 2 and shape[0] % 256 == 0 and shape[1] % 64 == 0:
                tensors.append((reader.base + offset, *shape))
    if len(tensors) != 125 or PACKED.itemsize != 2368:
        raise ValueError("Unexpected Q4_K tensor count or panel size")
    if output.exists():
        raise FileExistsError(output)
    payload_crc = 0
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(bytes(HEADER_BYTES))
            for offset, width, rows in tensors:
                source = np.ndarray((rows, width // 256, 144), dtype=np.uint8, buffer=reader.data, offset=offset)
                for first in range(0, rows, 512):
                    panels = pack_rows(source[first:first + 512])
                    payload_crc = zlib.crc32(panels, payload_crc)
                    stream.write(panels)
            stream.seek(0)
            stream.write(struct.pack("<8sII", MAGIC, zlib.crc32(reader.data), payload_crc))
            stream.flush()
            if os.fstat(stream.fileno()).st_size != FILE_BYTES:
                raise ValueError("Invalid Q4_K sidecar size")
            os.link(temporary, output)
        finally:
            temporary.unlink()
    print(f"Packed {len(tensors)} Q4_K tensors, {FILE_BYTES} bytes, CRC {payload_crc:08x}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.model, args.output)
