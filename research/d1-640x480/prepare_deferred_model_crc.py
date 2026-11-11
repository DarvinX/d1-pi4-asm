#!/usr/bin/env python3
"""Inventory raw d1 tensors that already have validated packed panels."""

import argparse
import json
import mmap
from pathlib import Path
import re
import zlib

from prepare_embedding_crc import END, FIRST, MODEL_BYTES, MODEL_CRC


PROJECTOR_BYTES = 203_031_584
PROJECTOR_CRC = 0xFBEF6164
DESCRIPTORS = r"\.quad (\d+)\n\.word (\d+),(\d+),(\d+),(\d+)\n\.quad (\d+)"


def indexed_tensors(path):
    source = path.read_text().split("layer_tensors:", 1)[1]
    fields = re.findall(DESCRIPTORS, source)
    if len(fields) != 24 * 18:
        raise ValueError("Expected 24 layers and 18 descriptors per layer")
    selected = []
    for index, values in enumerate(fields):
        offset, cols, rows, kind, _, length = map(int, values)
        if kind == 12 and cols and rows and cols % 256 == 0 and rows % 64 == 0:
            selected.append((offset, offset + length, kind))
        elif kind == 14 and index % 18 == 6 and index // 18 < 23:
            if (cols, rows) != (7680, 2048):
                raise ValueError("Unexpected packed Q6 tensor geometry")
            selected.append((offset, offset + length, kind))
    if sum(kind == 12 for _, _, kind in selected) != 125 or sum(kind == 14 for _, _, kind in selected) != 11:
        raise ValueError("Descriptors do not match the pinned packed panels")
    return sorted(selected)


def indexed_projector(path):
    tensors = [tuple(map(int, fields)) for fields in re.findall(DESCRIPTORS, path.read_text())]
    selected = sorted((offset, offset + length, kind) for offset, _, _, kind, _, length in tensors if kind == 8)
    if len(selected) != 84 or sum(end - start for start, end, _ in selected) != 199_426_048:
        raise ValueError("Descriptors do not match the pinned vision and merger panels")
    return selected


def eager_segments(deferred, reserved, total_bytes):
    cursor = 0
    ranges = []
    for start, end in sorted(reserved + [(a, b) for a, b, _ in deferred]):
        if not cursor <= start < end <= total_bytes or (start | end) % 8:
            raise ValueError("Unexpected alignment or overlapping model tensor")
        if cursor != start:
            ranges.append((cursor, start))
        cursor = end
    if cursor != total_bytes:
        ranges.append((cursor, total_bytes))
    return ranges


def write_assembly(path, eager, eager_crc, deferred, prefix):
    constants = prefix.upper()
    lines = [
        f".equ {constants}_EAGER_CRC,0x{eager_crc:08x}",
        f".equ {constants}_EAGER_COUNT,{len(eager)}",
        f".equ {constants}_TENSOR_COUNT,{len(deferred)}",
        ".balign 16",
        f"{prefix}_eager_table:",
    ]
    lines.extend(f"    .word {start},{end - start}" for start, end in eager)
    lines.extend([".balign 16", f"{prefix}_tensor_table:"])
    for row in deferred:
        start = row["offset"]
        lines.append(f"    .word {start},{start + row['bytes']},0x{row['crc32']:08x},0")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--assembly", type=Path)
    parser.add_argument("--projector", action="store_true")
    args = parser.parse_args()
    if args.projector:
        total_bytes, full_crc = PROJECTOR_BYTES, PROJECTOR_CRC
        reserved, prefix = [], "d1_defer_vis"
        deferred = indexed_projector(args.index)
    else:
        total_bytes, full_crc = MODEL_BYTES, MODEL_CRC
        reserved, prefix = [(FIRST, END)], "d1_defer"
        deferred = indexed_tensors(args.index)
    eager = eager_segments(deferred, reserved, total_bytes)
    deferred_rows = []
    with args.model.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if len(data) != total_bytes or zlib.crc32(data) != full_crc:
            raise ValueError("Expected the pinned d1-2B checkpoint")
        with memoryview(data) as model:
            eager_crc = 0
            for start, end in eager:
                eager_crc = zlib.crc32(model[start:end], eager_crc)
            for start, end, kind in deferred:
                deferred_rows.append({"offset": start, "bytes": end - start, "kind": kind,
                                      "crc32": zlib.crc32(model[start:end])})
    eager_bytes = sum(end - start for start, end in eager)
    deferred_bytes = sum(row["bytes"] for row in deferred_rows)
    embedding_bytes = sum(end - start for start, end in reserved)
    if eager_bytes + deferred_bytes + embedding_bytes != total_bytes:
        raise ValueError("Model segments do not cover the original file")
    report = {"model_bytes": total_bytes, "embedding_bytes": embedding_bytes,
              "eager_bytes": eager_bytes, "eager_crc32": eager_crc,
              "eager": [{"offset": start, "bytes": end - start} for start, end in eager],
              "deferred_bytes": deferred_bytes, "deferred": deferred_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.assembly:
        write_assembly(args.assembly, eager, eager_crc, deferred_rows, prefix)


if __name__ == "__main__":
    main()
