#!/usr/bin/env python3
"""Generate exact embedding-row CRCs and other GGUF segment CRCs for the pinned d1 file."""

import argparse
import mmap
from pathlib import Path
import struct
import zlib


MODEL_BYTES = 1_099_629_888
MODEL_CRC = 0xD53CE5F4
FIRST = 8_257_856
ROWS = 128_000
STRIDE = 1680
END = FIRST + ROWS * STRIDE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with args.model.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if len(data) != MODEL_BYTES or zlib.crc32(data) != MODEL_CRC:
            raise ValueError("Expected the pinned d1-2B Q4_K_M GGUF")
        with memoryview(data) as model:
            split = END + (MODEL_BYTES - END) // 8192 * 4096
            constants = {
                "D1_EMBED_ROW_FIRST": FIRST, "D1_EMBED_ROW_END": END,
                "D1_EMBED_ROW_COUNT": ROWS, "D1_EMBED_ROW_BYTES": STRIDE,
                "D1_EMBED_PREFIX_CRC": zlib.crc32(model[:FIRST]),
                "D1_EMBED_TAIL_LEFT_CRC": zlib.crc32(model[END:split]),
                "D1_EMBED_TAIL_RIGHT_CRC": zlib.crc32(model[split:]),
            }
            with (args.out / "rows.bin").open("wb") as table:
                for offset in range(FIRST, END, STRIDE):
                    table.write(struct.pack("<I", zlib.crc32(model[offset:offset + STRIDE])))
            if args.fixture:
                with (args.out / "row-fixture.bin").open("wb") as fixture:
                    for token in (41, 124894, 127999):
                        offset = FIRST + token * STRIDE
                        fixture.write(model[offset:offset + STRIDE])
    (args.out / "rows.inc").write_text("".join(f".equ {name},{value:#x}\n" for name, value in constants.items()))


if __name__ == "__main__":
    main()
