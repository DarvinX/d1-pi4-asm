#!/usr/bin/env python3
"""Prepare one image question, run the ARM64 engine, and decode its scores."""

import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory

from PIL import Image, ImageOps

from render_prompt import label_plan, render


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "projector", "image", "question"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    question = json.loads(args.question.read_text())
    prompt, labels = render(args.model, {"questions": {"answer": question}}, "answer")
    ordinal = question["type"] == "score"
    mode = "score" if ordinal else "choice"
    profile = json.loads((ROOT / "profiles" / f"{mode}.json").read_text())
    env = dict(os.environ, **profile, LFM_THREADS="4")
    with TemporaryDirectory(prefix="d1-decision-") as directory:
        root = Path(directory)
        pixels, text, plan = (root / name for name in ("image.rgb", "prompt.txt", "answer.labels"))
        with Image.open(args.image) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            image = ImageOps.pad(rgb, (640, 480), method=Image.Resampling.LANCZOS, color=(127, 127, 127))
            pixels.write_bytes(image.tobytes())
        text.write_text(prompt)
        plan.write_bytes(label_plan(labels))
        result = subprocess.run(
            [str(ROOT / "build/d1-vision-asm-sparse"), "--decision", str(args.model.resolve()),
             str(args.projector.resolve()), str(pixels), str(text), f"@{plan}"],
            cwd=ROOT, env=env, check=True, stdout=subprocess.PIPE,
        )
    scores = struct.unpack(f"<{len(labels['labels'])}d", result.stdout)
    probabilities = dict(zip(labels["labels"], scores, strict=True))
    if ordinal:
        cumulative = 0.0
        for index, probability in enumerate(scores):
            cumulative += probability
            if cumulative >= 0.5:
                answer = index
                break
        else:
            raise ValueError("No median in engine output")
    else:
        answer = max(probabilities, key=probabilities.__getitem__)
    print(json.dumps({"answer": answer, "probabilities": probabilities}, indent=2))


if __name__ == "__main__":
    main()
