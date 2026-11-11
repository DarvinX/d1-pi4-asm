#!/usr/bin/env python3
"""Freeze a balanced CV-Bench subset with original choices or direct counts."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageOps
from pyarrow.parquet import ParquetFile
from render_prompt import label_plan, render



REVISION = "bc284db50d036958861cb60cdd7b77612052ce0d"
SEED = "20261110-cv-bench"
SOURCES = {
    "Relation": ("ADE20K Validation Set", "COCO 2017 Validation Set"),
    "Depth": ("Omni3D_Hypersim", "Omni3D_SUNRGBD", "Omni3D_nuScenes"),
    "Distance": ("Omni3D_Hypersim", "Omni3D_SUNRGBD", "Omni3D_nuScenes"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def order(row, purpose, seed):
    return digest(f"{seed}/{purpose}/{row['filename']}".encode())


def select(rows, seed, excluded):
    eligible = sorted((row for row in rows if len(row["choices"]) == 2),
                      key=lambda row: order(row, "sample", seed))
    assert len(eligible) == 1850 and all(row["task"] in SOURCES for row in eligible)
    used = set(excluded)

    def take(task, source=None, label=None):
        for row in eligible:
            identity = row["source_dataset"], row["source_filename"]
            if (row["task"] == task and identity not in used
                    and (source is None or row["source_dataset"] == source)
                    and (label is None or row["answer"] == f"({label})")):
                used.add(identity)
                return row
        raise ValueError(f"Cannot fill CV-Bench stratum {task}/{source}/{label}")

    selected = []
    for block in range(1, 4):
        cases = []
        for task, sources in SOURCES.items():
            for index, source in enumerate(sources):
                if len(sources) == 3 and index == block - 1:
                    continue
                cases.extend(take(task, source, label) for label in "AB")
        assert len(cases) == 12
        selected.extend(dict(row, block=block) for row in sorted(cases, key=lambda row: order(row, "run", seed)))
    selected.extend(dict(take(task), block=0) for task in SOURCES)
    return selected


def select_counts(rows, existing, seed, excluded):
    used = set(excluded) | {(row["source_dataset"], row["source_filename"]) for row in existing}
    pool = sorted((row for row in rows if row["task"] == "Count"),
                  key=lambda row: order(row, "count-sample", seed))

    def take(arity, source, label, block):
        for row in pool:
            identity = row["source_dataset"], row["source_filename"]
            if (len(row["choices"]) == arity and row["source"] == source
                    and row["answer"] == f"({label})" and identity not in used):
                used.add(identity)
                return dict(row, block=block)
        raise ValueError(f"Cannot fill count stratum {arity}/{source}/{label}")

    result = []
    for arity, labels in ((4, "ADBC"), (5, "AEBD"), (6, "AFBE")):
        for index, label in enumerate(labels):
            source = "ADE20K" if index < 2 else "COCO"
            result.append(take(arity, source, label, (index + arity - 4) % 3 + 1))
        result.append(take(arity, "COCO", labels[1], 0))
    return result


def count_value(row):
    return int(row["choices"][ord(row["answer"][1]) - ord("A")])


def select_numeric(rows, seed, excluded):
    pool = sorted((row for row in rows if row["task"] == "Count"),
                  key=lambda row: (count_value(row) != 0, order(row, "numeric-sample", seed)))
    used = set(excluded)
    selected = []
    for low, high in ((0, 1), (2, 3), (4, 5), (6, 9)):
        for source in ("ADE20K", "COCO"):
            taken = 0
            for row in pool:
                identity = row["source_dataset"], row["source_filename"]
                if row["source"] != source or identity in used or not low <= count_value(row) <= high:
                    continue
                used.add(identity)
                selected.append(dict(row, task="Numeric count", block=1))
                taken += 1
                if taken == 2:
                    break
            if taken != 2:
                raise ValueError(f"Cannot fill numeric count stratum {source}/{low}-{high}")
    return sorted(selected, key=lambda row: order(row, "numeric-run", seed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "model", "out"):
        parser.add_argument(arg, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--extended", action="store_true", help="add original four-to-six-choice counting questions")
    mode.add_argument("--numeric", action="store_true", help="select direct 0–9 counting questions only")
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--block", type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    for directory in ("images", "rgb", "prompts"):
        (args.out / directory).mkdir()
    if args.extended or args.numeric:
        (args.out / "labels").mkdir()
    files = sorted(args.source.glob("test_*.jsonl"))
    rows = [dict(json.loads(line), id=f"{path.stem[5:]}-{index:04d}")
            for path in files for index, line in enumerate(path.read_text().splitlines())]
    assert len(rows) == 2638 and len({row["filename"] for row in rows}) == len(rows)
    excluded = set()
    for path in args.exclude_manifest:
        prior = json.loads(path.read_text())
        excluded.update((row["source_dataset"], row["source_filename"]) for row in prior["cases"])
    cases = select_numeric(rows, args.seed, excluded) if args.numeric else select(rows, args.seed, excluded)
    if args.extended:
        cases.extend(select_counts(rows, cases, args.seed, excluded))
        cases.sort(key=lambda row: (row["block"], order(row, "mixed-run", args.seed)))
    if args.block:
        cases = [row for row in cases if row["block"] == args.block]
    by_filename = {row["filename"]: row for row in cases}
    for source in sorted(args.source.glob("test_*.parquet")):
        columns = ("filename", "image", "choices", "answer", "question")
        for batch in ParquetFile(source).iter_batches(batch_size=32, columns=list(columns)):
            for item in batch.to_pylist():
                if item["filename"] not in by_filename:
                    continue
                row = by_filename[item["filename"]]
                assert all(item[key] == row[key] for key in ("choices", "answer", "question"))
                name = row["id"]
                original = item["image"]["bytes"]
                with Image.open(BytesIO(original)) as image:
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    canvas = ImageOps.pad(rgb, (640, 480), method=Image.Resampling.LANCZOS,
                                          color=(127, 127, 127))
                png, raw = args.out / "images" / f"{name}.png", canvas.tobytes()
                canvas.save(png)
                (args.out / "rgb" / f"{name}.rgb").write_bytes(raw)
                if args.numeric:
                    question = {"type": "score", "instructions": row["question"] +
                                " Give the exact object count as a single integer from 0 to 9.",
                                "criteria": [str(i) for i in range(10)]}
                    truth = str(count_value(row))
                else:
                    question = {"type": "choice", "instructions": row["question"],
                                "criteria": {chr(ord("A") + i): choice for i, choice in enumerate(row["choices"])}}
                    truth = row["answer"].strip("()")
                prompt, labels = render(args.model, {"questions": {"answer": question}}, "answer")
                if question["type"] == "choice" and len(row["choices"]) == 2:
                    assert labels["labels"] == {"A": [41, 334], "B": [42, 378]}
                assert prompt.count("<__media__>") == 1
                (args.out / "prompts" / f"{name}.txt").write_text(prompt)
                row.update(question_spec=question, truth=truth,
                           original_sha256=digest(original), png_sha256=digest(png.read_bytes()),
                           rgb_sha256=digest(raw), prompt_sha256=digest(prompt.encode()))
                if args.extended or args.numeric:
                    plan = label_plan(labels)
                    (args.out / "labels" / f"{name}.labels").write_bytes(plan)
                    row["label_plan_sha256"] = digest(plan)
    assert all("rgb_sha256" in row for row in cases)
    manifest = {"dataset": "nyu-visionx/CV-Bench", "revision": REVISION, "seed": args.seed,
                "pixels": "EXIF orientation, RGB, aspect-preserving Lanczos resize with gray padding to 640x480",
                "source_sha256": {path.name: digest(path.read_bytes()) for path in files}, "cases": cases}
    if args.numeric:
        manifest["answer_mode"] = "Direct integer 0–9 via the pinned d1 ordinal score head"
    excluded_digests = [digest(path.read_bytes()) for path in args.exclude_manifest]
    if len(excluded_digests) == 1:
        manifest["excluded_manifest_sha256"] = excluded_digests[0]
    elif excluded_digests:
        manifest["excluded_manifests_sha256"] = excluded_digests
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
