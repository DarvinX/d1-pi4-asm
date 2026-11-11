#!/usr/bin/env python3
"""Render one d1 image question and its label metadata."""

import argparse
from itertools import chain
import json
from pathlib import Path
from string import ascii_letters, ascii_uppercase
import struct
import sys

from jinja2 import Environment, StrictUndefined

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from prepare import Reader


MAX_OPTIONS = 255
PINNED_TEMPERATURE = 1.3373984098434448


def choice_options(criteria, vocab):
    if not isinstance(criteria, dict) or not 1 <= len(criteria) <= MAX_OPTIONS:
        raise ValueError(f"Choice needs 1 to {MAX_OPTIONS} named options")
    double_letters = [a + b for a in ascii_uppercase for b in ascii_uppercase]
    letters = [*ascii_uppercase, *double_letters]
    codes = list(criteria) if all(len(key) == 1 and key in ascii_letters for key in criteria) else letters
    fallback = [*ascii_uppercase, *(f"{i:02d}" for i in range(100)), *ascii_letters[:26],
                *(f"#{i}" for i in range(200)), *double_letters]
    used = set()
    for index, (key, description) in enumerate(criteria.items()):
        code = next((code for code in chain((codes[index],), fallback)
                     if code in vocab and vocab[code] not in used), None)
        if code is None:
            raise ValueError(f"No distinct single-token label for option {key!r}")
        used.add(vocab[code])
        yield dict(key=key, description=description, label=code), (code, "Ġ" + code)


def label_plan(decision):
    groups = list(decision["labels"].values())
    if decision["temperature"] != PINNED_TEMPERATURE:
        raise ValueError("Native label plans require the pinned d1-2B calibration")
    if not 1 <= len(groups) <= MAX_OPTIONS or any(not 1 <= len(group) <= 3 for group in groups):
        raise ValueError("Invalid number of label groups or token forms")
    tokens = list(chain.from_iterable(groups))
    fields = (len(groups), len(tokens), *(len(group) for group in groups), *tokens)
    return b"D1LABEL1" + struct.pack(f"<{len(fields)}I", *fields)


def render(model, scene, key):
    reader = Reader(model)
    question = scene["questions"][key]
    vocab = {token.decode(): i for i, token in enumerate(reader.meta["tokenizer.ggml.tokens"])}
    if question["type"] == "choice":
        if question.get("multiple", False):
            raise ValueError("This readout expects one winning choice")
        options, forms = zip(*choice_options(question["criteria"], vocab), strict=True)
    elif question["type"] == "noul":
        descriptions = question.get("criteria") or {}
        options = [dict(key=key, description=descriptions.get(key)) for key in ("true", "false")]
        forms = [["yes", "Yes", "YES"], ["no", "No", "NO"]]
    elif question["type"] == "score":
        criteria = question["criteria"]
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError("Score needs 2 to 10 ordered levels")
        options = [dict(key=str(index), description=description) for index, description in enumerate(criteria)]
        forms = [(option["key"],) for option in options]
    else:
        raise ValueError("Expected a choice, yes/no (noul), or ordinal score question")
    labels = {option["key"]: [vocab[form] for form in group if form in vocab]
              for option, group in zip(options, forms, strict=True)}
    if not all(labels.values()):
        raise ValueError("A decision label does not have a single-token form")
    template = reader.meta["tokenizer.chat_template.systemone"].decode()
    prompt = Environment(undefined=StrictUndefined).from_string(template).render(
        state=scene.get("state", "Answer the questions about this photograph."), images=["<__media__>"],
        instructions=question["instructions"], type=question["type"], options=options,
    )
    temperature = reader.meta[f"lfm3.decision.temperature.{question['type']}"]
    return prompt, dict(labels=labels, temperature=temperature)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("scene")
    parser.add_argument("question")
    parser.add_argument("out", type=Path)
    parser.add_argument("--label-plan", action="store_true", help="write a .labels file for the native scorer")
    args = parser.parse_args()
    scene = next(scene for scene in json.loads(args.suite.read_text()) if scene["id"] == args.scene)
    prompt, labels = render(args.model, scene, args.question)
    args.out.write_text(prompt)
    args.out.with_suffix(".json").write_text(json.dumps(labels, indent=2) + "\n")
    if args.label_plan:
        args.out.with_suffix(".labels").write_bytes(label_plan(labels))


if __name__ == "__main__":
    main()
