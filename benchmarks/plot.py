#!/usr/bin/env python3
"""Render the published figures from the archived Raspberry Pi results."""

import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread


ROOT = Path(__file__).resolve().parent
BG, FG, MUTED = "#141917", "#F2F0E6", "#BBC5BA"
LIME, SAGE, LINE, GREEN = "#D3F677", "#7D968A", "#3A4740", "#A9C477"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "font.family": "DejaVu Sans"})


def report(name):
    return json.loads((ROOT / "reports" / f"{name}.json").read_text())


def label(fig, x, y, value, size=16, color=FG, bold=False, mono=False, align="left"):
    fig.text(x, y, value, fontsize=size, color=color, weight="bold" if bold else "normal",
             fontfamily="DejaVu Sans Mono" if mono else "DejaVu Sans", ha=align, va="center")


def rule(fig, y):
    fig.add_artist(plt.Line2D((0.07, 0.93), (y, y), color=LINE, lw=1, transform=fig.transFigure))


def bars(fig, box, values, limit, fmt):
    axis = fig.add_axes(box)
    axis.set_xlim(0, limit)
    axis.set_ylim(-0.55, 1.55)
    axis.axis("off")
    for y, value, color, ink in zip((1, 0), values, (LIME, SAGE), (FG, MUTED), strict=True):
        axis.barh(y, value, height=0.48, color=color, zorder=2)
        axis.text(value + limit * 0.021, y, fmt(value), color=ink, fontfamily="DejaVu Sans Mono",
                  fontsize=16, va="center", zorder=3)


def pair(row, suffix):
    return [row[f"assembly_{suffix}"], row[f"llama_{suffix}"]]


def main():
    choice_report, count_report = report("choices-independent"), report("counts")
    choice, count = (r["paired_pi_probe"] for r in (choice_report, count_report))
    original = report("choices-original")
    accuracy = original["correct"] + choice_report["correct"]
    accuracy_ref = original["reference_correct"] + choice_report["reference_correct"]
    modes = ("Multiple choice", "Direct counts")
    fig = plt.figure(figsize=(9, 9), dpi=120)
    board = fig.add_axes((0.72, 0.74, 0.23, 0.23))
    board.imshow(imread(ROOT / "board.png"))
    board.axis("off")
    label(fig, 0.07, 0.895, "Faster decisions on a", 29, bold=True)
    label(fig, 0.07, 0.825, "4 GB Raspberry Pi 4", 33, bold=True)
    label(fig, 0.07, 0.748, "48 of 300 image positions stay active", 17, LIME, True)
    label(fig, 0.07, 0.708, "Ranked by the question’s attention  ·  d1-2B in ARM64 assembly", 13, MUTED)
    label(fig, 0.07, 0.635, "■ Assembly", 13, LIME, True)
    label(fig, 0.29, 0.635, "■ llama.cpp", 13, SAGE, True)
    label(fig, 0.93, 0.635, "Median time per question", 13, MUTED, align="right")
    for y, title, row in zip((0.575, 0.379), ("Multiple choice · 48 active", "Direct counts · 300 active"),
                             (choice, count), strict=True):
        label(fig, 0.07, y, title, 20, bold=True)
        label(fig, 0.93, y, f"{row['median_paired_speedup']:.2f}×", 20, LIME, True, True, "right")
        rule(fig, y - 0.028)
        bars(fig, (0.07, y - 0.159, 0.86, 0.117), pair(row, "median_wall_s"),
             88, lambda x: f"{x:.2f} s")
    rule(fig, 0.176)
    label(fig, 0.07, 0.137, f"Choice accuracy  {accuracy}/64 vs {accuracy_ref}/64", 16, bold=True)
    label(fig, 0.07, 0.098, "All 16 count answers matched llama.cpp.", 15, bold=True)
    label(fig, 0.07, 0.058, "4 paired Pi runs per format  ·  Codex + autoresearch", 13, MUTED)
    fig.savefig(ROOT / "social-card.png")
    plt.close(fig)

    fig = plt.figure(figsize=(12, 6.75), dpi=120)
    label(fig, 0.07, 0.885, "Faster decisions on a 4 GB Raspberry Pi 4", 28, bold=True)
    label(fig, 0.07, 0.807, "■ Assembly", 14, LIME, True)
    label(fig, 0.23, 0.807, "■ llama.cpp", 14, SAGE, True)
    rule(fig, 0.754)
    label(fig, 0.07, 0.688, "Median latency", 20, bold=True)
    label(fig, 0.53, 0.688, "Peak process RSS", 20, bold=True)
    for y, title, row in zip((0.596, 0.351), modes, (choice, count), strict=True):
        for x in (0.07, 0.53):
            label(fig, x, y, title, 14, MUTED, True)
        bars(fig, (0.07, y - 0.165, 0.40, 0.146), pair(row, "median_wall_s"), 101, lambda x: f"{x:.2f} s")
        bars(fig, (0.53, y - 0.165, 0.40, 0.146), pair(row, "peak_rss_mib"), 2240,
             lambda x: f"{x:,.0f} MiB")
    rule(fig, 0.127)
    label(fig, 0.07, 0.077, "Four same-input pairs per format  ·  Process startup + one image encode per question",
          13, MUTED)
    fig.savefig(ROOT / "pi-comparison.png")
    plt.close(fig)

    fig = plt.figure(figsize=(12, 5.7), dpi=120)
    label(fig, 0.07, 0.88, "Where direct counts spend the extra time", 28, bold=True)
    label(fig, 0.07, 0.792, "Median ARM64 phase time per question", 16, MUTED)
    label(fig, 0.07, 0.688, "■ Image encode", 14, SAGE, True)
    label(fig, 0.30, 0.688, "■ Backbone and scoring", 14, LIME, True)
    rule(fig, 0.619)
    for y, title, caption, data in (
        (0.519, "Multiple choice", "48 active image positions", choice_report),
        (0.293, "Direct counts", "300 active image positions", count_report),
    ):
        label(fig, 0.07, y, title, 17, bold=True)
        label(fig, 0.07, y - 0.059, caption, 12, MUTED)
        axis = fig.add_axes((0.32, y - 0.094, 0.61, 0.14))
        axis.set_xlim(0, 47)
        axis.set_ylim(-1, 1)
        axis.axis("off")
        left = 0
        for phase, color in (("vision", SAGE), ("backbone_score", LIME)):
            val = median(row["phases_s"][phase] for row in data["measured"])
            axis.barh(0, val, left=left, color=color, height=0.84)
            axis.text(left + val / 2, 0, f"{val:.1f} s", ha="center", va="center",
                      color=BG, fontsize=15, weight="bold", fontfamily="DejaVu Sans Mono")
            left += val
    rule(fig, 0.146)
    label(fig, 0.07, 0.09, "16 Pi runs per format  ·  Phase medians exclude startup and need not sum to the wall median",
          13, MUTED)
    fig.savefig(ROOT / "backbone-phases.png")
    plt.close(fig)

    fig = plt.figure(figsize=(12, 5), dpi=120)
    label(fig, 0.07, 0.864, "How the question chooses 48 positions", 28, bold=True)
    label(fig, 0.07, 0.758, "Sum post-image text attention across heads in the first global layer.", 15, MUTED)
    rule(fig, 0.685)
    grid = fig.add_axes((0.07, 0.219, 0.54, 0.391))
    grid.set_xlim(-0.5, 29.5)
    grid.set_ylim(9.5, -0.5)
    grid.axis("off")
    ranks = range(300)
    grid.scatter([rank % 30 for rank in ranks], [rank // 30 for rank in ranks],
                 c=[LIME if rank < 48 else SAGE for rank in ranks], s=44, marker="s", linewidths=0)
    label(fig, 0.67, 0.571, "48 highest-ranked", 19, LIME, True)
    label(fig, 0.67, 0.497, "Continue updating", 14)
    label(fig, 0.67, 0.381, "252 remaining", 19, SAGE, True)
    label(fig, 0.67, 0.306, "Keep prior hidden states", 14)
    label(fig, 0.67, 0.239, "Still supply keys and values", 14)
    rule(fig, 0.166)
    label(fig, 0.07, 0.106, "Each square is an image position. The grid shows rank order.", 13, MUTED)
    fig.savefig(ROOT / "attention-ranking.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
