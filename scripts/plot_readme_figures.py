"""Generate the README figures from the audited MATH-500 pilot counts."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


def plot_results():
    labels = ["Base", "GeoRA (untrained)", "LoRA (25 steps)", "GeoRA (25 steps)"]
    correct = [270, 279, 269, 284]
    total = 500
    accuracy = [100 * count / total for count in correct]
    colors = ["#9caab3", "#768fa2", "#397a81", "#cb7844"]

    figure, axis = plt.subplots(figsize=(8, 3.5), layout="constrained")
    axis.barh(labels, accuracy, color=colors, height=0.58)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xticks([0, 25, 50, 75, 100], labels=["0", "25", "50", "75", "100%"])
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#e4e8eb", linewidth=0.8)
    axis.tick_params(axis="both", length=0, labelsize=10, pad=6)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for row, (value, count) in enumerate(zip(accuracy, correct)):
        axis.text(value + 1.2, row, f"{value:.1f}%  ({count}/{total})", va="center", fontsize=10)
    axis.set_title("MATH-500 accuracy", loc="left", fontsize=14, fontweight="bold", pad=14)
    axis.set_xlabel("Qwen2.5-1.5B-Instruct · GSM8K training · 25 steps · one seed", fontsize=9, labelpad=10)
    figure.savefig(FIGURES / "math500_pilot.png", dpi=180, facecolor="white")
    plt.close(figure)


def plot_training_flow():
    figure, axis = plt.subplots(figsize=(9, 2.1), layout="constrained")
    axis.set_xlim(-0.1, 10)
    axis.set_ylim(-0.35, 2)
    axis.axis("off")
    blocks = [
        (0.0, "GSM8K\nprompts", "#eef2f4"),
        (2.5, "vLLM\nrollouts", "#dfeef1"),
        (5.0, "Answer\nreward", "#f5eadf"),
        (7.5, "GRPO\nupdate", "#e2eee8"),
    ]
    for x, label, color in blocks:
        axis.add_patch(FancyBboxPatch((x, 0.62), 2, 0.9, boxstyle="round,pad=0.05,rounding_size=0.12", linewidth=0, facecolor=color))
        axis.text(x + 1, 1.07, label, ha="center", va="center", fontsize=11, color="#1f3038")
    for start in (2.05, 4.55, 7.05):
        axis.add_patch(FancyArrowPatch((start, 1.07), (start + 0.38, 1.07), arrowstyle="-|>", mutation_scale=12, color="#60747c", linewidth=1.3))
    axis.add_patch(FancyArrowPatch((8.5, 0.53), (3.5, 0.53), connectionstyle="arc3,rad=-0.17", arrowstyle="-|>", mutation_scale=12, color="#60747c", linewidth=1.3))
    axis.text(6, -0.10, "sync updated weights", ha="center", fontsize=9, color="#4d6068")
    figure.savefig(FIGURES / "training_flow.png", dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    plot_results()
    plot_training_flow()
