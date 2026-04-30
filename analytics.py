"""
Matplotlib analytics — population curves, class distribution, policy fitness.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def plot_metrics(metrics: List[dict], save_path: str = "analytics.png") -> None:
    """
    Generate a multi-panel analytics dashboard from recorded metrics.

    Parameters
    ----------
    metrics : list of snapshot dicts from World._snapshot()
    save_path : output image path
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Analytics] matplotlib not installed — skipping plots.")
        return

    if not metrics:
        print("[Analytics] No metrics to plot.")
        return

    steps = [m["step"] for m in metrics]
    alive = [m["alive"] for m in metrics]
    avg_food = [m["avg_food"] for m in metrics]
    avg_health = [m["avg_health"] for m in metrics]

    classes = ["FARMER", "WARRIOR", "TRADER", "LEADER"]
    class_series = {c: [m["class_counts"].get(c, 0) for m in metrics] for c in classes}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Multi-Agent RL — Simulation Analytics", fontsize=15, fontweight="bold")
    fig.patch.set_facecolor("#1a1a2e")

    colors = {
        "FARMER": "#FFD832",
        "WARRIOR": "#DC3232",
        "TRADER": "#32B4DC",
        "LEADER": "#C878FF",
    }

    for ax in axes.flat:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaa")
        ax.xaxis.label.set_color("#ccc")
        ax.yaxis.label.set_color("#ccc")
        ax.title.set_color("#eee")
        for spine in ax.spines.values():
            spine.set_color("#333")

    # ── Panel 1: Total Population ─────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.plot(steps, alive, color="#4ecca3", linewidth=1.5)
    ax1.fill_between(steps, alive, alpha=0.15, color="#4ecca3")
    ax1.set_title("Total Population")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Agents Alive")

    # ── Panel 2: Class Distribution ───────────────────────────────────
    ax2 = axes[0, 1]
    for cls_name in classes:
        ax2.plot(steps, class_series[cls_name], label=cls_name,
                 color=colors[cls_name], linewidth=1.2)
    ax2.set_title("Class Distribution")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444", labelcolor="#ddd")

    # ── Panel 3: Average Food ─────────────────────────────────────────
    ax3 = axes[1, 0]
    ax3.plot(steps, avg_food, color="#e8a838", linewidth=1.2)
    ax3.fill_between(steps, avg_food, alpha=0.1, color="#e8a838")
    ax3.set_title("Average Food per Agent")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Food")

    # ── Panel 4: Average Health ───────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.plot(steps, avg_health, color="#e84040", linewidth=1.2)
    ax4.fill_between(steps, avg_health, alpha=0.1, color="#e84040")
    ax4.set_title("Average Health per Agent")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Health")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Analytics] Dashboard saved to {save_path}")


def plot_policy_fitness(
    policy_stats_history: List[List[dict]],
    save_path: str = "policy_fitness.png",
) -> None:
    """
    Plot policy fitness over time.

    Parameters
    ----------
    policy_stats_history : list of PolicyPool.get_stats() snapshots per step
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Analytics] matplotlib not installed.")
        return

    if not policy_stats_history:
        return

    # Collect unique policy IDs
    all_ids = sorted({s["id"] for snap in policy_stats_history for s in snap})

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="#aaa")
    ax.set_title("Policy Fitness Over Time", color="#eee")
    ax.set_xlabel("Step", color="#ccc")
    ax.set_ylabel("Fitness", color="#ccc")

    cmap = plt.cm.viridis
    for i, pid in enumerate(all_ids):
        fitness_vals = []
        for snap in policy_stats_history:
            val = next((s["fitness"] for s in snap if s["id"] == pid), 0.0)
            fitness_vals.append(val)
        color = cmap(i / max(len(all_ids) - 1, 1))
        ax.plot(fitness_vals, label=f"Policy {pid}", color=color, alpha=0.8, linewidth=1)

    ax.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#444", labelcolor="#ddd",
              ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Analytics] Policy fitness saved to {save_path}")
