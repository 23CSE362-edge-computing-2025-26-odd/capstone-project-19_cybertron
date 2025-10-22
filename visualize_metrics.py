# visualize_metrics.py
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def ensure_plot_dir():
    """Ensure /plots folder exists."""
    os.makedirs("plots", exist_ok=True)


def plot_latency_distribution(results):
    """Plot histogram of task completion times (latency)."""
    latencies = [finish for finish, _, _ in results]
    plt.figure(figsize=(7, 4))
    plt.hist(latencies, bins=10, edgecolor="black", alpha=0.7)
    plt.title("Task Latency Distribution")
    plt.xlabel("Completion Time (ms)")
    plt.ylabel("Number of Tasks")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("plots/latency_distribution.png")
    plt.close()


def plot_qoe_classes(tasks):
    """Pie chart of QoE classes across all tasks."""
    qoe_classes = [t.get("QoE_class", "Unknown") for t in tasks]
    counts = Counter(qoe_classes)
    labels, values = zip(*counts.items())

    plt.figure(figsize=(5, 5))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("QoE Class Distribution")
    plt.tight_layout()
    plt.savefig("plots/qoe_classes.png")
    plt.close()


def plot_tier_distribution(results):
    """Bar chart of Tier-1 vs Tier-2 executions."""
    tiers = [tier for _, _, tier in results]
    counts = Counter(tiers)
    labels, values = zip(*counts.items())

    plt.figure(figsize=(5, 4))
    plt.bar(labels, values, color=["#4e79a7", "#f28e2b"])
    plt.title("Tier Distribution")
    plt.xlabel("Tier")
    plt.ylabel("Tasks Executed")
    plt.tight_layout()
    plt.savefig("plots/tier_distribution.png")
    plt.close()


def plot_energy_vs_time(tasks):
    """Plot energy vs completion time scatter, handling missing keys."""
    plt.figure(figsize=(7, 4))

    local_times = [t.get("local_time", t.get("completion_ms", 0)) for t in tasks]
    local_energy = [t.get("local_energy", 0) for t in tasks]
    tiers = [t.get("tier", "Unknown") for t in tasks]

    plt.scatter(local_times, local_energy, c=[
        "red" if tier == "Tier-1" else "blue" for tier in tiers
    ])
    plt.title("Energy vs Completion Time")
    plt.xlabel("Completion Time (ms)")
    plt.ylabel("Energy (mJ)")
    plt.grid(True)

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/energy_vs_time.png", bbox_inches="tight")
    plt.close()



def visualize_all(tasks, results):
    """Generate all visualizations from runtime data."""
    print("\n📊 Generating visualization plots...")
    ensure_plot_dir()
    plot_latency_distribution(results)
    plot_qoe_classes(tasks)
    plot_tier_distribution(results)
    plot_energy_vs_time(tasks)
    print("✅ All plots saved in /plots folder.")
