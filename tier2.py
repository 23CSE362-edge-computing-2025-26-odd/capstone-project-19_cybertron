import json

# === Baseline Deadlines (default) ===
DEADLINE_TABLE = {
    "slam": 10,                  # ms baseline for SLAM
    "voice_recognition": 150     # ms baseline for speech
}

# === QoE-based Tier Decision ===
def decide_tier(task):
    """
    Decide execution tier (Tier-1 local vs Tier-2 edge).
    """
    deadline = task["deadline_ms"]

    # Safety / Time-critical → must stay on Tier-1
    if task["QoE_class"] in ["QoE-Safety", "QoE-Time", "QoE-Robustness"]:
        return "Tier-1"

    # If offloading is feasible and within deadline → Tier-2
    if task["offload_time"] <= deadline and task["offload_energy"] < task["local_energy"]:
        return "Tier-2"

    # Otherwise → Tier-1
    return "Tier-1"


# === Tier-2 Scheduler (batch mode) ===
def schedule_tier2(tasks_all, max_time=200):
    """
    Schedule Tier-2 tasks in arrival order.
    - Keeps total offload time under max_time budget.
    - Pushes overflow tasks back to Tier-1 (fallback).
    """
    executed, fallback = [], []
    used_time = 0

    for t in tasks_all:
        if used_time + t["offload_time"] <= max_time:
            executed.append(t)
            used_time += t["offload_time"]
        else:
            # fallback → cannot fit in Tier-2
            t["assigned_tier"] = "Tier-1"
            fallback.append(t)

    return executed, fallback