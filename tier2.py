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
    executed = []
    fallback = []
    used_time = 0.0

    for t in tasks_all:
        off_t = float(t.get("offload_time", float("inf")))
        if used_time + off_t <= max_time:
            t["assigned_tier"] = "Tier-2"
            executed.append(t)
            used_time += off_t
        else:
            # fallback to Tier-1
            t["assigned_tier"] = "Tier-1"
            fallback.append(t)

            # 🔹 Immediately push to Tier-1 scheduler (HEFT or Dif-Min)
            fallback_allocations = tier1.add_to_tier1(t, use_heft=True)
            if fallback_allocations:
                # You can optionally handle/print the allocation results here
                print("Tier-1 allocation:", fallback_allocations)

    return executed, fallback
