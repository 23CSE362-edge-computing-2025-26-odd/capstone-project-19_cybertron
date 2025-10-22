import json
import tier1
import random

# === Baseline Deadlines (default) ===
DEADLINE_TABLE = {
    "slam": 10,                  # ms baseline for SLAM
    "voice_recognition": 150     # ms baseline for speech
}

# === QoE-based Tier Decision ===
import random

def decide_tier(task):
    """
    Balanced tier decision: roughly half of tasks go to Tier-2
    depending on latency, energy, and a small random factor.
    """
    local_time = task.get("local_time", 5)
    offload_time = task.get("offload_time", 10)
    local_energy = task.get("local_energy", 5)
    offload_energy = task.get("offload_energy", 3)
    deadline = task.get("deadline_ms", 50)

    # compute benefits
    time_gain = local_time - offload_time
    energy_gain = local_energy - offload_energy

    # weighted score combining speed & energy efficiency
    score = (0.6 * time_gain) + (0.4 * energy_gain)

    # random threshold for load balancing
    randomness = random.uniform(-2, 2)

    # decision boundary
    if score + randomness > 0 and offload_time < deadline * 0.9:
        return "Tier-2"
    else:
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
