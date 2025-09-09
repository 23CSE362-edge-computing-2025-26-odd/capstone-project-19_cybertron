import json
import heapq
from collections import defaultdict

# === Baseline Deadlines (default) ===
DEADLINE_TABLE = {
    "slam": 10,                 # ms baseline for SLAM
    "voice_recognition": 150    # ms baseline for speech
}

# === Local execution (Eq.5 & Eq.6) ===.
def local_exec(w, fL, alpha=0.5, beta=0.2):
    """Return local execution time (ms) and energy (J)."""
    tL = w / fL
    eL = alpha + beta * (w / fL)
    return tL, eL

# === Offloaded execution (Eq.7–Eq.9 simplified) ===
def offload_exec(w, f_off, Din, Dout, r_in, r_out, Pin=0.3, Pout=0.2):
    """Return offloaded execution time (ms) and energy (J)."""
    t_tx = Din / r_in + Dout / r_out   # transmission delay
    t_exec = w / f_off                 # execution delay
    t_total = t_tx + t_exec

    e_tx = (Din / r_in) * Pin + (Dout / r_out) * Pout
    return t_total, e_tx

# === QoE-aware Task Enrichment ===
def enrich_task(ci_json, task_type):
    deadline = DEADLINE_TABLE.get(task_type, 100)
    qoe_class = "QoE-Insensitive"

    # ----- SLAM -----
    if task_type == "slam":
        qoe_class = "QoE-Time"
        deadline = 10

        if ci_json.get("obstacle", False):
            deadline = 5
            qoe_class = "QoE-Safety"

        if not ci_json.get("vo_ok", True):
            deadline = max(deadline, 20)
            qoe_class = "QoE-Robustness"

        if any(det.get("near", False) for det in ci_json.get("detections", [])):
            deadline = 5
            qoe_class = "QoE-Safety"

        if ci_json.get("detections") and all(det.get("confidence", 1) < 0.4
                                             for det in ci_json["detections"]):
            deadline = max(deadline, 50)
            qoe_class = "QoE-Insensitive"

    # ----- Voice Recognition -----
    if task_type == "voice_recognition":
        voice_text = ci_json.get("voice_text", "").lower()
        if "obstacle" in voice_text:
            deadline = 30
            qoe_class = "QoE-Safety"
        elif any(cmd in voice_text for cmd in ["start", "stop", "turn left", "turn right"]):
            deadline = 60
            qoe_class = "QoE-Time"
        else:
            deadline = 200
            qoe_class = "QoE-Energy"

    return {
        "task": task_type,
        "timestamp": ci_json["ts"],
        "deadline_ms": deadline,
        "QoE_class": qoe_class,
        "data": ci_json
    }

# === Tier-1 vs Tier-2 Decision ===
def decide_tier(task):
    """Decide execution tier (Tier-1 local vs Tier-2 edge)."""
    deadline = task["deadline_ms"]

    # Safety/Time-Critical → Local only
    if task["QoE_class"] in ["QoE-Safety", "QoE-Time", "QoE-Robustness"]:
        return "Tier-1"

    # If offloading is feasible and energy-saving → Edge
    if task["offload_time"] <= deadline and task["offload_energy"] < task["local_energy"]:
        return "Tier-2"

    return "Tier-1"

# === Tier-2 Scheduler (preserve order, fit within max_time) ===
def schedule_tier2(tasks_all, max_time=100):
    tier2 = [t for t in tasks_all if t["assigned_tier"] == "Tier-2"]

    executed, fallback = [], []
    used_time = 0

    for t in tier2:
        if used_time + t["offload_time"] <= max_time:
            executed.append(t)
            used_time += t["offload_time"]
        else:
            t["assigned_tier"] = "Tier-1"  # fallback if over budget
            fallback.append(t)

    return executed, fallback

# === Main Runtime Simulation ===
def simulate_runtime(slam_file, voice_file, fL=50, f_off=200):
    task_queue = []
    metrics = defaultdict(lambda: {"count": 0, "total_deadline": 0})
    tasks_all = []

    # Load logs
    slam_events = [json.loads(line) for line in open(slam_file)]
    voice_events = [json.loads(line) for line in open(voice_file)]

    events = [(e["ts"], "slam", e) for e in slam_events] + \
             [(e["ts"], "voice_recognition", e) for e in voice_events]
    events.sort(key=lambda x: x[0])

    for ts, ttype, e in events:
        task = enrich_task(e, ttype)

        # Workload & comms
        w = e.get("workload", 100)
        Din, Dout = e.get("Din", 50), e.get("Dout", 20)
        r_in, r_out = 100, 100

        # Local vs Offload
        tL, eL = local_exec(w, fL)
        tO, eO = offload_exec(w, f_off, Din, Dout, r_in, r_out)

        task["local_time"] = tL
        task["local_energy"] = eL
        task["offload_time"] = tO
        task["offload_energy"] = eO
        task["assigned_tier"] = decide_tier(task)

        heapq.heappush(task_queue, (task["deadline_ms"], task["timestamp"], task))
        tasks_all.append(task)

        # Metrics
        metrics[task["QoE_class"]]["count"] += 1
        metrics[task["QoE_class"]]["total_deadline"] += task["deadline_ms"]

    return task_queue, metrics, tasks_all

# === Demo Runner ===
if __name__ == "__main__":
    tasks, metrics, tasks_all = simulate_runtime("slam_output.jsonl", "voice_output.jsonl")

    print("\n=== Final Task Queue (EDF Order) ===")
    while tasks:
        _, _, task = heapq.heappop(tasks)
        print(f"[{task['QoE_class']}] {task['task']} "
              f"| deadline={task['deadline_ms']}ms "
              f"| local=({task['local_time']:.1f}ms,{task['local_energy']:.2f}J) "
              f"| offload=({task['offload_time']:.1f}ms,{task['offload_energy']:.2f}J) "
              f"| assigned={task['assigned_tier']} "
              f"| ts={task['timestamp']}")

    print("\n=== QoE Metrics Summary ===")
    for qoe_class, data in metrics.items():
        count = data["count"]
        avg_deadline = data["total_deadline"] / count if count else 0
        print(f"{qoe_class}: {count} tasks | avg deadline={avg_deadline:.2f} ms")

    print("\n=== Tier Allocation Summary ===")
    tier1 = [t for t in tasks_all if t["assigned_tier"] == "Tier-1"]
    tier2 = [t for t in tasks_all if t["assigned_tier"] == "Tier-2"]
    print(f"Tier-1: {len(tier1)} tasks → {[t['task'] for t in tier1]}")
    print(f"Tier-2: {len(tier2)} tasks → {[t['task'] for t in tier2]}")

    print("\n=== Tier-2 Scheduling (Preserved Order) ===")
    executed, fallback = schedule_tier2(tasks_all, max_time=100)
    print("Executed on Tier-2 (original order, within max_time):")
    for t in executed:
        print(f"  {t['task']} | ts={t['timestamp']} "
              f"| time={t['offload_time']:.1f}ms")

    if fallback:
        print("\nTasks pushed back to Tier-1 (Tier-2 overloaded):")
        for t in fallback:
            print(f"  {t['task']} | ts={t['timestamp']} "
                  f"| offload_time={t['offload_time']:.1f}ms")
