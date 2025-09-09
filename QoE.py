import json
import heapq
from collections import defaultdict

# === Baseline Deadlines ===
DEADLINE_TABLE = {
    "slam": 10,
    "voice_recognition": 150
}

# === Local execution model ===
def local_exec(w, fL=50, alpha=0.5, beta=0.2):
    """Return local execution time (ms) and energy (J)."""
    tL = w / fL
    eL = alpha + beta * (w / fL)
    return tL, eL

# === Offloaded execution model ===
def offload_exec(w, f_off=200, Din=20, Dout=10, r_in=10, r_out=10,
                 Pin=0.3, Pout=0.2):
    """Return offloaded execution time (ms) and energy (J)."""
    t_tx = Din / r_in + Dout / r_out   # transmission delay
    t_exec = w / f_off                 # compute delay
    t_total = t_tx + t_exec

    e_tx = (Din / r_in) * Pin + (Dout / r_out) * Pout
    return t_total, e_tx

# === Compute Deadline + QoE Class ===
def enrich_task(ci_json, task_type, prev_task_ts=None, last_voice=None):
    base = DEADLINE_TABLE.get(task_type, 100)
    deadline = base
    qoe_class = "QoE-Insensitive"
    score = 0

    # ----- SLAM (fused with camera) -----
    if task_type == "slam":
        if ci_json.get("obstacle", False):
            score += 50
        if any(det.get("near", False) for det in ci_json.get("detections", [])):
            score += 30
        if not ci_json.get("vo_ok", True):
            score -= 10
        max_conf = max([d.get("confidence", 0) for d in ci_json.get("detections", [])], default=0)
        score += int(max_conf * 20)
        if prev_task_ts and ci_json["ts"] - prev_task_ts < 0.05:
            score += 10
        if last_voice and "obstacle" in last_voice:
            score += 20

        deadline = max(5, base - score)

        if score >= 60:
            qoe_class = "QoE-Safety"
        elif score >= 30:
            qoe_class = "QoE-Time"
        elif not ci_json.get("vo_ok", True):
            qoe_class = "QoE-Robustness"
        elif score <= 0:
            qoe_class = "QoE-Insensitive"
        else:
            qoe_class = "QoE-Time"

    # ----- Voice Recognition -----
    elif task_type == "voice_recognition":
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

    # --- Execution cost models ---
    w = ci_json.get("workload", 100)
    Din = ci_json.get("Din", 20)
    Dout = ci_json.get("Dout", 10)

    local_time, local_energy = local_exec(w)
    offload_time, offload_energy = offload_exec(w, Din=Din, Dout=Dout)

    return {
        "task": task_type,
        "timestamp": ci_json["ts"],
        "deadline_ms": deadline,
        "QoE_class": qoe_class,
        "local_time": local_time,
        "local_energy": local_energy,
        "offload_time": offload_time,
        "offload_energy": offload_energy,
        "data": ci_json
    }

# === Unified Runtime Simulation ===
def simulate_runtime(slam_file, voice_file):
    task_queue = []
    metrics = defaultdict(lambda: {"count": 0, "total_deadline": 0})

    slam_events   = [json.loads(line) for line in open(slam_file)]
    voice_events  = [json.loads(line) for line in open(voice_file)]

    events = [(e["ts"], "slam", e) for e in slam_events] + \
             [(e["ts"], "voice_recognition", e) for e in voice_events]
    events.sort(key=lambda x: x[0])

    prev_task_ts = None
    last_voice = None

    for ts, ttype, e in events:
        if ttype == "voice_recognition":
            last_voice = e.get("voice_text", "").lower()

        task = enrich_task(e, ttype, prev_task_ts, last_voice)
        prev_task_ts = ts

        heapq.heappush(task_queue, (task["deadline_ms"], task))

        metrics[task["QoE_class"]]["count"] += 1
        metrics[task["QoE_class"]]["total_deadline"] += task["deadline_ms"]

    return task_queue, metrics

# === Main Run ===
if __name__ == "__main__":
    tasks, metrics = simulate_runtime("slam_output.jsonl", "voice_output.jsonl")

    print("\n=== Final Task Queue (min-heap order by deadline) ===")
    for deadline, task in tasks:
        print(f"[{task['QoE_class']}] {task['task']} | deadline={deadline}ms "
              f"| local=({task['local_time']:.1f}ms,{task['local_energy']:.2f}J) "
              f"| offload=({task['offload_time']:.1f}ms,{task['offload_energy']:.2f}J) "
              f"| ts={task['timestamp']}")

    print("\n=== QoE Metrics Summary ===")
    for qoe_class, data in metrics.items():
        count = data["count"]
        avg_deadline = data["total_deadline"] / count if count else 0
        print(f"{qoe_class}: {count} tasks | avg deadline={avg_deadline:.2f} ms")
