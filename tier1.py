import json
import heapq
from collections import defaultdict
import tier1  # ADDED

# === Hardware Profiles (example) ===
RESOURCES = ["CPU", "GPU", "DSP"]

BASE_EXECUTION_TIME = {
    "slam": 50,
    "object_detection": 40,
    "voice_recognition": 30
}

# === Baseline Deadlines (default) ===
DEADLINE_TABLE = {
    "slam": 10,                 # ms baseline for SLAM
    "voice_recognition": 150    # ms baseline for speech
}

# === Local execution (Eq.5 & Eq.6) ===
def local_exec(w, fL, alpha=0.5, beta=0.2):
    """Return local execution time (ms) and energy (J)."""
    tL = w / fL
    eL = alpha + beta * (w / fL)
    return tL, eL

# === Offloaded execution (Eq.7–Eq.9 simplified) ===
def offload_exec(w, f_off, Din, Dout, r_in, r_out, Pin=0.3, Pout=0.2):
    """Return offloaded execution time (ms) and energy (J)."""
    # Transmission time
    t_tx = Din / r_in + Dout / r_out
    t_exec = w / f_off
    t_total = t_tx + t_exec

    # Transmission energy
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

        if ci_json.get("detections") and all(det.get("confidence", 1) < 0.4 for det in ci_json["detections"]):
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

# === Build EET Table ===
def build_eet(task):
    base = BASE_EXECUTION_TIME.get(task["task"], 50)
    return {
        "CPU": base * 1.0,
        "GPU": base * 0.5,
        "DSP": base * 0.8
    }

# === Dif-Min (Inter-core Scheduler) ===
def inter_core_schedule(tasks):
    U = tasks[:]          # unmapped tasks
    ET_table = {res: 0 for res in RESOURCES}
    allocations = {}

    while U:
        # compute Div and Sub for each task
        diffs = []
        for t in U:
            eet = build_eet(t)
            best = min(eet.values())
            worst = max(eet.values())
            Div = worst / best
            Sub = worst - best
            diffs.append((Div, Sub, t))

        # pick task with min Div, break ties with max Sub
        diffs.sort(key=lambda x: (x[0], -x[1]))
        _, _, ti = diffs[0]

        # assign to resource with minimal ET(cj)+EET(ti,cj)
        best_res, best_ct = None, float("inf")
        eet = build_eet(ti)
        for res in RESOURCES:
            ct = ET_table[res] + eet[res]
            if ct < best_ct:
                best_ct = ct
                best_res = res

        allocations[ti["timestamp"], ti["task"]] = (best_res, best_ct)
        ET_table[best_res] += eet[best_res]
        U.remove(ti)

        print(f"Allocated {ti['task']} to {best_res} with completion time {best_ct}")  # ADDED

    return allocations

# === Dif-HEFT (Inner-core Scheduler, simplified) ===
def inner_core_schedule(dep_tasks):
    # Here we only show a stub – real version requires DAG parsing.
    # We sort dependent tasks by priority (deadline + Div).
    priorities = []
    for t in dep_tasks:
        eet = build_eet(t)
        Div = max(eet.values()) / min(eet.values())
        Pri = t["deadline_ms"] + Div
        priorities.append((Pri, t))

    priorities.sort(key=lambda x: -x[0])  # higher = earlier
    return [t for _, t in priorities]

# === Bridge Code ===
def dispatch_task(task):
    # If local, run heterogeneity scheduling
    if task.get("execution_place", "local") == "local":
        print(f"Task {task['task']} dispatched locally")  # ADDED
        return "local"
    else:
        print(f"Task {task['task']} dispatched remotely")  # ADDED
        return "remote"

def execute_pipeline(task_queue):
    print(f"Task queue: {task_queue}")  # ADDED
    local_tasks, remote_tasks = [], []

    while task_queue:
        _, task = heapq.heappop(task_queue)
        if dispatch_task(task) == "local":
            local_tasks.append(task)
        else:
            remote_tasks.append(task)

    # Inter-core scheduling on local tasks
    allocations = inter_core_schedule(local_tasks)
    print("\n=== Local Task Allocations (Dif-Min) ===")
    for key, (res, ct) in allocations.items():
        ts, ttype = key
        print(f"Task {ttype} at ts={ts} → {res}, completion={ct:.2f}ms")

    # (Remote tasks forwarded to edge)
    print("\n=== Remote Tasks (to edge) ===")
    for t in remote_tasks:
        print(f"{t['task']} at ts={t['timestamp']} → REMOTE edge node")

# === Main Runtime Simulation ===
def simulate_runtime(slam_file, voice_file, fL=50, f_off=200):
    task_queue = []
    metrics = defaultdict(lambda: {"count": 0, "total_deadline": 0})

    # Load logs
    slam_events = [json.loads(line) for line in open(slam_file)]
    voice_events = [json.loads(line) for line in open(voice_file)]

    events = [(e["ts"], "slam", e) for e in slam_events] + \
             [(e["ts"], "voice_recognition", e) for e in voice_events]
    events.sort(key=lambda x: x[0])

    # Process events
    for ts, ttype, e in events:
        task = enrich_task(e, ttype)

        # Estimate workload & data sizes
        w = e.get("workload", 100)          # instructions (arbitrary unit)
        Din = e.get("Din", 20)
        Dout = e.get("Dout", 10)
        r_in = 10   # bandwidth (Mb/s)
        r_out = 10  # bandwidth (Mb/s)

        # Local execution
        tL, eL = local_exec(w, fL)

        # Offloaded execution
        tO, eO = offload_exec(w, f_off, Din, Dout, r_in, r_out)

        task["local_time"] = tL
        task["local_energy"] = eL
        task["offload_time"] = tO
        task["offload_energy"] = eO

        # Queue task (earliest deadline first)
        heapq.heappush(task_queue, (task["deadline_ms"], id(task), task))

        # Update metrics
        metrics[task["QoE_class"]]["count"] += 1
        metrics[task["QoE_class"]]["total_deadline"] += task["deadline_ms"]

    return task_queue, metrics

# === Demo Runner ===
if __name__ == "__main__":
    tasks, metrics = simulate_runtime("slam_output.jsonl", "voice_output.jsonl")

    print("\n=== Final Task Queue (EDF Order) ===")
    while tasks:
        _, _, task = heapq.heappop(tasks)
        print(task)

    print("\n=== QoE Metrics Summary ===")
    for qoe_class, data in metrics.items():
        print(f"{qoe_class}: {data}")

    tier1.execute_pipeline(tasks)  # ADDED