import heapq

# === Hardware Profiles (example) ===
RESOURCES = ["CPU", "GPU", "DSP"]

# === Batch Buffer ===
TIER1_BUFFER = []

# === Build EET Table (from QoE enriched tasks) ===
def build_eet(task):
    """
    Build Estimated Execution Time table for this task
    on different resources (CPU, GPU, DSP).
    Uses task['local_time'] from QoE as baseline.
    """
    base = task.get("local_time", 50)  # fallback if missing
    return {
        "CPU": base * 1.0,
        "GPU": base * 0.5,
        "DSP": base * 0.8
    }

# === Dif-Min (Inter-core Scheduler) ===
def inter_core_schedule(tasks):
    """
    Schedule a batch of tasks across heterogeneous cores.
    Returns allocations: {(timestamp, task): (resource, completion_time)}.
    """
    U = tasks[:]   # copy of unmapped tasks
    ET_table = {res: 0 for res in RESOURCES}
    allocations = {}

    while U:
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

        allocations[(ti["timestamp"], ti["task"])] = (best_res, best_ct)
        ET_table[best_res] += eet[best_res]
        U.remove(ti)

    return allocations

def dif_heft_schedule(tasks):
    ready_time = {res: 0 for res in RESOURCES}
    finish_times = {}
    allocations = {}
    avg_cost = {(t["timestamp"], t["task"]): sum(build_eet(t).values()) / len(RESOURCES) for t in tasks}
    sorted_tasks = sorted(tasks, key=lambda t: avg_cost[(t["timestamp"], t["task"])], reverse=True)

    for t in sorted_tasks:
        tid = (t["timestamp"], t["task"])
        deps = t.get("deps", [])
        pred_finish = max(finish_times.get(d, 0) for d in deps) if deps else 0
        eet_table = build_eet(t)

        best_res, best_finish = None, float("inf")
        for res in RESOURCES:
            start_time = max(ready_time[res], pred_finish)
            finish_time = start_time + eet_table[res]
            if finish_time < best_finish:
                best_finish = finish_time
                best_res = res

        allocations[tid] = (best_res, best_finish)
        ready_time[best_res] = best_finish
        finish_times[tid] = best_finish

    return allocations

def add_to_tier1(task, batch_size=5, use_heft=False):
    TIER1_BUFFER.append(task)
    if len(TIER1_BUFFER) >= batch_size:
        if use_heft:
            allocations = dif_heft_schedule(TIER1_BUFFER)
        else:
            allocations = inter_core_schedule(TIER1_BUFFER)
        TIER1_BUFFER.clear()
        return allocations
    return None