# simulation.py
import simPy
import json
import heapq
import  QoE
import tier1
import tier2
from collections import defaultdict

# === Parameters ===
SIM_END = 500          # ms simulation horizon
MAX_TIER2_TIME = 100   # budget for Tier-2 (same as tier2.py)

# === SimPy resources for Tier-1 hardware ===
def create_resources(env):
    return {
        "CPU": simPy.Resource(env, capacity=1),
        "GPU": simPy.Resource(env, capacity=1),
        "DSP": simPy.Resource(env, capacity=1),
    }

# === Task Source ===
def task_source(env, yolo_vo_state, task_queue):
    # Load YOLO+VO JSON (not JSONL)
    slam_data = json.load(open(yolo_vo_state))
    slam_events = slam_data["records"]

    events = [(e["ts"], "slam", e) for e in slam_events]
    events.sort(key=lambda x: x[0])

    prev_task_ts = None
    last_voice = None  # unused since no voice

    for ts, ttype, e in events:
        task = QoE.enrich_task(e, ttype, prev_task_ts, last_voice)
        prev_task_ts = ts

        # assign tier (Tier-1 vs Tier-2)
        task["assigned_tier"] = tier2.decide_tier(task)

        # push into EDF heap
        heapq.heappush(task_queue, (task["deadline_ms"], task["timestamp"], task))

        yield env.timeout(5)  # inter-arrival gap

# === Scheduler with Tier-1 + Tier-2 integration ===
def scheduler(env, task_queue, results, resources, metrics):
    tier2_used_time = 0  # track Tier-2 budget

    while True:
        if task_queue:
            _, _, task = heapq.heappop(task_queue)
            start_time = env.now

            # --- Tier-1 local execution ---
            if task["assigned_tier"] == "Tier-1":
                allocations = tier1.inter_core_schedule([task])
                for (ts, ttype), (res, ct) in allocations.items():
                    with resources[res].request() as req:
                        yield req
                        yield env.timeout(ct)
                        finish_time = env.now
                        results.append((finish_time, f"LOCAL-{res}", task))

                        update_metrics(task, start_time, finish_time, "Tier-1", metrics)

            # --- Tier-2 remote execution ---
            else:
                exec_time = task["offload_time"]

                if tier2_used_time + exec_time <= MAX_TIER2_TIME:
                    yield env.timeout(exec_time)
                    finish_time = env.now
                    tier2_used_time += exec_time
                    results.append((finish_time, "REMOTE", task))

                    update_metrics(task, start_time, finish_time, "Tier-2", metrics)
                else:
                    # fallback to Tier-1
                    allocations = tier1.inter_core_schedule([task])
                    for (ts, ttype), (res, ct) in allocations.items():
                        with resources[res].request() as req:
                            yield req
                            yield env.timeout(ct)
                            finish_time = env.now
                            results.append((finish_time, f"FALLBACK-{res}", task))

                            update_metrics(task, start_time, finish_time, "Fallback", metrics)
        else:
            yield env.timeout(1)

# === Metrics Updater ===
def update_metrics(task, start, finish, tier, metrics):
    deadline = task["deadline_ms"]
    response = finish - start
    met_deadline = response <= deadline

    metrics["total_tasks"] += 1
    metrics["tiers"][tier] += 1
    metrics["QoE"][task["QoE_class"]] += 1

    if met_deadline:
        metrics["deadline_met"] += 1
    else:
        metrics["deadline_miss"] += 1

    if tier == "Tier-1" or "LOCAL" in tier or "Fallback" in tier:
        metrics["energy"]["Tier-1"] += task["local_energy"]
    elif tier == "Tier-2":
        metrics["energy"]["Tier-2"] += task["offload_energy"]

# === Simulation Runner ===
def run_simulation(slam_file):
    env = simPy.Environment()
    task_queue = []
    results = []
    resources = create_resources(env)

    metrics = {
        "total_tasks": 0,
        "deadline_met": 0,
        "deadline_miss": 0,
        "tiers": defaultdict(int),
        "QoE": defaultdict(int),
        "energy": defaultdict(float),
    }

    env.process(task_source(env, slam_file, task_queue))
    env.process(scheduler(env, task_queue, results, resources, metrics))

    env.run(until=SIM_END)

    print("\n=== Simulation Results ===")
    for finish_time, where, task in results:
        print(f"[{where}] {task['task']} "
              f"| ts={task['timestamp']} "
              f"| deadline={task['deadline_ms']}ms "
              f"| QoE={task['QoE_class']} "
              f"| finish={finish_time:.1f}ms")

    print("\n=== Metrics Summary ===")
    print(f"Total tasks: {metrics['total_tasks']}")
    print(f"Deadline met: {metrics['deadline_met']} | Deadline missed: {metrics['deadline_miss']}")
    print(f"Deadline miss ratio: {(metrics['deadline_miss']/metrics['total_tasks']*100 if metrics['total_tasks'] else 0):.2f}%")

    print("\nTier Distribution:")
    for tier, count in metrics["tiers"].items():
        print(f"  {tier}: {count}")

    print("\nQoE Distribution:")
    for qclass, count in metrics["QoE"].items():
        print(f"  {qclass}: {count}")

    print("\nEnergy Consumption (J):")
    for tier, energy in metrics["energy"].items():
        print(f"  {tier}: {energy:.2f}")

    return results, metrics

if __name__ == "__main__":
    run_simulation("yolo_vo_state.json")
