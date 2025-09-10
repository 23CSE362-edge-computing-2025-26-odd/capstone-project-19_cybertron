# simulation.py
import simpy
import json
import heapq
import qoe
import tier1
import tier2
from collections import defaultdict

# === Parameters ===
SIM_END = 500          # ms simulation horizon
MAX_TIER2_TIME = 100   # budget for Tier-2 (same as tier2.py)

# === SimPy resources for Tier-1 hardware ===
def create_resources(env):
    return {
        "CPU": simpy.Resource(env, capacity=1),
        "GPU": simpy.Resource(env, capacity=1),
        "DSP": simpy.Resource(env, capacity=1),
    }

# === Task Source ===
def task_source(env, slam_file, voice_file, task_queue):
    slam_events = [json.loads(line) for line in open(slam_file)]
    voice_events = [json.loads(line) for line in open(voice_file)]
    events = [(e["ts"], "slam", e) for e in slam_events] + \
             [(e["ts"], "voice_recognition", e) for e in voice_events]
    events.sort(key=lambda x: x[0])

    prev_task_ts = None
    last_voice = None
    for ts, ttype, e in events:
        if ttype == "voice_recognition":
            last_voice = e.get("voice_text", "").lower()
        task = qoe.enrich_task(e, ttype, prev_task_ts, last_voice)
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

                        # metrics update
                        update_metrics(task, start_time, finish_time, "Tier-1", metrics)

            # --- Tier-2 remote execution ---
            else:
                exec_time = task["offload_time"]

                # check Tier-2 budget
                if tier2_used_time + exec_time <= MAX_TIER2_TIME:
                    yield env.timeout(exec_time)
                    finish_time = env.now
                    tier2_used_time += exec_time
                    results.append((finish_time, "REMOTE", task))

                    # metrics update
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

                            # metrics update
                            update_metrics(task, start_time, finish_time, "Fallback", metrics)
        else:
            yield env.timeout(1)

# === Metrics Updater ===
def update_metrics(task, start, finish, tier, metrics):
    deadline = task["deadline_ms"]
    response = finish - start
    met_deadline = response <= deadline

    # QoE-based metrics
    metrics["total_tasks"] += 1
    metrics["tiers"][tier] += 1
    metrics["QoE"][task["QoE_class"]] += 1

    # deadline stats
    if met_deadline:
        metrics["deadline_met"] += 1
    else:
        metrics["deadline_miss"] += 1

    # energy accounting
    if tier == "Tier-1" or "LOCAL" in tier or "Fallback" in tier:
        metrics["energy"]["Tier-1"] += task["local_energy"]
    elif tier == "Tier-2":
        metrics["energy"]["Tier-2"] += task["offload_energy"]

# === Simulation Runner ===
def run_simulation(slam_file, voice_file):
    env = simpy.Environment()
    task_queue = []
    results = []
    resources = create_resources(env)

    # metrics container
    metrics = {
        "total_tasks": 0,
        "deadline_met": 0,
        "deadline_miss": 0,
        "tiers": defaultdict(int),
        "QoE": defaultdict(int),
        "energy": defaultdict(float),
    }

    env.process(task_source(env, slam_file, voice_file, task_queue))
    env.process(scheduler(env, task_queue, results, resources, metrics))

    env.run(until=SIM_END)

    print("\n=== Simulation Results ===")
    for finish_time, where, task in results:
        print(f"[{where}] {task['task']} "
              f"| ts={task['timestamp']} "
              f"| deadline={task['deadline_ms']}ms "
              f"| QoE={task['QoE_class']} "
              f"| finish={finish_time:.1f}ms")

    # --- Metrics Summary ---
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
    run_simulation("slam_output.jsonl", "voice_output.jsonl")
