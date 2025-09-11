# simulation.py
import simpy
import json
import heapq
from collections import defaultdict

import qoe
import tier1
import tier2

# === Simulation parameters ===
SIM_END = 500                # ms simulation horizon
MAX_TIER2_TIME = 100         # Tier-2 offload budget (ms)
BATCH_SIZE = 5               # Tier-1 batching size
USE_HEFT = True              # If True => use Dif-HEFT; else Dif-Min
SIMULATE_TIER1 = True        # If True => run SimPy processes for Tier-1 local tasks

# === SimPy resources for Tier-1 hardware ===
def create_resources(env):
    return {
        "CPU": simpy.Resource(env, capacity=1),
        "GPU": simpy.Resource(env, capacity=1),
        "DSP": simpy.Resource(env, capacity=1),
    }

# === Task Source (generates tasks and pushes into EDF heap) ===
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

        # Tier decision (Tier-1 vs Tier-2)
        task["assigned_tier"] = tier2.decide_tier(task)

        # push to EDF heap: (deadline, timestamp, task)
        heapq.heappush(task_queue, (task["deadline_ms"], task["timestamp"], task))

        # inter-arrival gap (simulated)
        yield env.timeout(5)

# === Local run process for a single task on a resource (SimPy) ===
def run_local_task(env, task, resource_name, exec_time, resources, results, start_time, tier_label, metrics):
    with resources[resource_name].request() as req:
        yield req
        # record actual start
        actual_start = env.now
        yield env.timeout(exec_time)
        finish_time = env.now
        results.append((finish_time, tier_label + "-" + resource_name, task))
        update_metrics(task, start_time, finish_time, tier_label, metrics)

# === Scheduler process (main) ===
def scheduler(env, task_queue, results, resources, metrics):
    tier2_used_time = 0

    # local buffer for Tier-1 batching
    tier1_batch = []

    while True:
        if task_queue:
            # pop earliest-deadline task
            _, _, task = heapq.heappop(task_queue)
            start_time = env.now

            # decide: Tier-1 local or Tier-2 remote (may have been decided earlier)
            if task["assigned_tier"] == "Tier-1":
                # add to local batch
                tier1_batch.append((task, start_time))

                # if batch full -> schedule batch
                if len(tier1_batch) >= BATCH_SIZE:
                    # extract task dicts
                    batch_tasks = [t for (t, st) in tier1_batch]

                    # choose scheduler
                    if USE_HEFT:
                        allocations = tier1.dif_heft_schedule(batch_tasks)
                    else:
                        allocations = tier1.inter_core_schedule(batch_tasks)

                    # For each allocation, schedule SimPy process (or simulate immediate completion)
                    # Build a mapping tid -> original start_time and task dict
                    tid_to_info = { (t["timestamp"], t["task"]): (t, st) for (t, st) in tier1_batch }

                    for tid, (res, ct) in allocations.items():
                        tdict, t_start_time = tid_to_info.get(tid, (None, None))
                        if tdict is None:
                            continue
                        exec_time = tier1.build_eet(tdict)[res]
                        if SIMULATE_TIER1:
                            env.process(run_local_task(env, tdict, res, exec_time, resources, results, t_start_time, "Tier-1", metrics))
                        else:
                            # immediate completion (no SimPy modeling)
                            finish_time = env.now + exec_time
                            results.append((finish_time, f"Tier-1-{res}", tdict))
                            update_metrics(tdict, t_start_time, finish_time, "Tier-1", metrics)

                    # clear batch
                    tier1_batch.clear()
                # end if batch full

            else:
                # Tier-2 remote execution attempt
                exec_time = task["offload_time"]

                if tier2_used_time + exec_time <= MAX_TIER2_TIME:
                    # perform remote execution (simply wait exec_time)
                    yield env.timeout(exec_time)
                    finish_time = env.now
                    tier2_used_time += exec_time
                    results.append((finish_time, "Tier-2", task))
                    update_metrics(task, start_time, finish_time, "Tier-2", metrics)
                else:
                    # fallback -> push to tier1 batch
                    task["assigned_tier"] = "Tier-1"
                    tier1_batch.append((task, start_time))
                    # if batch full, handle in next loop iterations
        else:
            # no tasks: if there are pending tier1_batch tasks but not full, optionally schedule them after some wait
            # simple policy: if batch not empty and environment progressed enough, flush small batches after timeout
            if tier1_batch:
                # we can flush small batches after short waiting period to avoid starvation
                # here we wait 1 ms, then if still some tasks, schedule them (smaller batch)
                yield env.timeout(1)
                if len(tier1_batch) > 0:
                    # schedule current small batch (flush)
                    batch_tasks = [t for (t, st) in tier1_batch]

                    if USE_HEFT:
                        allocations = tier1.dif_heft_schedule(batch_tasks)
                    else:
                        allocations = tier1.inter_core_schedule(batch_tasks)

                    tid_to_info = { (t["timestamp"], t["task"]): (t, st) for (t, st) in tier1_batch }
                    for tid, (res, ct) in allocations.items():
                        tdict, t_start_time = tid_to_info.get(tid, (None, None))
                        if tdict is None:
                            continue
                        exec_time = tier1.build_eet(tdict)[res]
                        if SIMULATE_TIER1:
                            env.process(run_local_task(env, tdict, res, exec_time, resources, results, t_start_time, "Tier-1", metrics))
                        else:
                            finish_time = env.now + exec_time
                            results.append((finish_time, f"Tier-1-{res}", tdict))
                            update_metrics(tdict, t_start_time, finish_time, "Tier-1", metrics)
                    tier1_batch.clear()
            else:
                # truly idle -> advance time a bit
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

    # energy accounting
    if tier == "Tier-1" or "Tier-1" in tier or "Fallback" in tier:
        metrics["energy"]["Tier-1"] += task.get("local_energy", 0)
    elif tier == "Tier-2":
        metrics["energy"]["Tier-2"] += task.get("offload_energy", 0)

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

    # start processes
    env.process(task_source(env, slam_file, voice_file, task_queue))
    env.process(scheduler(env, task_queue, results, resources, metrics))

    env.run(until=SIM_END)

    # Print results
    print("\n=== Simulation Results ===")
    for finish_time, where, task in results:
        print(f"[{where}] {task['task']} | ts={task['timestamp']} | deadline={task['deadline_ms']}ms | QoE={task['QoE_class']} | finish={finish_time:.1f}ms")

    # --- Metrics Summary ---
    print("\n=== Metrics Summary ===")
    print(f"Total tasks: {metrics['total_tasks']}")
    print(f"Deadline met: {metrics['deadline_met']} | Deadline missed: {metrics['deadline_miss']}")
    miss_ratio = (metrics['deadline_miss'] / metrics['total_tasks'] * 100) if metrics['total_tasks'] else 0
    print(f"Deadline miss ratio: {miss_ratio:.2f}%")

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
    # Example usage
    run_simulation("slam_output.jsonl", "voice_output.jsonl")
