import json
import heapq
import time
import threading

import  qoe         # QoE enrichment module
import tier1       # local edge node scheduling
import tier2       # remote edge node scheduling

# === Global Task Queue ===
task_queue = []

# === Buffers for batch scheduling ===
TIER1_BUFFER = []
TIER2_BUFFER = []
BATCH_SIZE = 5      # tasks per batch
FLUSH_INTERVAL = 0.1  # 100 ms max wait before flushing

last_flush_time = time.time()


# === Add Task ===
def add_task(json_event, task_type, last_voice=None):
    """Add new task to EDF queue after QoE enrichment and tier assignment."""
    task = qoe.enrich_task(json_event, task_type)
    task["assigned_tier"] = tier2.decide_tier(task)
    heapq.heappush(task_queue, (task["deadline_ms"], task["timestamp"], task))


# === Flush Functions ===
def flush_tier1():
    """Run Tier-1 inter-core scheduler on buffered tasks."""
    global TIER1_BUFFER
    if not TIER1_BUFFER:
        return
    result = tier1.inter_core_schedule(TIER1_BUFFER)
    print("\n=== Tier-1 Batch Scheduled ===")
    for (ts, ttype), (res, ct) in result.items():
        print(f"Task {ttype} at ts={ts} → {res}, completion={ct:.2f}ms")
    TIER1_BUFFER.clear()


def flush_tier2(max_time=200):
    """Run Tier-2 scheduler on buffered tasks."""
    global TIER2_BUFFER, TIER1_BUFFER
    if not TIER2_BUFFER:
        return
    executed, fallback = tier2.schedule_tier2(TIER2_BUFFER, max_time=max_time)
    print("\n=== Tier-2 Batch Scheduled ===")
    for t in executed:
        print(f"[REMOTE] Executed {t['task']} | ts={t['timestamp']} "
              f"| deadline={t['deadline_ms']}ms")
    for t in fallback:
        print(f"[FALLBACK] {t['task']} → pushed back to Tier-1")
        TIER1_BUFFER.append(t)
    TIER2_BUFFER.clear()


# === Scheduler Loop ===
def scheduler_loop():
    """Continuously pop EDF tasks and batch into Tier-1 or Tier-2."""
    global last_flush_time

    while True:
        now = time.time()

        if task_queue:
            _, _, task = heapq.heappop(task_queue)

            if task["assigned_tier"] == "Tier-1":
                TIER1_BUFFER.append(task)
                if len(TIER1_BUFFER) >= BATCH_SIZE:
                    flush_tier1()

            else:
                TIER2_BUFFER.append(task)
                if len(TIER2_BUFFER) >= BATCH_SIZE:
                    flush_tier2(max_time=200)

        else:
            # If idle, check time-based flush
            if (now - last_flush_time) >= FLUSH_INTERVAL:
                flush_tier1()
                flush_tier2()
                last_flush_time = now
            time.sleep(0.01)  # idle wait


# === Streaming Inputs ===
def stream_inputs(slam_file, voice_file):
    """Simulate continuous arrival of sensor events."""
    with open(slam_file) as sf, open(voice_file) as vf:
        slam_lines = iter(sf.readlines())
        voice_lines = iter(vf.readlines())

        while True:
            try:
                # Simulate SLAM input
                sline = next(slam_lines).strip()
                if sline:
                    event = json.loads(sline)
                    add_task(event, "slam")

                # Simulate Voice input
                vline = next(voice_lines).strip()
                if vline:
                    event = json.loads(vline)
                    add_task(event, "voice_recognition")

                time.sleep(0.05)  # simulate sensor arrival (50 ms)

            except StopIteration:
                print("No more events to stream.")
                break


# === Main Entry ===
if __name__ == "__main__":
    # Start scheduler thread
    sched_thread = threading.Thread(target=scheduler_loop, daemon=True)
    sched_thread.start()

    # Start input streaming
    stream_inputs("slam_output.jsonl", "voice_output.jsonl")

    # Let scheduler finish pending tasks
    time.sleep(2)
    flush_tier1()
    flush_tier2()
