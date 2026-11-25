from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from simulate_bsc import load_bits, run_trial
from ldpc import build_structure

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "final_bsc_metrics_restartable.csv"
CONFIG = ROOT / "results" / "final_bsc_config_restartable.json"
METHODS = ["uncoded", "hamming", "ldpc"]
P_VALUES = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
TRIALS_NONZERO = 100
MASTER_SEED = 20261125
MAX_WORKERS = min(8, os.cpu_count() or 1)
REFERENCE = None
SOURCE_BITS = None
STRUCTURE = None


def init_worker() -> None:
    global REFERENCE, SOURCE_BITS, STRUCTURE
    REFERENCE, SOURCE_BITS = load_bits()
    STRUCTURE = build_structure()


def child_seed(method_index: int, p_index: int, trial_index: int) -> int:
    sequence = np.random.SeedSequence([MASTER_SEED, method_index, p_index, trial_index])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def run_task(task: tuple[int, str, int, float, int]) -> dict[str, object]:
    method_index, method, p_index, p, trial_index = task
    row = run_trial(method, REFERENCE, SOURCE_BITS, p, child_seed(method_index, p_index, trial_index), STRUCTURE)
    row["trial"] = trial_index
    row["p_index"] = p_index
    return row


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tasks = []
    for p_index, p in enumerate(P_VALUES):
        count = 1 if p == 0.0 else TRIALS_NONZERO
        for method_index, method in enumerate(METHODS):
            for trial in range(count):
                tasks.append((method_index, method, p_index, p, trial))

    CONFIG.write_text(json.dumps({
        "methods": METHODS,
        "p_values": P_VALUES,
        "trials_at_zero": 1,
        "trials_per_nonzero_p": TRIALS_NONZERO,
        "master_seed": MASTER_SEED,
        "seed_convention": "SeedSequence([master_seed, method_index, p_index, trial_index])",
        "max_workers": MAX_WORKERS,
        "ldpc_iteration_cap": 50,
        "restartable_output": OUTPUT.name,
    }, indent=2) + "\n", encoding="utf-8")

    if OUTPUT.exists():
        OUTPUT.unlink()

    completed = 0
    fieldnames = None
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = None
        with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker) as executor:
            futures = [executor.submit(run_task, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                completed += 1
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"completed {completed}/{len(tasks)}", flush=True)
    print(f"Saved {completed} rows to {OUTPUT}", flush=True)
    print(f"Saved config to {CONFIG}", flush=True)


if __name__ == "__main__":
    main()
