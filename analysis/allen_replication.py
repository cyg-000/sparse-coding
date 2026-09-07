"""Run Allen VBN sessions in isolated processes and aggregate the results.

AllenSDK retains large NWB/HDF5 object graphs after a session is processed on
Windows.  Each session is therefore analyzed in a fresh child process.  The
operating system releases all memory and file handles when that child exits.
"""
from __future__ import annotations
import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from fisher_redundancy_core import clustered_inference, seedof


HERE = Path(__file__).parent
WORKER = HERE / "nn_redundancy_allen_vbn_replication.py"
UNIT_COUNTS = (20, 30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort)
    parts = HERE / "outputs" / "allen_vbn_parts"
    parts.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"

    for index, record in cohort.iterrows():
        session_id = int(record["ecephys_session_id"])
        relative = Path("allen_vbn_parts") / f"session_{session_id}.json"
        target = HERE / "outputs" / relative
        if target.exists():
            try:
                cached = json.loads(target.read_text(encoding="utf-8"))
                if cached.get("audit") and cached.get("rows"):
                    print(f"[{index + 1}/{len(cohort)}] reuse {session_id}", flush=True)
                    continue
            except Exception:
                pass
        command = [
            sys.executable, str(WORKER),
            "--cache-dir", str(args.cache_dir),
            "--cohort", str(args.cohort),
            "--start-index", str(index),
            "--limit", "1",
            "--output-name", str(relative),
        ]
        completed = subprocess.run(command, env=env, capture_output=True, text=True)
        if completed.returncode:
            print(completed.stdout[-4000:])
            print(completed.stderr[-4000:], file=sys.stderr)
            raise RuntimeError(f"session {session_id} failed with code {completed.returncode}")
        print(f"[{index + 1}/{len(cohort)}] completed {session_id}", flush=True)

    rows, audit = [], []
    for session_id in cohort["ecephys_session_id"].astype(int):
        payload = json.loads((parts / f"session_{session_id}.json").read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
        audit.extend(payload["audit"])

    summary = {}
    for n_units in UNIT_COUNTS:
        rr = [row for row in rows if row["n_units"] == n_units]
        summary[str(n_units)] = {}
        for field in ("fisher", "align_cv", "align_cv_null", "sharing"):
            summary[str(n_units)][field] = {
                "animal": clustered_inference(rr, field, "animal", seedof(n_units, field, "animal")),
                "session": clustered_inference(rr, field, "session", seedof(n_units, field, "session")),
            }

    output = HERE / "outputs" / "allen_vbn_active_passive_redundancy.json"
    final = {
        "metadata": {
            "dataset": "Allen Visual Behavior Neuropixels v0.5.0",
            "axis": "image change versus repeat",
            "window_s": [0.05, 0.25],
            "unit_counts": list(UNIT_COUNTS),
            "n_label_null": 20,
            "primary": "20 units; task-specific sharing; animal-level inference",
            "memory_strategy": "one fresh child process per NWB session",
        },
        "audit": audit,
        "summary": summary,
        "rows": rows,
    }
    output.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("saved", output)


if __name__ == "__main__":
    main()
