"""Preview or synchronize exactly the 26 audited offline W&B runs.

Run without --execute first. Original run IDs and recorded failure states are
preserved by wandb sync. Requires a working W&B login and outbound network access.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def validate_manifest(manifest):
    runs = manifest["runs"]
    if len(runs) != 26 or len({r["run_id"] for r in runs}) != 26:
        raise ValueError("The manifest must contain exactly 26 distinct original runs")
    for run in runs:
        source = (ROOT / run["file"]).resolve()
        source.relative_to(ROOT / "scripts/wandb")
        if hashlib.sha256(source.read_bytes()).hexdigest() != run["sha256"]:
            raise ValueError(f"Journal changed since analysis: {run['experiment']}")
        for image in run["media"]:
            path = (ROOT / image["file"]).resolve()
            path.relative_to(source.parent / "files")
            if hashlib.sha256(path.read_bytes()).hexdigest() != image["sha256"]:
                raise ValueError(f"Plot changed since analysis: {path}")
    return runs


def verify_remote(remote, source):
    errors = []
    expected_states = {"finished"} if source["exit_code"] == 0 else {"failed", "crashed"}
    if remote.state not in expected_states:
        errors.append(f"state {remote.state!r}, expected one of {sorted(expected_states)!r}")
    if remote.name != source["experiment"]:
        errors.append("run display name differs")
    for key, expected in source["benchmark"].items():
        actual = remote.summary.get(f"benchmark/{key}")
        if not isinstance(actual, (float, int)) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            errors.append(f"benchmark/{key} differs")
    progress_key = "episode" if source["experiment"].startswith("iql") else "iteration"
    if source["training_progress"] and remote.summary.get(progress_key) != source["training_progress"]:
        errors.append("training progress differs")
    remote_files = {f.name for f in remote.files()}
    for image in source["media"]:
        filename = image["file"].split("/files/", 1)[1]
        if filename not in remote_files:
            errors.append(f"missing plot {filename}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "analysis/reports/10_plus_16/manifest.json")
    parser.add_argument("--execute", action="store_true", help="Upload to W&B; otherwise only verify and print the plan")
    parser.add_argument("--entity", help="Explicit destination override; defaults to original recorded entity")
    parser.add_argument("--project", help="Explicit destination override; defaults to original recorded project")
    args = parser.parse_args()
    runs = validate_manifest(json.loads(args.manifest.read_text(encoding="utf-8")))
    for run in runs:
        entity, project = args.entity or run["entity"], args.project or run["project"]
        print(f"{run['status']:16s} {entity}/{project}/{run['run_id']}  {run['experiment']}")
    if not args.execute:
        print("Verified 26 journals and their plot hashes. No upload attempted. Add --execute to synchronize.")
        return

    import wandb
    api = wandb.Api(timeout=20)
    # Authentication/network preflight precedes all uploads. Never print credentials.
    api.viewer
    receipts = []
    receipt_file = args.manifest.parent / "sync_receipt.json"
    for run in runs:
        entity, project = args.entity or run["entity"], args.project or run["project"]
        target = f"{entity}/{project}/{run['run_id']}"
        receipt = {"run_id": run["run_id"], "url": f"https://wandb.ai/{entity}/{project}/runs/{run['run_id']}",
                   "verified": False}
        try:
            command = [sys.executable, "-m", "wandb", "sync", "--include-synced",
                       "--entity", entity, "--project", project, str(ROOT / run["file"])]
            subprocess.run(command, cwd=ROOT, check=True, timeout=600)
            fresh_api = wandb.Api(timeout=20)
            errors = verify_remote(fresh_api.run(target), run)
            receipt["errors"] = errors
            receipt["verified"] = not errors
        except (Exception, KeyboardInterrupt) as error:
            receipt["errors"] = [f"{type(error).__name__}: {error}"]
            receipts.append(receipt)
            receipt_file.write_text(json.dumps(receipts, indent=2), encoding="utf-8")
            raise
        receipts.append(receipt)
        receipt_file.write_text(json.dumps(receipts, indent=2), encoding="utf-8")
    if not all(r["verified"] for r in receipts):
        raise SystemExit(f"Some remote checks failed. Inspect {receipt_file}")
    print("All 26 original runs verified remotely, including metrics, progress, plots and failure states.")


if __name__ == "__main__":
    main()
