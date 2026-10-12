"""Move only the 15 verified CUDA-OOM result folders aside before rerunning.

By default prints a preview. --execute performs checked, non-overwriting renames.
The original W&B journals and Slurm logs remain in their existing locations.
"""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "analysis/reports/10_plus_16/manifest.json"


def archive_pairs(manifest):
    runs = [run for run in manifest["runs"] if run["status"] == "failed_cuda_oom"]
    if len(runs) != 15 or len({run["experiment"] for run in runs}) != 15:
        raise ValueError("Expected exactly 15 distinct CUDA-OOM runs")
    for run in runs:
        name = run["experiment"]
        if not name.startswith("qmix_clusters_") or "/" in name or "\\" in name:
            raise ValueError(f"Unexpected experiment name: {name}")
        yield ROOT / "results" / name, ROOT / "results" / f"{name}_failed_oom"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    pairs = list(archive_pairs(json.loads(MANIFEST.read_text(encoding="utf-8"))))
    for source, archived in pairs:
        if source.exists() and archived.exists():
            files = sorted(p.name for p in source.iterdir())
            if files not in (["exp_config.json", "paths.csv"], []):
                raise ValueError(f"Refusing to remove unexpected files in {source}: {files}")
            print(f"Archive already exists ({archived.name}); redundant source will be cleaned: {source.name}")
        elif source.exists():
            files = sorted(p.name for p in source.iterdir())
            if files != ["exp_config.json", "paths.csv"]:
                raise ValueError(f"Refusing to move a modified or finished run: {source}: {files}")
            config = json.loads((source / "exp_config.json").read_text(encoding="utf-8"))
            task_name = config.get("task_config", "")
            expected_tag = (f"alt{task_name.removeprefix('altruistic_')}"
                            if task_name.startswith("altruistic_") else task_name)
            if expected_tag not in source.name or config.get("algorithm") != "qmix_torchrl":
                raise ValueError(f"Experiment config does not match expected failed run: {source}")
            print(f"{source.name} -> {archived.name}")
        elif archived.exists():
            print(f"Already archived: {archived.name}")
        else:
            raise FileNotFoundError(f"Neither failed folder nor archive exists: {source}")
    if not args.execute:
        print("Preview only; add --execute to archive the failed folders.")
        return
    for source, archived in pairs:
        if source.exists() and archived.exists():
            shutil.rmtree(source)
        elif source.exists():
            source.rename(archived)
    print("Archived 15 failed folders; original experiment IDs are available for retry.")


if __name__ == "__main__":
    main()
