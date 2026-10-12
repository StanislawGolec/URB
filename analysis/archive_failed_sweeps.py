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
        if source.exists() and (source / "BenchmarkMetrics.csv").exists():
            raise ValueError(f"Refusing to move a finished run with metrics: {source}")
        if source.exists() and archived.exists():
            print(f"Archive already exists ({archived.name}); remaining files in {source.name} will be merged into archive.")
        elif source.exists():
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
            for item in list(source.iterdir()):
                dest = archived / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            source.rmdir()
        elif source.exists():
            source.rename(archived)
    print("Archived 15 failed folders; original experiment IDs are available for retry.")


if __name__ == "__main__":
    main()
