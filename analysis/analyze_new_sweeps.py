"""Audit the 10 penetration and 16 reward runs without changing source results.

Requires wandb to read the original offline records; charts also use reportlab.
Example: python analysis/analyze_new_sweeps.py --charts
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[1]
MIXTURES = [(100, 0), (85, 15), (70, 30), (50, 50), (30, 70), (15, 85), (5, 95), (0, 100)]


def expected_experiments():
    names = [f"qmix_clusters_alt{p}_ing2_s42" for p in range(10, 101, 10)]
    for algorithm in ("qmix", "iql"):
        for individual, collective in MIXTURES:
            label = "05" if individual == 5 else str(individual)
            names.append(f"{algorithm}_clusters_mixed_reward_ind{label}_all{collective}_ing2_s42")
    return names


def record_values(items):
    return {item.key or ".".join(item.nested_key): json.loads(item.value_json) for item in items}


def scan_run(path):
    # W&B's record reader validates the journal framing/checksums.
    from wandb.proto.wandb_internal_pb2 import Record
    from wandb.sdk.internal.datastore import DataStore

    reader = DataStore()
    reader.open_for_scan(str(path))
    result = {"file": path.relative_to(ROOT).as_posix(), "history": [], "summary": {}, "exit_code": None}
    output_parts = []
    try:
        while (data := reader.scan_data()) is not None:
            record = Record()
            record.ParseFromString(data)
            kind = record.WhichOneof("record_type")
            if kind == "run":
                run = record.run
                result.update(id=run.run_id, name=run.display_name, entity=run.entity,
                              project=run.project, git_commit=run.git.commit,
                              config=record_values(run.config.update))
            elif kind == "history":
                result["history"].append(record_values(record.history.item))
            elif kind == "summary":
                result["summary"].update(record_values(record.summary.update))
            elif kind == "exit":
                result["exit_code"] = record.exit.exit_code
            elif kind == "output_raw":
                output_parts.append(record.output_raw.line)
    finally:
        reader.close()
    output = re.sub(r"\x1b\[[0-9;]*m", "", "".join(output_parts))
    result["cuda_oom"] = "CUDA out of memory" in output
    result["test_completed"] = bool(re.search(r"Testing:\s*100%", output))
    result["metrics_finished"] = "[W&B] Logged 21 benchmark metrics" in output
    return result


def derive_metrics(metrics, config):
    """Derived values retain source minutes; percentages use each run's baseline."""
    n, cav = config["num_agents"], config["num_machines"]
    human = n - cav
    pre, test = metrics["t_pre"], metrics["t_test"]
    cav_pre = (n * pre - human * metrics["t_HDV_pre"]) / cav
    return {
        "system_change_pct": 100 * (test / pre - 1),
        "system_change_seconds": 60 * (test - pre),
        "cav_baseline_inferred_min": cav_pre,
        "cav_change_pct": 100 * (metrics["t_CAV"] / cav_pre - 1),
        "human_change_pct": 100 * (metrics["t_HDV_test"] / metrics["t_HDV_pre"] - 1),
        "test_population_reconciliation_min": test - (cav * metrics["t_CAV"] + human * metrics["t_HDV_test"]) / n,
    }


def build_inventory():
    expected = expected_experiments()
    by_name = {}
    for path in sorted((ROOT / "scripts/wandb").glob("offline-run-*/run-*.wandb")):
        run = scan_run(path)
        if run.get("name") in expected:
            by_name.setdefault(run["name"], []).append(run)
    rows, manifest, histories = [], [], []
    for name in expected:
        candidates = by_name.get(name, [])
        if len(candidates) != 1:
            raise ValueError(f"Expected one archived run for {name}, found {len(candidates)}; select explicitly before syncing")
        run = candidates[0]
        folder = ROOT / "results" / name
        if run["cuda_oom"] and (ROOT / "results" / f"{name}_failed_oom").exists():
            folder = ROOT / "results" / f"{name}_failed_oom"
        config = json.loads((folder / "exp_config.json").read_text(encoding="utf-8"))
        # Saved W&B configuration must agree with the result directory's configuration.
        for key, value in config.items():
            if key in run["config"] and value != run["config"][key]:
                raise ValueError(f"Config mismatch: {name}: {key}")
        source = folder / "metrics/BenchmarkMetrics.csv"
        metrics = {}
        if source.exists():
            records = list(csv.DictReader(source.open(encoding="utf-8", newline="")))
            if len(records) != 1:
                raise ValueError(f"Expected one benchmark row: {source}")
            metrics = {k: float(v) for k, v in records[0].items() if v}
            for key, value in metrics.items():
                logged = run["summary"].get(f"benchmark/{key}")
                if logged is None or not math.isclose(value, logged, rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(f"Benchmark/W&B mismatch: {name}: {key}")
        if run["cuda_oom"]:
            status = "failed_cuda_oom"
        elif run["exit_code"] == 0 and metrics and run["test_completed"]:
            status = "completed"
        else:
            status = "incomplete"
        algorithm = "IQL" if name.startswith("iql") else "QMIX"
        behavior = config["av_behavior"]
        alpha = max(0.0, -behavior[0]) if isinstance(behavior, list) else 0.0
        row = dict(experiment=name, batch="mixed_reward" if "mixed_reward" in name else "penetration",
                   algorithm=algorithm, cav_share=config["num_machines"] / config["num_agents"],
                   nominal_cav_share=config["ratio_machines"], individual_weight=alpha,
                   status=status, run_id=run["id"], git_commit=run["git_commit"],
                   exit_code=run["exit_code"], test_completed=run["test_completed"],
                   runtime_hours=run["summary"].get("_runtime", 0) / 3600,
                   source_config=(folder / "exp_config.json").relative_to(ROOT).as_posix(),
                   source_metrics=source.relative_to(ROOT).as_posix() if metrics else "")
        if metrics:
            row.update(metrics)
            row.update(derive_metrics(metrics, config))
        training = [h for h in run["history"] if "iteration" in h or "episode" in h]
        row["training_records"] = len(training)
        row["training_progress"] = max((h.get("iteration", h.get("episode", 0)) for h in training), default=0)
        for key in ("train/loss", "train/mean_reward"):
            values = [h[key] for h in training if key in h]
            if values:
                prefix = key.replace("train/", "")
                row[f"{prefix}_nonfinite"] = sum(not math.isfinite(v) for v in values)
                row[f"{prefix}_first20"] = statistics.fmean(values[:20])
                row[f"{prefix}_last20"] = statistics.fmean(values[-20:])
                row[f"{prefix}_last"] = values[-1]
        for history in training:
            histories.append({"experiment": name, **{k: v for k, v in history.items() if not k.startswith("_")}})
        media = []
        summary = run["summary"]
        for key, value in summary.items():
            if key.endswith(".path") and key.startswith("plots/"):
                media_file = ROOT / run["file"]
                media_file = media_file.parent / "files" / value
                digest = hashlib.sha256(media_file.read_bytes()).hexdigest()
                if digest != summary[key[:-5] + ".sha256"]:
                    raise ValueError(f"Media hash mismatch: {media_file}")
                media.append({"file": media_file.relative_to(ROOT).as_posix(), "sha256": digest})
        requirements = ROOT / run["file"]
        requirements = requirements.parent / "files/requirements.txt"
        versions = [s for s in requirements.read_text().splitlines()
                    if re.match(r"^(routerl|torch|torchrl|wandb|numpy|pandas)==", s)]
        manifest.append({"experiment": name, "run_id": run["id"], "entity": run["entity"],
                         "project": run["project"], "status": status, "file": run["file"],
                         "sha256": hashlib.sha256((ROOT / run["file"]).read_bytes()).hexdigest(),
                         "exit_code": run["exit_code"], "git_commit": run["git_commit"],
                         "versions": versions, "media": media,
                         "benchmark": metrics, "training_progress": row["training_progress"]})
        rows.append(row)
    return rows, manifest, histories


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(rows):
    done = [r for r in rows if r["status"] == "completed"]
    lines = ["# Analysis of the 10 + 16 experiments", "", "Analysis date: 2026-10-12. Source checkout: 7e43300.", "",
             f"{len(done)}/26 runs completed; {sum(r['status'] == 'failed_cuda_oom' for r in rows)} failed with CUDA out of memory. "
             "The 11 benchmark CSVs agree with all 21 corresponding W&B summary metrics. "
             "Original journal exit codes and completed testing progress corroborate completion. "
             "Raw episode CSVs are absent, so per-day coverage, uncertainty and corrected cumulative learning costs cannot be independently recomputed.", "",
             "## Completed results", "", "Travel times are minutes. Positive changes mean worse travel time. "
             "System changes use each run's own pre-mutation baseline (4.847303 min). "
             "Human changes use the baseline of the agents who remained human.", "",
             "| Experiment | System | CAV | Human | System change | Human change |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in done:
        label = f"QMIX {r['nominal_cav_share']:.0%} CAV" if r['batch'] == 'penetration' else f"IQL individual {r['individual_weight']:.0%}"
        lines.append(f"| {label} | {r['t_test']:.6f} | {r['t_CAV']:.6f} | {r['t_HDV_test']:.6f} | {r['system_change_pct']:+.3f}% | {r['human_change_pct']:+.3f}% |")
    lines.extend(["", "## Interpretation", "",
        "The best observed IQL result is individual weight 100% (4.796552 min, 1.047% below baseline, about 3.05 seconds per trip). "
        "The 70% individual variant is close (4.805529 min); the gap is only 0.54 seconds per trip. "
        "One seed cannot establish that the ranking is robust. All eight IQL treatments slightly increase the travel time of the remaining humans. "
        "The best system result therefore combines a CAV benefit with a small human cost, rather than a benefit for both groups.", "",
        "With 15%, 5% or 0% individual reward, IQL is worse than the all-human baseline. "
        "The pure all-population reward produces 5.100479 min (+5.223%). Increasing the social weight did not improve system performance in these trained policies. "
        "A plausible mechanism is diluted individual credit from the shared reward; these data do not isolate that mechanism from optimization and exploration effects.", "",
        "Completed altruistic QMIX runs worsen system travel time by 2.880%, 7.093%, and 10.273% at 10%, 20%, and 30% CAVs. "
        "CAV travel times remain around 6.4 min. The saved 30% travel-time plot rises during training and then plateaus at poor performance. "
        "Training completion is not evidence of successful learning.", "",
        "All eight QMIX reward-mixture runs failed before a logged training iteration, so there is no completed, matched-penetration QMIX/IQL comparison. "
        "The equivalent QMIX altruistic-40 and individual-0/all-100 runs both failed, preventing the intended reproducibility comparison.", "",
        "## Training diagnostics", "",
        "| QMIX CAV share | Mean loss, first 20 iterations | Mean loss, last 20 | Mean reward, first 20 | Mean reward, last 20 | Runtime (h) |",
        "|---|---:|---:|---:|---:|---:|"])
    for r in done:
        if r["algorithm"] == "QMIX":
            lines.append(f"| {r['nominal_cav_share']:.0%} | {r['loss_first20']:.3e} | {r['loss_last20']:.3e} | {r['mean_reward_first20']:.6f} | {r['mean_reward_last20']:.6f} | {r['runtime_hours']:.2f} |")
    lines.extend(["", "All 600 QMIX loss values are finite, but are approximately 10^32–10^34. "
        "These extreme TD losses require investigating value scale, observations, masks and gradients. "
        "The logged reward worsens within every completed QMIX run. Across penetration levels its magnitude is also diluted by the number of sequential CAV decisions; do not compare the uncorrected curves directly. "
        "IQL journals contain 4,000 episode/epsilon records each, but no numeric loss or travel-time history. Its saved plots support qualitative inspection only.", "",
        "## Failure and runtime evidence", "",
        "All 15 failed runs report CUDA OOM on an RTX 2080 Ti with about 10.58 GiB available device capacity. "
        "The eight mixed QMIX runs and altruistic-40 fail while Adam initializes optimizer state (a further 1.84 GiB allocation). "
        "Altruistic-100 fails creating the first mixer hypernetwork (11.49 GiB allocation). More walltime or host RAM does not fix this VRAM requirement.", "",
        "The mixer state contains all CAV observations. With 15 features and embedding size 192, its first weight matrix contains 2880*N_CAV^2 parameters. "
        "At 100% CAVs one FP32 copy needs about 12.34 GB (11.49 GiB). Online/target weights, gradients and Adam states multiply this requirement. "
        "Validate the GPU capacity on bonk before rerunning; access to a new partition alone does not establish sufficient GPU memory.", "",
        "The journals record RouteRL 2.0.0, Torch 2.6.0, TorchRL 0.7.0 and W&B 0.29.0. "
        "Run commits are recorded individually in manifest.json. Altruistic-10/20 use babe8cb, the remaining penetration runs fece3e2, and reward-mixture runs 2ebca12.", "",
        "## Methodological limits and code findings", "",
        "- Only seed 42 is represented. Testing days are not independent policy replications; no seed-level confidence intervals or significance claims are justified.",
        "- IQL records the environment config as trip_info_eta but the executed script hardcodes previous_agents_plus_start_time. QMIX uses ETA plus an OD embedding. The information given to the algorithms differs.",
        "- QMIX evaluates at epsilon 0.1; IQL evaluates greedily. Match evaluation policy before comparing algorithms.",
        "- QMIX mixes all proposed actions despite executing only the active AEC agent's action. Validate inactive-agent masking and credit assignment with a small deterministic environment.",
        "- A concrete numerical lead is invalid-route ETA padding: the local RouteRL dependency fills missing cluster free-flow values with 1e9 (environment.py:863), and TripInfoWithETA feeds these values into observations. QMIX passes those observations to its mixer without normalization. This can overwhelm the hypernetworks even when action selection is masked. The archived server package version is known, but its source/observations are not included, so this is a hypothesis to reproduce, not a confirmed explanation of the enormous losses.",
        "- Gamma 0.95 is applied per sequential decision. Terminal reward discount to the first decision is 0.00534 at 103 CAVs and 6.31e-10 at 414 CAVs. The exact runtime environment semantics should be reproduced before altering the algorithm.",
        "- cost_of_learning is a max-minus-min range, not cumulative baseline-relative cost. It is preserved as an original field in summary.csv but excluded from conclusions.",
        "- avg_time_lost and its subgroup columns use inconsistent aggregation. Effect_of_change uses the remaining-human baseline, rather than the same cohort before it became CAVs. Derived CAV baseline values in summary.csv use population-weighted reconstruction, assuming consistent agent coverage.",
        "- The 100% CAV metric path suppresses a valid human-learning baseline. It did not affect these completed metrics because that run failed before evaluation.",
        "- The saved raw episode files, optimizer state and model checkpoints are absent. Exact recalculation and greedy reevaluation cannot be performed from this checkout.", "",
        "## Recommended next experiments", "",
        "1. Resolve the QMIX numerical scale and AEC formulation on a tiny deterministic case before a full sweep.",
        "2. Match observations and evaluation epsilon, save checkpoints, terminal rewards, gradient norms, and episode-level travel times.",
        "3. Compare IQL individual weights 100%, 70% and 50%, plus pure collective control, on at least five matched seeds. The present 100%/70% gap is too small for a robust ranking.",
        "4. Verify bonk GPU VRAM, partition/QoS association and runtime limits; select a tractable mixer configuration or explicitly label any architecture change as a new experiment.",
        "5. Preserve the original failed runs in W&B. Failure is part of the experiment outcome.", "",
        "## Reproduction and provenance", "",
        "Run `python analysis/analyze_new_sweeps.py --charts` with wandb and reportlab installed. "
        "The script leaves original CSVs, journals and plots untouched. summary.csv contains all 26 statuses, original metrics, derived comparisons and source paths. "
        "training_history.csv contains the numerical training records. manifest.json identifies the 26 original W&B run IDs, journal hashes, dependency versions and plot hashes. "
        "Run `python analysis/sync_new_sweeps.py` to preview synchronization, then add `--execute` on a machine with W&B network access and an existing login. "
        "The destination defaults to the entity/project recorded in the original journals. A local manifest is not evidence of a successful cloud upload.", "",
        "Current upload status (2026-10-12): an authenticated API connection attempt was blocked by session network permissions (WinError 10013). "
        "No remote upload was verified in this session. See upload_status.json; execute the prepared synchronizer from a network-enabled environment.", ""])
    return "\n".join(lines)


def draw_charts(rows, output):
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.shapes import Drawing, String, Rect
    from reportlab.graphics import renderSVG
    from reportlab.lib.colors import HexColor

    chart = Drawing(1060, 480)
    chart.add(Rect(0, 0, 1060, 480, fillColor=HexColor("#ffffff"), strokeColor=None))
    chart.add(String(35, 450, "26 experiments: 11 completed, 15 CUDA out-of-memory failures", fontSize=20, fontName="Helvetica-Bold"))
    chart.add(String(35, 425, "System travel time relative to the same run's human baseline. Lower is better. One seed per treatment.", fontSize=11))
    for index, batch in enumerate(("penetration", "mixed_reward")):
        selected = [r for r in rows if r["batch"] == batch and r["status"] == "completed"]
        points = sorted([(100 * (r["nominal_cav_share"] if index == 0 else r["individual_weight"]), r["system_change_pct"]) for r in selected])
        plot = LinePlot()
        plot.x, plot.y, plot.width, plot.height = 80 + index * 510, 110, 405, 240
        plot.data = [points, [(0, 0), (100, 0)]]
        plot.xValueAxis.valueMin, plot.xValueAxis.valueMax, plot.xValueAxis.valueStep = 0, 100, 20
        plot.yValueAxis.valueMin, plot.yValueAxis.valueMax, plot.yValueAxis.valueStep = (-1, 11, 2) if index == 0 else (-2, 6, 2)
        plot.lines[0].strokeColor = HexColor("#bd4e2c" if index == 0 else "#225db0")
        plot.lines[0].strokeWidth = 2.5
        plot.lines[1].strokeColor = HexColor("#929ca8")
        plot.lines[1].strokeDashArray = [4, 4]
        plot.joinedLines = 1
        plot.xValueAxis.labels.fontSize = plot.yValueAxis.labels.fontSize = 10
        chart.add(plot)
        chart.add(String(plot.x, 380, "QMIX altruistic penetration" if index == 0 else "IQL reward mixtures (40% CAV)", fontSize=15, fontName="Helvetica-Bold"))
        chart.add(String(plot.x, 357, "Change from baseline (%)", fontSize=10))
        chart.add(String(plot.x + 100, 75, "CAV share (%)" if index == 0 else "Individual reward weight (%)", fontSize=12))
        chart.add(String(plot.x, 46, "40-100%: no results, CUDA OOM" if index == 0 else "All eight matched QMIX runs: CUDA OOM", fontSize=10, fillColor=HexColor("#8a3434")))
    renderSVG.drawToFile(chart, str(output / "comparison.svg"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/reports/10_plus_16")
    parser.add_argument("--charts", action="store_true")
    args = parser.parse_args()
    rows, manifest, histories = build_inventory()
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "summary.csv", rows)
    write_csv(args.output / "training_history.csv", histories)
    (args.output / "manifest.json").write_text(json.dumps({"schema_version": 1, "runs": manifest}, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "report.md").write_text(report(rows), encoding="utf-8")
    if args.charts:
        draw_charts(rows, args.output)
    print(json.dumps({"runs": len(rows), "completed": sum(r["status"] == "completed" for r in rows),
                      "cuda_oom": sum(r["status"] == "failed_cuda_oom" for r in rows),
                      "training_records": len(histories), "output": str(args.output)}))


if __name__ == "__main__":
    main()
