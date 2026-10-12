# Bonk execution and archived results

The server screenshot shows partition `rknodes` with six GPUs per node and
`1024000` MB of **node RAM**. It does not show GPU models or their VRAM. The
user's association lists QoS `batch,big,normal,pr+`, not `big_bonk`; the retry
defaults to `rknodes` / `big`. The previous seven-day request, 32 GB host RAM,
four CPUs and one GPU are retained. Slurm must still validate that this
partition/QoS combination and walltime are allowed for the account.

Check the server's actual association, time limit and GPU types before submitting:

```bash
sinfo -p rknodes -o '%P %l %G %m %a'
scontrol show partition rknodes
sacctmgr show assoc where user="$USER" format=User,Partition,QOS
sacctmgr show qos big format=Name,MaxWall
TASKS=1 bash slurm_requests/launch_retry_failed.sh --check-access
```

The completed experiments used RTX 2080 Ti GPUs with 10.58 GiB capacity.
Fifteen QMIX runs exhausted GPU memory. Host `--mem` and walltime do not increase
VRAM. At 414 CAVs, the first mixer hypernetwork plus online/target weights,
gradients and Adam moments already needs about 9.19 GiB; at 1,035 CAVs this
one lower bound rises to 57.46 GiB. Replay, other parameters and activations
need more. The preflight checks free VRAM with at least 2 GiB or 25% of this
lower bound reserved for additional allocations. It screens out obviously
insufficient GPUs but cannot prove that a full training run will fit.

For a quick inventory, use `sinfo -N -p rknodes -o '%N %G %m %T'`. GRES `gpu:6`
means six GPUs, not six GB VRAM. The `big` QoS on this cluster accepts only
`sbatch`, so an interactive `srun` allocation will be rejected. Inspect an
allocated GPU with a short batch job:

```bash
mkdir -p slurm_requests/container_printouts
sbatch slurm_requests/gpu_probe.sbatch
# After completion, use the job ID printed by sbatch:
cat slurm_requests/container_printouts/gpu-probe-<job-id>.out
```

This requests a real five-minute GPU allocation. Its output, or the first
lines of a job log, establishes the GPU model and per-device VRAM. Even an
80 GB GPU needs a trial at 100% CAV because 57.46 GiB is only one lower bound.

Preview or submit a sweep from any directory:

```bash
bash slurm_requests/launch_sweep.sh altruistic --dry-run
bash slurm_requests/launch_sweep.sh mixed --dry-run
ARRAY_LIMIT=2 bash slurm_requests/launch_sweep.sh mixed
```

`PARTITION`, `QOS`, `WALLTIME` and `ARRAY_LIMIT` can override submission settings.
New runs use suffix `_bonk` to keep the original experimental results intact.
Set `EXP_SUFFIX` to another unique suffix for subsequent reruns. Do not reuse an
existing experiment ID. The legacy `launch_alt_sweep.sh all` now includes 100%.
Every job propagates Python/container failures through `tee` using `pipefail`.

## W&B

Authenticate on the server with `wandb login` in the Conda URB environment, or
provide `WANDB_API_KEY` through the server's existing secret environment. Never
commit the key. The default destination is `aintern26coexistence/URB-Traffic-Routing`,
matching the original run journals; `WANDB_ENTITY` and `WANDB_PROJECT` override it.
New jobs require successful online initialization before simulation. Deliberate
offline operation requires `WANDB_REQUIRE_ONLINE=0 WANDB_MODE=offline`.

The transferred 26 runs were recorded offline because the server had no W&B API
key configured. Preview the exact journal and plot hashes, then upload from a
machine with a working W&B login and network access:

```bash
python analysis/sync_new_sweeps.py
python analysis/sync_new_sweeps.py --execute
```

This preserves original run IDs, history, plots, metadata and failed exit states.
It does not create substitute successful runs for failed experiments. The script
checks remote status, summary metrics, training progress and plot presence and
writes `analysis/reports/10_plus_16/sync_receipt.json` after each upload. Only a
verified remote receipt establishes upload success; `.syncstate` alone does not.

Analysis: `analysis/reports/10_plus_16/report.md`. Rebuild with
`python analysis/analyze_new_sweeps.py --charts` (requires W&B and ReportLab).

## Retry exactly the failed jobs

The 15 OOM failures are altruistic QMIX at 40, 50, 60, 70, 80, 90 and 100%
CAVs, and all eight QMIX reward mixtures at 40% CAVs. IQL (eight mixtures) and
altruistic QMIX at 10, 20 and 30% completed and are excluded from this retry.

```bash
bash slurm_requests/launch_retry_failed.sh --dry-run
TASKS=1 bash slurm_requests/launch_retry_failed.sh --check-access
TASKS=1 bash slurm_requests/launch_retry_failed.sh
TASKS=2-15 ARRAY_LIMIT=2 bash slurm_requests/launch_retry_failed.sh
```

The launcher checks each original OOM folder, then renames it with `_failed_oom`
before submission. It never overwrites a completed or modified folder. Original
W&B journals, run IDs and Slurm logs stay intact. New results use the original
experiment IDs; the retry sbatch refuses to start if that result directory exists.
The new run receives a new W&B run ID even though its display name matches the
original failed run. Every retry uses the original algorithm `config1`, task,
network, environment, seeds and training script. There is no pooled mixer or
CPU replay variant in this retry. A preflight can abort for insufficient GPU
memory without changing any experiment parameter. Run task 1 first and verify
startup, training progress, VRAM and metrics before dispatching tasks 2-15.

The previous 30% run required about 96 hours. Larger runs may exceed seven
days even if they fit in VRAM; verify QoS walltime before submitting the full
array. Exact reruns retain the sequential-AEC credit-assignment and gamma
concerns identified in the analysis report. Those are separate methodological
questions, not permission to change this retry's architecture.

## GPU VRAM Requirements & Administrator Request Template

The first hypernetwork of QMIX mixer scales quadratically ($2880 \times N_{\text{CAV}}^2$ parameters):
- **414 CAVs (40%):** Requires $\ge 9.19$ GiB VRAM just for weights, targets, gradients and Adam moments.
- **1,035 CAVs (100%):** Requires $\ge 57.46$ GiB VRAM minimum before replay buffer and activations.

If `gpu_probe.sbatch` reveals that `rknodes` GPUs lack sufficient VRAM ($\ge 80$ GB), contact cluster administrators (`piotr.kubaty@doctoral.uj.edu.pl` / Slack `#servers`) using the template below:

```text
Subject: Dostęp do węzła GPU o wysokiej pamięci (>=80GB) dla eksperymentów QMIX URB

Dzień dobry,
Potrzebuję przeprowadzić serię eksperymentów wieloagentowych QMIX w ramach benchmarku URB 
dla flot od 414 do 1035 pojazdów CAV na mapie Ingolstadt. 

Dotychczasowe zadania na RTX 2080 Ti (10.58 GiB VRAM) kończą się błędem CUDA Out of Memory:
- Przy 414 CAV (40%) błąd występuje podczas inicjalizacji optymalizatora Adam (~9.2 GiB).
- Przy 1035 CAV (100%) sama pierwsza macierz hiper-sieci miksera wraz ze stanami treningowymi 
  wymaga co najmniej 57.5 GiB VRAM (przed alokacją bufora powtórek i aktywacji).

Posiadam dostęp do partycji rknodes, jednak potrzebuję potwierdzić:
1. Jakie modele GPU i z jaką pamięcią VRAM są dostępne na rknodes?
2. Czy istnieje możliwość uruchomienia zadań na węzłach z kartami A100 (40/80GB) lub H100 (80GB) 
   (np. partycje dgxa100 / dgxh100) z limitem czasu umożliwiającym ukończenie treningu (poprzedni 
   run na 30% trwał ok. 96h)?

Zależy mi na zachowaniu oryginalnych hiperparametrów eksperymentu bez modyfikacji architektury.
Z góry dziękuję za pomoc!
```
