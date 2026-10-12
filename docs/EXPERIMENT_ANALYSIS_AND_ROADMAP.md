# Comprehensive Experiment Analysis, Critical Findings & Research Roadmap

> **Author / Origin:** Compiled from experimental sweeps, codebase analysis, and deep-dive technical evaluations conducted across the `URB` benchmark.  
> **Target Audience:** Collaborators, researchers, and developers continuing the Connected Autonomous Vehicle (CAV) routing benchmark suite.  
> **Source Commit:** `7e43300` on branch `issue-40-and-44`  
> **Date:** October 2026  

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Experiment Suite Overview (10 + 16 Runs)](#2-experiment-suite-overview-10--16-runs)
3. [Empirical Results & Comparative Findings](#3-empirical-results--comparative-findings)
4. [Six Critical Methodological & Algorithmic Flaws](#4-six-critical-methodological--algorithmic-flaws)
   - [4.1 Asymmetric Information & Evaluation (QMIX vs IQL)](#41-asymmetric-information--evaluation-qmix-vs-iql)
   - [4.2 QMIX Sequential AEC Credit Assignment Failure](#42-qmix-sequential-aec-credit-assignment-failure)
   - [4.3 QMIX Reward Logging Artifact](#43-qmix-reward-logging-artifact)
   - [4.4 Unnormalized Observations & $10^9$ ETA Padding](#44-unnormalized-observations--109-eta-padding)
   - [4.5 Metric Calculation Inconsistencies in `metrics.py`](#45-metric-calculation-inconsistencies-in-metricspy)
   - [4.6 Reward Mixture Nuances & Population Weighting](#46-reward-mixture-nuances--population-weighting)
5. [Hardware Scaling Bottleneck & VRAM Physics](#5-hardware-scaling-bottleneck--vram-physics)
6. [Cluster Operations & Retry Pipeline (GMUM / Bonk)](#6-cluster-operations--retry-pipeline-gmum--bonk)
7. [Weights & Biases (W&B) Synchronization](#7-weights--biases-wb-synchronization)
8. [Concrete Action Items & Next Steps for Collaborators](#8-concrete-action-items--next-steps-for-collaborators)

---

## 1. Executive Summary

This document synthesizes the experimental outcomes, architectural diagnostics, and algorithmic investigations for the **10 + 16 experimental sweep** on the `ingolstadt_custom2` network within the **Urban Routing Benchmark (URB)**.

### High-Level Takeaways:
- **11 of 26 runs completed successfully**, while **15 runs crashed due to CUDA Out of Memory (OOM)** on an RTX 2080 Ti (10.58 GiB VRAM).
- **Independent Q-Learning (IQL)** completed all 8 mixed-reward variants. **100% individual reward ($\alpha=1.0$) achieved the best result** (4.797 min, a **1.05% improvement** over the all-human baseline). Conversely, **purely altruistic IQL ($\alpha=0.0$) severely degraded traffic** (+5.22%), likely due to credit assignment dilution.
- **QMIX** completed only at 10%, 20%, and 30% CAV penetration. It produced severe performance degradation (+2.88%, +7.09%, and +10.27% respectively) and exhibited **numerical divergence with TD losses on the order of $10^{32}$–$10^{34}$**.
- **Crucial Caution:** Current QMIX and IQL runs **cannot be directly compared** because QMIX was evaluated with active exploration ($\epsilon=0.1$) and extra ETA observations, while IQL was evaluated greedily ($\epsilon=0.0$) with route choice counts. Furthermore, QMIX failed all 40% CAV runs.

---

## 2. Experiment Suite Overview (10 + 16 Runs)

Both batches targeted the `ingolstadt_custom2` network under identical timeline settings:
- **300 days** of human-driver warm-up / learning.
- **4,000 days** of CAV training.
- **100 days** of final policy evaluation / testing.
- **Seed 42** (single-seed baseline; 100 test days test a single trained policy, not 100 distinct policies).

| Batch | Algorithm | Target Variable | Fixed Parameters | Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **10 experiments** | QMIX Clusters | CAV penetration: 10% to 100% (step: 10%) | Altruistic reward ($\alpha=0.0$) | **3 completed** (10, 20, 30%); **7 failed (OOM)** (40–100%) |
| **16 experiments** | IQL Clusters (8) & QMIX Clusters (8) | Individual weight $\alpha \in \{1.0, 0.85, 0.70, 0.50, 0.30, 0.15, 0.05, 0.0\}$ | 40% CAV penetration (414 CAVs) | **8 IQL completed**; **8 QMIX failed (OOM)** |

---

## 3. Empirical Results & Comparative Findings

### 3.1 Completed Experiments Table
Baseline pre-mutation travel time: **4.8473 min** (all-human equilibrium).

| Experiment | System Travel Time (min) | CAV Travel Time (min) | Human Travel Time (min) | System Change (%) | Human Change (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **IQL Individual 100% ($\alpha=1.0$)** | **4.7966** | **4.6970** | 4.8629 | **-1.047% (Best)** | +0.120% | Completed |
| **IQL Individual 85% ($\alpha=0.85$)** | 4.8215 | 4.7301 | 4.8825 | -0.532% | +0.523% | Completed |
| **IQL Individual 70% ($\alpha=0.70$)** | 4.8055 | 4.7148 | 4.8660 | -0.862% | +0.183% | Completed |
| **IQL Individual 50% ($\alpha=0.50$)** | 4.8315 | 4.7639 | 4.8765 | -0.327% | +0.399% | Completed |
| **IQL Individual 30% ($\alpha=0.30$)** | 4.8406 | 4.8004 | 4.8674 | -0.139% | +0.211% | Completed |
| **IQL Individual 15% ($\alpha=0.15$)** | 4.9071 | 4.9399 | 4.8852 | +1.233% | +0.579% | Completed |
| **IQL Individual 5% ($\alpha=0.05$)** | 4.9823 | 5.1036 | 4.9014 | +2.784% | +0.911% | Completed |
| **IQL Individual 0% ($\alpha=0.0$)** | 5.1005 | 5.4345 | 4.8778 | +5.223% (Worst) | +0.426% | Completed |
| **QMIX Altruistic 10%** | 4.9869 | 6.4062 | 4.8300 | +2.880% | -0.213% | Completed |
| **QMIX Altruistic 20%** | 5.1911 | 6.3820 | 4.8934 | +7.093% | +0.396% | Completed |
| **QMIX Altruistic 30%** | 5.3453 | 6.4119 | 4.8892 | +10.273% | +0.226% | Completed |
| *QMIX Altruistic 40%–100% (7 runs)* | — | — | — | — | — | **Failed (CUDA OOM)** |
| *QMIX Mixed 40% (8 runs)* | — | — | — | — | — | **Failed (CUDA OOM)** |

### 3.2 Key Behavioral Insights:
1. **Selfish Incentives Align Best with Traffic Efficiency in IQL:**
   As the individual weight drops below 30%, performance collapses. When $\alpha=0.0$ (pure social reward), IQL agents cannot distinguish their personal contribution from the average travel time of 1,035 vehicles, resulting in random or destructive routing choices.
2. **Human Externality:**
   In all completed IQL experiments, human travel time slightly *increased* (by +0.12% to +0.91%). The system improvement at 100% individual weight was achieved purely because CAVs reduced their own travel time significantly (from 4.84 min down to 4.69 min), somewhat at the expense of human drivers.
3. **QMIX Numerical Collapse:**
   QMIX training logs show losses of magnitude $10^{32}$–$10^{34}$. The policies failed to learn, and vehicle travel times remained stuck near 6.4 minutes.

---

## 4. Six Critical Methodological & Algorithmic Flaws

Before drawing scientific conclusions or ranking algorithms, the following structural flaws must be resolved:

### 4.1 Asymmetric Information & Evaluation (QMIX vs IQL)
- **Observation Asymmetry:**
  - `scripts/qmix_torchrl_clusters.py:394`: QMIX uses real-time travel-time estimates (`TripInfoWithETA`) and Origin-Destination (OD) spatial embeddings.
  - `scripts/iql_clusters.py:321`: IQL hardcodes historical route-choice frequency counts plus departure time (`previous_agents_plus_start_time`).
- **Evaluation Exploration Asymmetry:**
  - `scripts/qmix_torchrl_clusters.py:564`: QMIX tests with $\epsilon = 0.1$. In TorchRL, calling `.eval()` **does not disable $\epsilon$-greedy exploration**!
  - `scripts/iql_clusters.py:410`: IQL tests greedily with $\epsilon = 0.0$.
- **Impact:** Any observed performance gap reflects differences in input information and test-time random jitter rather than algorithmic superiority.

### 4.2 QMIX Sequential AEC Credit Assignment Failure
- **The Wrapper Reality:**
  In RouteRL / PettingZoo, agents act sequentially in an Agent-Environment-Cycle (AEC): 1 CAV executes its route choice per transition (`torchrl/envs/libs/pettingzoo.py:758`).
- **The QMIX Loss Flaw:**
  `torchrl/objectives/multiagent/qmixer.py:342` mixes the action values of **all** agents simultaneously, including unexecuted actions. The active agent mask is **not** consumed by the mixer.
- **The Compounding $\gamma$ Decay:**
  Rewards are delivered only at the terminal step of the day. With $\gamma = 0.95$ per sequential transition, the effective discount factor backpropagated to agent $k$ (out of $N$) is:
  $$\gamma_{\text{effective}} = \gamma^{N - k}$$
  - At 10% CAVs ($N = 103$): $\gamma^{103} \approx 0.0053$.
  - At 40% CAVs ($N = 414$): $\gamma^{414} \approx 6.31 \times 10^{-10}$.
  - At 100% CAVs ($N = 1035$): $\gamma^{1035} \approx 10^{-24}$.
  **The earliest CAVs receive virtually zero learning signal.**

### 4.3 QMIX Reward Logging Artifact
- `scripts/qmix_torchrl_clusters.py:547` averages accumulated reward across all sequential steps in the episode.
- Because non-terminal steps have zero reward, this calculation approximately divides the terminal daily reward by the number of CAVs:
  $$\text{Logged Reward} \approx \frac{R_{\text{daily}}}{N_{\text{CAV}}}$$
- As $N_{\text{CAV}}$ increases, a negative daily reward divided by a larger number moves closer to zero. This creates the **false illusion of learning improvement** across penetration sweeps.

### 4.4 Unnormalized Observations & $10^9$ ETA Padding
- In `RouteRL` (`environment.py:863`), missing cluster free-flow values are padded with `1e9`.
- `TripInfoWithETA` directly passes these unnormalized $10^9$ values into QMIX observations.
- When unnormalized $10^9$ floats enter the hypernetworks, logits and gradients explode immediately, directly causing the observed **$10^{32}$–$10^{34}$ TD loss values**.

### 4.5 Metric Calculation Inconsistencies in `metrics.py`
- **`cost_of_learning` Bug (`analysis/metrics.py:587`):**
  Calculates $\max(t) - \min(t)$ across training days for each agent, instead of cumulative excess travel time relative to the pre-mutation baseline $\sum (t - t_{\text{pre}})$. A policy that is consistently harmful reports a learning cost of zero!
- **Baseline Suppression at 100% CAV (`analysis/metrics.py:512`):**
  At 100% penetration, the script suppresses $t_{\text{pre}}$ baseline comparisons even when human baseline data exists.
- **`avg_time_lost` Inconsistency (`analysis/metrics.py:615`):**
  The overall metric sums post-mutation samples, while human and CAV subgroups average samples.

### 4.6 Reward Mixture Nuances & Population Weighting
- The reward function is:
  $$r_i = -\alpha t_i - (1-\alpha)\bar{t}_{\text{all}}$$
- Where $\bar{t}_{\text{all}} = p \cdot \bar{t}_{\text{CAV}} + (1-p) \cdot \bar{t}_{\text{HDV}}$ ($p$ is CAV penetration).
- At $p = 0.40$, a nominal 50/50 mixture ($\alpha = 0.50$) actually weights the CAV population mean by:
  $$0.50 + 0.50 \times 0.40 = 70\% \text{ (CAV weight)}, \quad 30\% \text{ (Human weight)}$$
- **Reproducibility check:** QMIX Altruistic 40% ($\alpha=0$) and QMIX Mixed Ind-0 / All-100 ($\alpha=0$) represent the exact same experimental condition.

---

## 5. Hardware Scaling Bottleneck & VRAM Physics

### Why did 15 runs fail with CUDA OOM?
In QMIX (`config1.json`), the global state is constructed by concatenating all agent observations. With 15 observation features per agent and embedding dimension 192:
$$\text{Weight Matrix Dimension} = 192 \times (15 \times N_{\text{CAV}}) = 2,880 \times N_{\text{CAV}}^2$$

The memory scales **quadratically** with the number of CAVs:

| CAV Share | $N_{\text{CAV}}$ | Hypernetwork 1 Weights | FP32 Weights Only | Minimum Training Footprint (Weights + Target + Grads + Adam) | Result on RTX 2080 Ti (10.58 GiB) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10%** | 103 | 30.5M | 0.12 GiB | ~0.57 GiB | Completed |
| **20%** | 207 | 123.4M | 0.49 GiB | ~2.30 GiB | Completed |
| **30%** | 310 | 276.8M | 1.10 GiB | ~5.16 GiB | Completed |
| **40%** | 414 | 493.5M | 1.96 GiB | **~9.19 GiB** (+ Replay & Activations $\to$ **OOM**) | **Crashed on Adam initialization** |
| **100%** | 1035 | 3.08B | 12.26 GiB | **$\ge 57.46$ GiB** | **Crashed on Tensor allocation** |

> ⚠️ **Key Rule:** Setting Slurm `--mem=32G` only increases the host CPU RAM. It has zero effect on GPU VRAM.

---

## 6. Cluster Operations & Retry Pipeline (GMUM / Bonk)

### 6.1 Cluster Partition Specifics (`rknodes` / node `bonk`)
- In `sinfo`, `gpu:6` denotes **6 physical GPUs** on the node, not 6 GB of memory.
- The `big` QoS on GMUM allows up to 7 days of walltime, but **only accepts batch jobs (`sbatch`)** and rejects interactive `srun` allocations.

### 6.2 Probing the Node Hardware
Before submitting large jobs, verify the GPU model and VRAM using the prepared probe script:
```bash
mkdir -p slurm_requests/container_printouts
sbatch slurm_requests/gpu_probe.sbatch
# Check output once finished:
cat slurm_requests/container_printouts/gpu-probe-<job-id>.out
```

### 6.3 Admin Request Template (Requesting High-VRAM Nodes)
If the probe shows GPUs with $< 80$ GB VRAM, send this inquiry to the cluster caretakers (`piotr.kubaty@doctoral.uj.edu.pl` / Slack `#servers`):

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

### 6.4 The Retry Pipeline
The retry pipeline preserves original failed runs under `results/<exp_id>_failed_oom`:
```bash
# 1. Test access without submitting:
TASKS=1 bash slurm_requests/launch_retry_failed.sh --check-access

# 2. Run only Task 1 (40% CAV altruistic) to verify memory fit:
TASKS=1 bash slurm_requests/launch_retry_failed.sh

# 3. Only if Task 1 succeeds, dispatch the remaining tasks with a concurrency throttle:
TASKS=2-15 ARRAY_LIMIT=2 bash slurm_requests/launch_retry_failed.sh
```

---

## 7. Weights & Biases (W&B) Synchronization

The 26 initial runs were saved offline because `WANDB_API_KEY` was not configured during the cluster execution.

### To sync all 26 runs (including failure statuses and plots):
1. Configure your API key in `~/.bashrc`:
   ```bash
   export WANDB_API_KEY="your_40_character_wandb_api_key"
   ```
2. Activate your environment:
   ```bash
   conda activate URB
   ```
3. Preview and execute the sync:
   ```bash
   python analysis/sync_new_sweeps.py
   python analysis/sync_new_sweeps.py --execute
   ```
4. Verify that `analysis/reports/10_plus_16/sync_receipt.json` records `"verified": true` across all entries.

---

## 8. Concrete Action Items & Next Steps for Collaborators

### Phase 1: Immediate Experimental Hygiene
- [ ] **Probe GPU Hardware:** Run `sbatch slurm_requests/gpu_probe.sbatch` on `rknodes`.
- [ ] **Sync Initial Runs to W&B:** Run `python analysis/sync_new_sweeps.py --execute`.
- [ ] **Fix Observation & Evaluation Parity:**
  - In `scripts/qmix_torchrl_clusters.py`, ensure testing sets $\epsilon = 0.0$ explicitly:
    ```python
    greedy_policy = exploration_module.set_exploration_mode("mode") # or force eps=0.0
    ```
  - Standardize observation features so both IQL and QMIX receive identical inputs.
- [ ] **Fix $10^9$ Value Padding:** Apply normalization or mask out invalid cluster ETA values before passing to neural networks.

### Phase 2: Metric Calculations Fix
- [ ] Correct `cost_of_learning` in `analysis/metrics.py` to calculate cumulative excess over pre-mutation baseline:
  $$\text{Cost} = \sum_{\tau \in \text{train}} (t_\tau - \hat{t}_{\text{pre}})$$
- [ ] Restore pre-mutation baseline tracking for 100% CAV penetration.
- [ ] Unify sample aggregation in `avg_time_lost`.

### Phase 3: Architectural Scaling (If $\ge 80$ GB VRAM is unavailable)
If GPUs with $\ge 80$ GB VRAM cannot be allocated for the 100% CAV runs:
- [ ] **Pooled State QMIX:** Implement mean/max pooling over agent embeddings before feeding the hypernetworks to decouple mixer parameter count from $N_{\text{CAV}}^2$.
- [ ] **Independent Actor-Critic / MAPPO:** Evaluate centralized training with decentralized execution (CTDE) using architectures that do not require quadratic mixing matrices.
- [ ] **Multi-Seed Sweep:** Re-evaluate top-performing configurations ($\alpha \in \{1.0, 0.70, 0.50\}$) across at least 5 random seeds to establish statistical significance.
