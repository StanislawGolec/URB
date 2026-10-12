# Analysis of the 10 + 16 experiments

Analysis date: 2026-10-12. Source checkout: 7e43300.

> **Full Detailed Analysis & Roadmap:** See [docs/EXPERIMENT_ANALYSIS_AND_ROADMAP.md](../../../docs/EXPERIMENT_ANALYSIS_AND_ROADMAP.md) for in-depth derivations, code line references, VRAM scaling physics, Slurm configurations, and the collaborator guide.

11/26 runs completed; 15 failed with CUDA out of memory. The 11 benchmark CSVs agree with all 21 corresponding W&B summary metrics. Original journal exit codes and completed testing progress corroborate completion. Raw episode CSVs are absent, so per-day coverage, uncertainty and corrected cumulative learning costs cannot be independently recomputed.

## Completed results

Travel times are minutes. Positive changes mean worse travel time. System changes use each run's own pre-mutation baseline (4.847303 min). Human changes use the baseline of the agents who remained human.

| Experiment | System | CAV | Human | System change | Human change |
|---|---:|---:|---:|---:|---:|
| QMIX 10% CAV | 4.986898 | 6.406214 | 4.830042 | +2.880% | -0.213% |
| QMIX 20% CAV | 5.191101 | 6.381990 | 4.893379 | +7.093% | +0.396% |
| QMIX 30% CAV | 5.345271 | 6.411873 | 4.889206 | +10.273% | +0.226% |
| IQL individual 100% | 4.796552 | 4.696978 | 4.862935 | -1.047% | +0.120% |
| IQL individual 85% | 4.821525 | 4.730052 | 4.882506 | -0.532% | +0.523% |
| IQL individual 70% | 4.805529 | 4.714790 | 4.866022 | -0.862% | +0.183% |
| IQL individual 50% | 4.831473 | 4.763914 | 4.876513 | -0.327% | +0.399% |
| IQL individual 30% | 4.840578 | 4.800409 | 4.867357 | -0.139% | +0.211% |
| IQL individual 15% | 4.907085 | 4.939895 | 4.885211 | +1.233% | +0.579% |
| IQL individual 5% | 4.982276 | 5.103644 | 4.901364 | +2.784% | +0.911% |
| IQL individual 0% | 5.100479 | 5.434494 | 4.877802 | +5.223% | +0.426% |

## Interpretation

The best observed IQL result is individual weight 100% (4.796552 min, 1.047% below baseline, about 3.05 seconds per trip). The 70% individual variant is close (4.805529 min); the gap is only 0.54 seconds per trip. One seed cannot establish that the ranking is robust. All eight IQL treatments slightly increase the travel time of the remaining humans. The best system result therefore combines a CAV benefit with a small human cost, rather than a benefit for both groups.

With 15%, 5% or 0% individual reward, IQL is worse than the all-human baseline. The pure all-population reward produces 5.100479 min (+5.223%). Increasing the social weight did not improve system performance in these trained policies. A plausible mechanism is diluted individual credit from the shared reward; these data do not isolate that mechanism from optimization and exploration effects.

Completed altruistic QMIX runs worsen system travel time by 2.880%, 7.093%, and 10.273% at 10%, 20%, and 30% CAVs. CAV travel times remain around 6.4 min. The saved 30% travel-time plot rises during training and then plateaus at poor performance. Training completion is not evidence of successful learning.

All eight QMIX reward-mixture runs failed before a logged training iteration, so there is no completed, matched-penetration QMIX/IQL comparison. The equivalent QMIX altruistic-40 and individual-0/all-100 runs both failed, preventing the intended reproducibility comparison.

## Training diagnostics

| QMIX CAV share | Mean loss, first 20 iterations | Mean loss, last 20 | Mean reward, first 20 | Mean reward, last 20 | Runtime (h) |
|---|---:|---:|---:|---:|---:|
| 10% | 1.684e+34 | 3.798e+33 | -0.047668 | -0.048515 | 42.51 |
| 20% | 8.063e+33 | 5.103e+33 | -0.024140 | -0.025105 | 59.27 |
| 30% | 3.888e+32 | 4.810e+32 | -0.016354 | -0.017310 | 95.68 |

All 600 QMIX loss values are finite, but are approximately 10^32–10^34. These extreme TD losses require investigating value scale, observations, masks and gradients. The logged reward worsens within every completed QMIX run. Across penetration levels its magnitude is also diluted by the number of sequential CAV decisions; do not compare the uncorrected curves directly. IQL journals contain 4,000 episode/epsilon records each, but no numeric loss or travel-time history. Its saved plots support qualitative inspection only.

## Failure and runtime evidence

All 15 failed runs report CUDA OOM on an RTX 2080 Ti with about 10.58 GiB available device capacity. The eight mixed QMIX runs and altruistic-40 fail while Adam initializes optimizer state (a further 1.84 GiB allocation). Altruistic-100 fails creating the first mixer hypernetwork (11.49 GiB allocation). More walltime or host RAM does not fix this VRAM requirement.

The mixer state contains all CAV observations. With 15 features and embedding size 192, its first weight matrix contains 2880*N_CAV^2 parameters. At 100% CAVs one FP32 copy needs about 12.34 GB (11.49 GiB). Online/target weights, gradients and Adam states multiply this requirement. Validate the GPU capacity on bonk before rerunning; access to a new partition alone does not establish sufficient GPU memory.

The journals record RouteRL 2.0.0, Torch 2.6.0, TorchRL 0.7.0 and W&B 0.29.0. Run commits are recorded individually in manifest.json. Altruistic-10/20 use babe8cb, the remaining penetration runs fece3e2, and reward-mixture runs 2ebca12.

## Methodological limits and code findings

- Only seed 42 is represented. Testing days are not independent policy replications; no seed-level confidence intervals or significance claims are justified.
- IQL records the environment config as trip_info_eta but the executed script hardcodes previous_agents_plus_start_time. QMIX uses ETA plus an OD embedding. The information given to the algorithms differs.
- QMIX evaluates at epsilon 0.1; IQL evaluates greedily. Match evaluation policy before comparing algorithms.
- QMIX mixes all proposed actions despite executing only the active AEC agent's action. Validate inactive-agent masking and credit assignment with a small deterministic environment.
- A concrete numerical lead is invalid-route ETA padding: the local RouteRL dependency fills missing cluster free-flow values with 1e9 (environment.py:863), and TripInfoWithETA feeds these values into observations. QMIX passes those observations to its mixer without normalization. This can overwhelm the hypernetworks even when action selection is masked. The archived server package version is known, but its source/observations are not included, so this is a hypothesis to reproduce, not a confirmed explanation of the enormous losses.
- Gamma 0.95 is applied per sequential decision. Terminal reward discount to the first decision is 0.00534 at 103 CAVs and 6.31e-10 at 414 CAVs. The exact runtime environment semantics should be reproduced before altering the algorithm.
- cost_of_learning is a max-minus-min range, not cumulative baseline-relative cost. It is preserved as an original field in summary.csv but excluded from conclusions.
- avg_time_lost and its subgroup columns use inconsistent aggregation. Effect_of_change uses the remaining-human baseline, rather than the same cohort before it became CAVs. Derived CAV baseline values in summary.csv use population-weighted reconstruction, assuming consistent agent coverage.
- The 100% CAV metric path suppresses a valid human-learning baseline. It did not affect these completed metrics because that run failed before evaluation.
- The saved raw episode files, optimizer state and model checkpoints are absent. Exact recalculation and greedy reevaluation cannot be performed from this checkout.

## Recommended next experiments

1. Resolve the QMIX numerical scale and AEC formulation on a tiny deterministic case before a full sweep.
2. Match observations and evaluation epsilon, save checkpoints, terminal rewards, gradient norms, and episode-level travel times.
3. Compare IQL individual weights 100%, 70% and 50%, plus pure collective control, on at least five matched seeds. The present 100%/70% gap is too small for a robust ranking.
4. Verify bonk GPU VRAM, partition/QoS association and runtime limits; select a tractable mixer configuration or explicitly label any architecture change as a new experiment.
5. Preserve the original failed runs in W&B. Failure is part of the experiment outcome.

## Reproduction and provenance

Run `python analysis/analyze_new_sweeps.py --charts` with wandb and reportlab installed. The script leaves original CSVs, journals and plots untouched. summary.csv contains all 26 statuses, original metrics, derived comparisons and source paths. training_history.csv contains the numerical training records. manifest.json identifies the 26 original W&B run IDs, journal hashes, dependency versions and plot hashes. Run `python analysis/sync_new_sweeps.py` to preview synchronization, then add `--execute` on a machine with W&B network access and an existing login. The destination defaults to the entity/project recorded in the original journals. A local manifest is not evidence of a successful cloud upload.

Current upload status (2026-10-12): an authenticated API connection attempt was blocked by session network permissions (WinError 10013). No remote upload was verified in this session. See upload_status.json; execute the prepared synchronizer from a network-enabled environment.
