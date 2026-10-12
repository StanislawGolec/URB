"""Conservative lower bound on QMIX GPU memory, before costly human warm-up."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def minimum_mixer_bytes(n_agents, observation_dim, mixing_dim):
    # One hypernetwork's matrix only: online + target + gradient + two Adam moments.
    return n_agents * observation_dim * mixing_dim * n_agents * 4 * 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", required=True)
    parser.add_argument("--alg-conf", default="config1")
    parser.add_argument("--env-conf", default="clusters_eta")
    parser.add_argument("--task-conf", required=True)
    args, _ = parser.parse_known_args()
    algo = json.loads((ROOT / "config/algo_config/qmix_torchrl_clusters" / f"{args.alg_conf}.json").read_text())
    env = json.loads((ROOT / "config/env_config" / f"{args.env_conf}.json").read_text())
    task = json.loads((ROOT / "config/task_config" / f"{args.task_conf}.json").read_text())
    if env.get("observations") != "trip_info_eta":
        print("QMIX memory estimate unavailable for this observation encoder; no capacity claim made.")
        return
    with (ROOT / "networks" / args.net / "agents.csv").open() as stream:
        travelers = sum(1 for _ in csv.DictReader(stream))
    n_agents = int(travelers * task["ratio_machines"])
    # The clustered loader determines path count, which may override the env config.
    n_paths = env["number_of_paths"]
    if env.get("use_clustered_routes"):
        from clustered_routes import resolve_route_set
        route_parser = argparse.ArgumentParser(add_help=False)
        route_parser.add_argument("--route-set")
        route_args, _ = route_parser.parse_known_args()
        route_set = resolve_route_set(args.net, route_args.route_set)
        representatives = ROOT / "networks" / args.net / "clustered_routes" / route_set / f"{args.net}_clusters_representants.csv"
        with representatives.open() as stream:
            n_paths = len({row["cluster"] for row in csv.DictReader(stream)})
    observation_dim = n_paths + 3 + algo.get("od_embedding_dim", 8)
    lower_bound = minimum_mixer_bytes(n_agents, observation_dim, algo["mixing_embed_dim"])
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; Slurm requested a GPU but PyTorch cannot use it.")
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    print(f"QMIX: {n_agents} CAVs; one hypernetwork and training states alone need >= {lower_bound/gib:.2f} GiB; "
          f"GPU has {free/gib:.2f} GiB free / {total/gib:.2f} GiB total.")
    reserve = max(2 * gib, 0.25 * lower_bound)
    print(f"Preflight reserve for other allocations: {reserve/gib:.2f} GiB.")
    if lower_bound + reserve >= free:
        raise SystemExit("Insufficient GPU memory for the original model and a minimum reserve. Request a higher-VRAM GPU; the algorithm config is unchanged.")
    print("Passing this conservative screen does not guarantee the full run fits.")


if __name__ == "__main__":
    main()
