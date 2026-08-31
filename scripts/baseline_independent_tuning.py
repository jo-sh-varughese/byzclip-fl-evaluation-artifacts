"""Independently tunes (gamma, tau) for each external baseline (Byz-Clip-SGD,
Safe-DSHB), matching the source paper's own tuning range (Section 6 of
islamov2026byzclip: lr in {10,1,1e-1,1e-2,1e-3}, tau in
{1,1e-1,1e-2,1e-4,1e-5,1e-6}), rather than reusing Byz-Clip21-SGD2M's own
tuned (gamma=0.1, tau=1.0).

Disclosed simplification vs. the source paper: we tune once on the clean,
no-DP, no-Byzantine MNIST/CNN condition (T=60, seed=0), the same
single-seed-clean-condition-probe pattern this project's own Stage-A probe
uses for Byz-Clip21-SGD2M, then apply each baseline's winning (gamma, tau)
across the full replication grid (clean/ipm/label_flip x eps8/eps18, n=10).
The source paper tunes under DP noise directly and "allocates an equal amount
of privacy budget for tuning" per algorithm; reproducing that exactly would
multiply this already-large grid by 5 epsilons per baseline, which is out of
budget for this pass. This is a disclosed, partial improvement over reusing
Byz-Clip21-SGD2M's hyperparameters outright, not a full replication of the
source paper's tuning protocol.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_mnist
from models import MNIST_CNN
from federated_experiment import run_baseline_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "mnist", "external_baselines")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALGOS = ["byz_clip_sgd", "safe_dshb"]
GAMMAS = [10, 1, 0.1, 0.01, 0.001]
TAUS = [1, 0.1, 0.01, 1e-4, 1e-5, 1e-6]
BETA = 0.1
N_REGULAR = 20
T_PROBE = 60


def save_run(name, result):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def already_done(name):
    return os.path.exists(os.path.join(RESULTS_DIR, f"{name}.json"))


def run_tuning_grid(train, test):
    winners = {}
    for algo in ALGOS:
        best = None
        results = []
        for gamma in GAMMAS:
            for tau in TAUS:
                name = f"TUNE_{algo}__gamma{gamma}_tau{tau}"
                cache_path = os.path.join(RESULTS_DIR, f"{name}.json")
                if os.path.exists(cache_path):
                    with open(cache_path) as f:
                        res = json.load(f)
                else:
                    t0 = time.time()
                    res = run_baseline_experiment(
                        algo_name=algo, model_ctor=MNIST_CNN, train_dataset=train, test_dataset=test,
                        n_regular=N_REGULAR, n_byzantine=0, attack_type=None,
                        gamma=gamma, tau=tau, epsilon=None, ragg_name="trimmed_mean",
                        T=T_PROBE, batch_size=32, seed=0, beta=BETA, eval_every=20,
                    )
                    elapsed = time.time() - t0
                    save_run(name, res)
                    print(f"  [TUNE/{algo}] gamma={gamma} tau={tau}: "
                          f"final_acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)",
                          flush=True)
                results.append({"gamma": gamma, "tau": tau, "final_acc": res["final_test_acc"],
                                 "diverged": res["diverged"]})
                if not res["diverged"] and (best is None or res["final_test_acc"] > best["final_acc"]):
                    best = {"gamma": gamma, "tau": tau, "final_acc": res["final_test_acc"]}
        winners[algo] = best
        print(f"== {algo} winner: gamma={best['gamma']} tau={best['tau']} "
              f"(final_acc={best['final_acc']:.4f}) ==", flush=True)

    with open(os.path.join(RESULTS_DIR, "TUNE_winners.json"), "w") as f:
        json.dump(winners, f, indent=2)
    return winners


if __name__ == "__main__":
    print("Loading MNIST...", flush=True)
    train, test = load_mnist(DATA_ROOT)
    t0 = time.time()
    winners = run_tuning_grid(train, test)
    print(f"\nTuning grid done in {(time.time()-t0)/3600:.2f} hours. Winners: {winners}", flush=True)
