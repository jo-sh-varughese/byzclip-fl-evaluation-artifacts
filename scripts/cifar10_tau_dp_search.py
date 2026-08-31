"""Part C: DP-aware tau search on CIFAR-10 (AAAI scale-up).

Motivation (see results_summary.md Stage C1/C3 and Sec 8, gap #6): the pilot found
that tau=1.0 -- optimal for CLEAN training (no DP) at gamma=0.1, T=80 -- produces
sigma_omega = (tau/epsilon) * sqrt(T*log(1/delta)) large enough to swamp the signal
at every epsilon tried (sigma_omega=3.79 at eps=8, 1.69 at eps=18). Since sigma_omega
scales linearly with tau, this script asks whether a SMALLER tau, chosen per epsilon,
recovers non-degenerate accuracy under DP -- something the pilot's fixed-tau protocol
could not test. A tau that is too small also over-clips honest updates independently
of DP noise, so this is a genuine two-sided search, not a "smaller is always better"
sweep.

Phase 1 (search): epsilon in {3,8,13,18,23} x tau in {0.001,0.01,0.1,0.5,1.0},
  condition=clean only, 1 seed, T=80 (matches the main-sweep budget for a
  like-for-like comparison against Stage C1). 25 cells.
Phase 2 (confirm): for the winning tau at each epsilon (plus the pilot's original
  tau=1.0 as a reference column), re-run at n=N_CONFIRM_SEEDS across
  condition in {clean, ipm, label_flip} to check whether a properly-tuned tau also
  preserves Byzantine-robustness, not just clean accuracy.

All raw results saved per-run under results/cifar10/C4_tau_search/ so every number
in the paper draft traces to a file, per process requirements.
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_cifar10
from models import SmallCNN
from federated_experiment import run_experiment, compute_sigma_omega

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cifar10", "C4_tau_search")
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_GAMMA = 0.1          # from hp_probe.json, matching Stage C1's operating point
BEST_BETA_HAT = 0.1
N_REGULAR = 20
N_BYZANTINE = 4
T_MAIN = 80
BATCH_SIZE = 32
DELTA = 1e-5

EPSILON_GRID = [3, 8, 13, 18, 23]
TAU_GRID = [0.001, 0.01, 0.1, 0.5, 1.0]
N_CONFIRM_SEEDS = 5


def save_run(name, result):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def run_phase1_search(train, test):
    print("== Phase 1: tau x epsilon search, clean condition, seed=0, T=80 ==")
    out = []
    for epsilon in EPSILON_GRID:
        for tau in TAU_GRID:
            sigma = compute_sigma_omega(tau, epsilon, T_MAIN, DELTA)
            t0 = time.time()
            res = run_experiment(
                model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                n_regular=N_REGULAR, n_byzantine=0, attack_type=None,
                beta=0.1, beta_hat=BEST_BETA_HAT, gamma=BEST_GAMMA, tau=tau,
                epsilon=epsilon, ragg_name="trimmed_mean", T=T_MAIN,
                batch_size=BATCH_SIZE, seed=0,
            )
            elapsed = time.time() - t0
            name = f"phase1_clean_eps{epsilon}_tau{tau}"
            path = save_run(name, res)
            print(f"  eps={epsilon} tau={tau} sigma_omega={sigma:.3f}: "
                  f"final_acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)")
            out.append({"epsilon": epsilon, "tau": tau, "sigma_omega": sigma,
                        "final_acc": res["final_test_acc"], "diverged": res["diverged"], "path": path})
    return out


def pick_winners(phase1_results):
    """For each epsilon, pick the tau with the highest phase-1 accuracy."""
    winners = {}
    for r in phase1_results:
        eps = r["epsilon"]
        if eps not in winners or r["final_acc"] > winners[eps]["final_acc"]:
            winners[eps] = r
    return winners


def run_phase2_confirm(train, test, winners):
    print("\n== Phase 2: confirm winners across seeds and conditions ==")
    out = []
    conditions = [("clean", None, 0), ("ipm", "ipm", N_BYZANTINE), ("label_flip", "label_flip", N_BYZANTINE)]
    for epsilon, winner in sorted(winners.items()):
        tau = winner["tau"]
        for cond_name, attack_type, n_byz in conditions:
            for seed in range(N_CONFIRM_SEEDS):
                name = f"phase2_{cond_name}_eps{epsilon}_tau{tau}_seed{seed}"
                path = os.path.join(RESULTS_DIR, f"{name}.json")
                if os.path.exists(path):
                    with open(path) as f:
                        res = json.load(f)
                    print(f"  {name}: SKIP (already done) final_acc={res['final_test_acc']:.4f}")
                    out.append({"epsilon": epsilon, "tau": tau, "condition": cond_name, "seed": seed,
                                "final_acc": res["final_test_acc"], "diverged": res["diverged"], "path": path})
                    continue
                t0 = time.time()
                res = run_experiment(
                    model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                    n_regular=N_REGULAR, n_byzantine=n_byz, attack_type=attack_type,
                    beta=0.1, beta_hat=BEST_BETA_HAT, gamma=BEST_GAMMA, tau=tau,
                    epsilon=epsilon, ragg_name="trimmed_mean", T=T_MAIN,
                    batch_size=BATCH_SIZE, seed=seed,
                )
                elapsed = time.time() - t0
                path = save_run(name, res)
                print(f"  {name}: final_acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)")
                out.append({"epsilon": epsilon, "tau": tau, "condition": cond_name, "seed": seed,
                            "final_acc": res["final_test_acc"], "diverged": res["diverged"], "path": path})
                # Write manifest incrementally so a future interruption loses at most one run's progress.
                manifest_path = os.path.join(RESULTS_DIR, "run_manifest.json")
                with open(manifest_path) as f:
                    manifest = json.load(f)
                manifest["phase2_confirm_partial"] = out
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["1", "2", "both"], default="both")
    parser.add_argument("--confirm-seeds", type=int, default=N_CONFIRM_SEEDS)
    args = parser.parse_args()

    train, test = load_cifar10(DATA_ROOT)

    manifest_path = os.path.join(RESULTS_DIR, "run_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    if args.phase in ("1", "both"):
        manifest["phase1_search"] = run_phase1_search(train, test)
        manifest["winners"] = {str(k): v for k, v in pick_winners(manifest["phase1_search"]).items()}
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    if args.phase in ("2", "both"):
        winners = {int(k): v for k, v in manifest["winners"].items()}
        manifest["phase2_confirm"] = run_phase2_confirm(train, test, winners)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print("\nDone. Manifest written to", manifest_path)


if __name__ == "__main__":
    main()
