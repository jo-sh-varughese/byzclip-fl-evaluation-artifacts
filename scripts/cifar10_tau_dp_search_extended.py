"""Extends cifar10_tau_dp_search.py's Phase 1 search down to the source
paper's own smaller tau values (1e-4, 1e-5, 1e-6) -- the original Part C grid
stopped at 0.001 and did not probe this range, disclosed as a limitation.
Reuses the same manifest (results/cifar10/C4_tau_search/run_manifest.json),
same protocol, same gamma=0.1. If a new tau wins for some epsilon (beats the
original grid's winner), Phase 2 (n=5, all 3 attack conditions) is re-run at
that epsilon with the new winning tau.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_cifar10
from models import SmallCNN
from federated_experiment import run_experiment, compute_sigma_omega

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cifar10", "C4_tau_search")
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_GAMMA = 0.1
BEST_BETA_HAT = 0.1
N_REGULAR = 20
N_BYZANTINE = 4
T_MAIN = 80
BATCH_SIZE = 32
DELTA = 1e-5

EPSILON_GRID = [3, 8, 13, 18, 23]
EXTRA_TAU_GRID = [1e-4, 1e-5, 1e-6]
N_CONFIRM_SEEDS = 5


def save_run(name, result):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def run_phase1_extended(train, test):
    print("== Phase 1 EXTENDED: tau x epsilon search, tau in {1e-4,1e-5,1e-6} ==", flush=True)
    out = []
    for epsilon in EPSILON_GRID:
        for tau in EXTRA_TAU_GRID:
            name = f"phase1_clean_eps{epsilon}_tau{tau}"
            path = os.path.join(RESULTS_DIR, f"{name}.json")
            if os.path.exists(path):
                with open(path) as f:
                    res = json.load(f)
            else:
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
                save_run(name, res)
                print(f"  eps={epsilon} tau={tau} sigma_omega={sigma:.5f}: "
                      f"final_acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)",
                      flush=True)
            out.append({"epsilon": epsilon, "tau": tau, "final_acc": res["final_test_acc"],
                        "diverged": res["diverged"]})
    return out


def run_phase2_confirm(train, test, epsilon, tau):
    print(f"\n== Phase 2 EXTENDED confirm: eps={epsilon}, new winning tau={tau} ==", flush=True)
    conditions = [("clean", None, 0), ("ipm", "ipm", N_BYZANTINE), ("label_flip", "label_flip", N_BYZANTINE)]
    for cond_name, attack_type, n_byz in conditions:
        for seed in range(N_CONFIRM_SEEDS):
            name = f"phase2_{cond_name}_eps{epsilon}_tau{tau}_seed{seed}"
            path = os.path.join(RESULTS_DIR, f"{name}.json")
            if os.path.exists(path):
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
            save_run(name, res)
            print(f"  {name}: final_acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)",
                  flush=True)


def main():
    train, test = load_cifar10(DATA_ROOT)
    manifest_path = os.path.join(RESULTS_DIR, "run_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    old_winners = {int(k): v for k, v in manifest["winners"].items()}

    extended = run_phase1_extended(train, test)
    manifest["phase1_extended"] = extended

    new_winners = dict(old_winners)
    changed = {}
    for r in extended:
        eps = r["epsilon"]
        if not r["diverged"] and r["final_acc"] > new_winners[eps]["final_acc"]:
            print(f"  NEW WINNER at eps={eps}: tau={r['tau']} "
                  f"(acc={r['final_acc']:.4f} > old {new_winners[eps]['final_acc']:.4f})", flush=True)
            new_winners[eps] = r
            changed[eps] = r

    manifest["winners_extended"] = {str(k): v for k, v in new_winners.items()}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if not changed:
        print("\nNo new tau value beat the original grid's winner at any epsilon; "
              "the plateau finding is confirmed down to tau=1e-6.", flush=True)
    else:
        for eps, winner in changed.items():
            run_phase2_confirm(train, test, eps, winner["tau"])
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print("\nExtended tau search done.", flush=True)


if __name__ == "__main__":
    main()
