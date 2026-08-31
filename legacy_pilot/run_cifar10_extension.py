"""CIFAR-10 pilot-scale extension of the MNIST protocol -- the core H1 stress test.

PILOT-SCALE NOTICE (see results_summary.md): reduced further than the MNIST pilot,
because CIFAR-10 SmallCNN forward/backward passes are markedly more expensive per
client-step on CPU-only hardware than the MNIST CNN. Reductions are recorded in
`PILOT_REDUCTIONS` below and mirrored into results/cifar10/run_manifest.json.

There is no prior "OTCD" SmallCNN or cached CIFAR-10 data on this machine (verified by
search before starting this build) -- the SmallCNN in src/models.py and the CIFAR-10
data pipeline in src/data.py were both built fresh for this study.

Structure:
  Stage A (scripts/hp_probe_cifar10.py, run separately): small clean-training grid,
    same purpose as the MNIST HP probe -- CIFAR-10 gradients behave very differently
    from MNIST's, so MNIST-tuned hyperparameters are NOT assumed to transfer (this is
    itself part of testing H1).
  Stage C1: main comparative sweep on CIFAR-10 SmallCNN, IID partition -- condition in
    {clean, ipm, label_flip} x epsilon in {8, 18} x seeds, mirroring MNIST Stage B2 for
    a like-for-like comparison.
  Stage C2: Dirichlet non-IID extension (alpha=0.5) -- NOT part of the paper's own
    protocol; added here as an explicitly-labeled extension, at condition in
    {clean, ipm} x epsilon={8} x seeds (reduced further -- this is an extension, not
    part of either the paper's or this study's primary replication target).
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data import load_cifar10, partition_dirichlet
from models import SmallCNN
from federated_experiment import run_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "cifar10")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Winning hyperparameters from scripts/hp_probe_cifar10.py (results/cifar10/hp_probe.json) ---
# Stage A results (T=60, clean, SmallCNN, beta=0.1, beta_hat=0.1) -- NOTE all six are far
# weaker than the identical grid on MNIST (best MNIST=0.5886 vs best CIFAR-10=0.1435),
# i.e. MNIST-tuned hyperparameters do NOT transfer -- see results_summary.md H1 discussion:
#   gamma=1.0,  tau=1.0  -> 0.1000 (chance level)
#   gamma=0.1,  tau=1.0  -> 0.1435  <-- winner (still far from MNIST-level accuracy)
#   gamma=0.1,  tau=0.1  -> 0.1331
#   gamma=0.01, tau=0.1  -> 0.1092 (near chance -- did not work)
#   gamma=0.1,  tau=0.01 -> 0.1310
#   gamma=1.0,  tau=0.1  -> 0.0851 (below chance-ish, unstable -- did not work)
BEST_GAMMA = 0.1
BEST_TAU = 1.0
BEST_BETA_HAT = 0.1

N_REGULAR = 20
N_BYZANTINE = 4
T_MAIN = 80
BATCH_SIZE = 32
SEEDS_MAIN = [0, 1, 2]
SEEDS_SECONDARY = [0, 1]
DIRICHLET_ALPHA = 0.5

PILOT_REDUCTIONS = {
    "seeds": "n=3 for the main IID sweep (n=2 for the Dirichlet non-IID extension) vs. "
             "the task spec's requested n=10-20.",
    "hyperparameter_grid": "same reduced gamma x tau probe as MNIST, re-run from scratch "
                            "on CIFAR-10 (MNIST-winning hyperparameters explicitly NOT "
                            "assumed to transfer -- see results_summary.md H1 discussion).",
    "epsilon_grid": "{8, 18} only for the main sweep; {8} only for the non-IID extension.",
    "byzantine_count": "fixed at 4-of-24 (delta_byz ~ 0.2), matching the MNIST pilot.",
    "communication_rounds_T": "80 rounds, matching the MNIST pilot for a like-for-like "
                               "comparison (not a full-convergence run).",
    "data_scope": "full CIFAR-10 train/test sets used for partitioning (50k/10k images), "
                  "but see results_summary.md for actual wall-clock/feasibility notes.",
}


def save_run(stage, name, result):
    path = os.path.join(RESULTS_DIR, f"{stage}__{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def run_stage_c1_main_sweep_iid(train, test):
    print("== Stage C1: main sweep on CIFAR-10 SmallCNN, IID (condition x epsilon x seed) ==")
    out = []
    conditions = [
        ("clean", None, 0),
        ("ipm", "ipm", N_BYZANTINE),
        ("label_flip", "label_flip", N_BYZANTINE),
    ]
    for cond_name, attack_type, n_byz in conditions:
        for epsilon in [8, 18]:
            for seed in SEEDS_MAIN:
                t0 = time.time()
                res = run_experiment(
                    model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                    n_regular=N_REGULAR, n_byzantine=n_byz, attack_type=attack_type,
                    beta=0.1, beta_hat=BEST_BETA_HAT, gamma=BEST_GAMMA, tau=BEST_TAU,
                    epsilon=epsilon, ragg_name="trimmed_mean", T=T_MAIN, batch_size=BATCH_SIZE,
                    seed=seed,
                )
                elapsed = time.time() - t0
                name = f"{cond_name}_eps{epsilon}_seed{seed}"
                path = save_run("C1_main_iid", name, res)
                print(f"  {name}: final_acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)")
                out.append({"condition": cond_name, "epsilon": epsilon, "seed": seed,
                            "final_acc": res["final_test_acc"], "diverged": res["diverged"],
                            "sigma_omega": res["sigma_omega"], "path": path})
    return out


def run_stage_c2_dirichlet_extension(train, test):
    print("== Stage C2: Dirichlet non-IID extension (alpha=%.2f) ==" % DIRICHLET_ALPHA)
    out = []
    conditions = [("clean", None, 0), ("ipm", "ipm", N_BYZANTINE)]
    for cond_name, attack_type, n_byz in conditions:
        for seed in SEEDS_SECONDARY:
            t0 = time.time()
            res = run_experiment(
                model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                n_regular=N_REGULAR, n_byzantine=n_byz, attack_type=attack_type,
                beta=0.1, beta_hat=BEST_BETA_HAT, gamma=BEST_GAMMA, tau=BEST_TAU, epsilon=8,
                ragg_name="trimmed_mean", T=T_MAIN, batch_size=BATCH_SIZE, seed=seed,
                partition_fn=partition_dirichlet, partition_kwargs={"alpha": DIRICHLET_ALPHA},
            )
            elapsed = time.time() - t0
            name = f"{cond_name}_eps8_seed{seed}"
            path = save_run("C2_dirichlet", name, res)
            print(f"  {name}: final_acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)")
            out.append({"condition": cond_name, "seed": seed, "final_acc": res["final_test_acc"],
                        "diverged": res["diverged"], "path": path})
    return out


def main():
    assert BEST_GAMMA is not None and BEST_TAU is not None, \
        "Fill in BEST_GAMMA / BEST_TAU from results/cifar10/hp_probe.json before running."

    train, test = load_cifar10(DATA_ROOT)

    manifest = {"pilot_reductions": PILOT_REDUCTIONS,
                "best_gamma": BEST_GAMMA, "best_tau": BEST_TAU, "best_beta_hat": BEST_BETA_HAT,
                "dirichlet_alpha": DIRICHLET_ALPHA}

    manifest["stage_c1_main_sweep_iid"] = run_stage_c1_main_sweep_iid(train, test)
    manifest["stage_c2_dirichlet_extension"] = run_stage_c2_dirichlet_extension(train, test)

    with open(os.path.join(RESULTS_DIR, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nDone. Manifest written to", os.path.join(RESULTS_DIR, "run_manifest.json"))


if __name__ == "__main__":
    main()
