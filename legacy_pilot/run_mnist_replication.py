"""MNIST pilot-scale replication of the paper's Byzantine/DP sweep.

PILOT-SCALE NOTICE (see results_summary.md for full justification): this is NOT the
full protocol described in the task spec. Given CPU-only compute in this session, the
grid is deliberately reduced along several axes (fewer hyperparameter combinations,
fewer seeds than the ideal n=10-20, shorter communication-round budget T). Every
reduction is recorded in the `pilot_reductions` block written to
results/mnist/run_manifest.json so it is auditable rather than silently assumed.

Structure:
  Stage A (scripts/hp_probe.py, run separately): small clean-training grid to find a
    hyperparameter setting that actually learns MNIST within the round budget.
  Stage B1: beta in {0.01, 0.05, 0.1} 3-way comparison at the winning (gamma, tau),
    clean training, CNN -- addresses the source paper's internally-contradictory
    description of beta for Byz-Clip21-SGD2M (see task spec / results_summary.md).
  Stage B2: main comparative sweep on CNN -- condition in {clean, ipm, label_flip}
    (byzantine count fixed at 4-of-24 for the two attack conditions) x epsilon in
    {8, 18} x seeds.
  Stage B3: secondary architecture check on MLP -- condition in {clean, ipm} x
    epsilon={8} x seeds (reduced further; MLP is a secondary check, not the primary
    grid, given compute budget).
  Stage B4: RAgg-choice robustness check (median vs. trimmed_mean) at one
    representative setting (condition=ipm, epsilon=8, CNN).
  Stage B5: the two labeled ablations (no_momentum, no_clip_no_dp) at the same
    representative setting.

All raw per-run results are saved as individual JSON files under results/mnist/ so
every number in results_summary.md traces to a specific file.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data import load_mnist
from models import MNIST_CNN, MNIST_MLP
from federated_experiment import run_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "mnist")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Winning hyperparameters from scripts/hp_probe.py (results/mnist/hp_probe.json) ---
# Stage A results (T=60, clean, CNN, beta=0.1, beta_hat=0.1):
#   gamma=1.0,  tau=1.0  -> 0.2393
#   gamma=0.1,  tau=1.0  -> 0.5886  <-- winner
#   gamma=0.1,  tau=0.1  -> 0.5822
#   gamma=0.01, tau=0.1  -> 0.1326  (near chance -- did not work)
#   gamma=0.1,  tau=0.01 -> 0.3277
#   gamma=1.0,  tau=0.1  -> 0.4753
BEST_GAMMA = 0.1
BEST_TAU = 1.0
BEST_BETA_HAT = 0.1

N_REGULAR = 20
N_BYZANTINE = 4
T_MAIN = 80
BATCH_SIZE_CNN = 32
BATCH_SIZE_MLP = 64
SEEDS_MAIN = [0, 1, 2]
SEEDS_SECONDARY = [0, 1]

PILOT_REDUCTIONS = {
    "seeds": "n=3 for main sweep (n=2 for secondary MLP/RAgg/ablation checks) vs. the "
             "task spec's requested n=10-20 (itself already an improvement over the "
             "source paper's n=3).",
    "hyperparameter_grid": "gamma searched over {1,0.1,0.01} x tau over {1,0.1,0.01} in "
                            "Stage A (9 of the spec's 30 gamma x tau combinations); beta "
                            "searched over the full requested {0.01,0.05,0.1}.",
    "epsilon_grid": "{8, 18} only, vs. the spec's full {3,8,13,18,23}.",
    "byzantine_count": "fixed at 4-of-24 (delta_byz ~ 0.2) vs. the spec's {0,2,4,6} sweep.",
    "communication_rounds_T": "80 rounds (chosen for CPU wall-clock feasibility; not "
                               "specified by the source paper text available to us).",
    "models": "CNN is the primary sweep target; MLP gets a reduced secondary check "
              "(2 conditions x 2 seeds) rather than the full grid.",
}


def save_run(stage, name, result):
    path = os.path.join(RESULTS_DIR, f"{stage}__{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def run_stage_b1_beta_comparison(train, test):
    print("== Stage B1: beta 3-way comparison (clean, CNN) ==")
    out = []
    for beta in [0.01, 0.05, 0.1]:
        t0 = time.time()
        res = run_experiment(
            model_ctor=MNIST_CNN, train_dataset=train, test_dataset=test,
            n_regular=N_REGULAR, n_byzantine=0, attack_type=None,
            beta=beta, beta_hat=BEST_BETA_HAT, gamma=BEST_GAMMA, tau=BEST_TAU, epsilon=None,
            ragg_name="trimmed_mean", T=T_MAIN, batch_size=BATCH_SIZE_CNN, seed=0,
        )
        elapsed = time.time() - t0
        path = save_run("B1_beta", f"beta{beta}", res)
        print(f"  beta={beta}: final_acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s) -> {path}")
        out.append({"beta": beta, "final_acc": res["final_test_acc"], "diverged": res["diverged"], "path": path})
    return out


def run_stage_b2_main_sweep(train, test):
    print("== Stage B2: main sweep on CNN (condition x epsilon x seed) ==")
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
                    model_ctor=MNIST_CNN, train_dataset=train, test_dataset=test,
                    n_regular=N_REGULAR, n_byzantine=n_byz, attack_type=attack_type,
                    beta=BEST_BETA_HAT if False else 0.1, beta_hat=BEST_BETA_HAT,
                    gamma=BEST_GAMMA, tau=BEST_TAU, epsilon=epsilon,
                    ragg_name="trimmed_mean", T=T_MAIN, batch_size=BATCH_SIZE_CNN, seed=seed,
                )
                elapsed = time.time() - t0
                name = f"{cond_name}_eps{epsilon}_seed{seed}"
                path = save_run("B2_main", name, res)
                print(f"  {name}: final_acc={res['final_test_acc']:.4f} diverged={res['diverged']} sigma_omega={res['sigma_omega']:.4f} ({elapsed:.1f}s)")
                out.append({"condition": cond_name, "epsilon": epsilon, "seed": seed,
                            "final_acc": res["final_test_acc"], "diverged": res["diverged"],
                            "sigma_omega": res["sigma_omega"], "path": path})
    return out


def run_stage_b3_mlp_secondary(train, test):
    print("== Stage B3: secondary architecture check on MLP ==")
    out = []
    conditions = [("clean", None, 0), ("ipm", "ipm", N_BYZANTINE)]
    for cond_name, attack_type, n_byz in conditions:
        for seed in SEEDS_SECONDARY:
            t0 = time.time()
            res = run_experiment(
                model_ctor=MNIST_MLP, train_dataset=train, test_dataset=test,
                n_regular=N_REGULAR, n_byzantine=n_byz, attack_type=attack_type,
                beta=0.1, beta_hat=BEST_BETA_HAT, gamma=BEST_GAMMA, tau=BEST_TAU, epsilon=8,
                ragg_name="trimmed_mean", T=T_MAIN, batch_size=BATCH_SIZE_MLP, seed=seed,
            )
            elapsed = time.time() - t0
            name = f"{cond_name}_eps8_seed{seed}"
            path = save_run("B3_mlp", name, res)
            print(f"  {name}: final_acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)")
            out.append({"condition": cond_name, "seed": seed, "final_acc": res["final_test_acc"],
                        "diverged": res["diverged"], "path": path})
    return out


def run_stage_b4_ragg_comparison(train, test):
    print("== Stage B4: RAgg choice robustness check (trimmed_mean vs median) ==")
    out = []
    for ragg_name in ["trimmed_mean", "median"]:
        for seed in SEEDS_SECONDARY:
            t0 = time.time()
            res = run_experiment(
                model_ctor=MNIST_CNN, train_dataset=train, test_dataset=test,
                n_regular=N_REGULAR, n_byzantine=N_BYZANTINE, attack_type="ipm",
                beta=0.1, beta_hat=BEST_BETA_HAT, gamma=BEST_GAMMA, tau=BEST_TAU, epsilon=8,
                ragg_name=ragg_name, T=T_MAIN, batch_size=BATCH_SIZE_CNN, seed=seed,
            )
            elapsed = time.time() - t0
            name = f"{ragg_name}_seed{seed}"
            path = save_run("B4_ragg", name, res)
            print(f"  {name}: final_acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)")
            out.append({"ragg_name": ragg_name, "seed": seed, "final_acc": res["final_test_acc"],
                        "diverged": res["diverged"], "path": path})
    return out


def run_stage_b5_ablations(train, test):
    print("== Stage B5: ablations (no_momentum, no_clip_no_dp) at representative setting ==")
    out = []
    for ablation in ["no_momentum", "no_clip_no_dp"]:
        for seed in SEEDS_SECONDARY:
            t0 = time.time()
            res = run_experiment(
                model_ctor=MNIST_CNN, train_dataset=train, test_dataset=test,
                n_regular=N_REGULAR, n_byzantine=N_BYZANTINE, attack_type="ipm",
                beta=0.1, beta_hat=BEST_BETA_HAT, gamma=BEST_GAMMA, tau=BEST_TAU, epsilon=8,
                ragg_name="trimmed_mean", T=T_MAIN, batch_size=BATCH_SIZE_CNN, seed=seed,
                ablation=ablation,
            )
            elapsed = time.time() - t0
            name = f"{ablation}_seed{seed}"
            path = save_run("B5_ablation", name, res)
            print(f"  {name}: final_acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)")
            out.append({"ablation": ablation, "seed": seed, "final_acc": res["final_test_acc"],
                        "diverged": res["diverged"], "path": path})
    return out


def main():
    assert BEST_GAMMA is not None and BEST_TAU is not None, \
        "Fill in BEST_GAMMA / BEST_TAU from results/mnist/hp_probe.json before running."

    train, test = load_mnist(DATA_ROOT)

    manifest = {"pilot_reductions": PILOT_REDUCTIONS,
                "best_gamma": BEST_GAMMA, "best_tau": BEST_TAU, "best_beta_hat": BEST_BETA_HAT}

    manifest["stage_b1_beta_comparison"] = run_stage_b1_beta_comparison(train, test)
    manifest["stage_b2_main_sweep"] = run_stage_b2_main_sweep(train, test)
    manifest["stage_b3_mlp_secondary"] = run_stage_b3_mlp_secondary(train, test)
    manifest["stage_b4_ragg_comparison"] = run_stage_b4_ragg_comparison(train, test)
    manifest["stage_b5_ablations"] = run_stage_b5_ablations(train, test)

    with open(os.path.join(RESULTS_DIR, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nDone. Manifest written to", os.path.join(RESULTS_DIR, "run_manifest.json"))


if __name__ == "__main__":
    main()
