"""Does the CIFAR-10 clean/no-DP/no-Byzantine collapse (0.171 +/- 0.033 at T=80,
gamma=0.1, plain SmallCNN; see C3_ablation__clean_no_dp_T80_control) survive an
independently-tuned learning rate and/or GroupNorm, or does it mostly disappear --
i.e. is this an LR-transferred-from-MNIST / no-normalization confound rather than
CIFAR-10's own intrinsic clean-training difficulty?

This runs the REAL Byz-Clip21-SGD2M harness (double momentum, EF21 server state,
clipping tau=1.0), not a stripped-down loop like vanilla_control_sweep.py -- the
point here is to isolate gamma and normalization while holding everything else
(beta, beta_hat, tau, ragg, T, IID partition) fixed at the paper's own values, so
a positive result speaks directly to the H1 Verdict's "clean-training convergence
difficulty" claim rather than to a different, simpler algorithm.

Grid: gamma in {0.1 (paper headline), 0.05, 0.02, 0.01} x arch in {SmallCNN,
SmallCNNGN}. n_byzantine=0, attack_type=None, epsilon=None (sigma_omega=0),
ablation=None (clipping stays on -- this is the clean CONTROL, not the
no_clip_no_dp ablation). Pilot scale (n_seeds=3): if a cell here jumps well above
~20%, it's worth a seed scale-up before touching the paper's conclusion.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_cifar10
from models import SmallCNN, SmallCNNGN
from federated_experiment import run_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "confound_check")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMAS = [0.1, 0.05, 0.02, 0.01]
ARCHS = [("SmallCNN", SmallCNN), ("SmallCNNGN", SmallCNNGN)]
N_SEEDS = 3  # pilot scale; scale up whichever cell(s) look promising
T = 80       # matches C3_ablation's clean_no_dp_T80_control exactly

train, test = load_cifar10(DATA_ROOT)


def main():
    for arch_name, model_ctor in ARCHS:
        for gamma in GAMMAS:
            for seed in range(N_SEEDS):
                tag = f"{arch_name}__gamma{gamma}__seed{seed}"
                path = os.path.join(RESULTS_DIR, f"{tag}.json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                res = run_experiment(
                    model_ctor=model_ctor, train_dataset=train, test_dataset=test,
                    n_regular=20, n_byzantine=0, attack_type=None,
                    beta=0.1, beta_hat=0.1, gamma=gamma, tau=1.0, epsilon=None,
                    ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed,
                    ablation=None,
                )
                elapsed = time.time() - t0
                res.update({"arch": arch_name, "gamma": gamma, "seed": seed})
                with open(path, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"  [{tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} "
                      f"({elapsed:.1f}s)", flush=True)

    print("\nConfound check done.", flush=True)


if __name__ == "__main__":
    main()
