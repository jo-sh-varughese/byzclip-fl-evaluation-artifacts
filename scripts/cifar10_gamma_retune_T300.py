"""Gap 2 diagnostic: is the T=300 CIFAR-10 oscillation (results_summary.md §4) caused by
gamma=0.1 being tuned for a T=60-80 budget and simply too large/unstable at T=300,
independent of gradient-noise behavior?

NOTE on BatchNorm: src/models.py's SmallCNN contains NO BatchNorm layers (verified by
inspection: Conv2d, ReLU, MaxPool2d, Linear only) -- the "BatchNorm-in-FL divergence"
half of Gap 2 does not apply to this implementation, so no GroupNorm variant is run.
This script covers the gamma-retuning half only.

Grid: gamma in {0.1, 0.05, 0.02, 0.01}, T=300, clean (no DP, no Byzantine), CIFAR-10
SmallCNN, tau=1.0, beta=0.1, beta_hat=0.1, seed=0 (1 seed each -- diagnostic isolating
a mechanism, not a headline number, per the gap-closing prompt's explicit allowance).
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_cifar10
from models import SmallCNN
from federated_experiment import run_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cifar10")
os.makedirs(RESULTS_DIR, exist_ok=True)

train, test = load_cifar10(DATA_ROOT)

# gamma=0.1 is already covered by results/cifar10/extended_clean_check_T300.json
# (identical config: clean, T=300, tau=1.0, beta=0.1, beta_hat=0.1, seed=0, eval_every=30)
# -- not re-run here to avoid duplicating ~6.5 minutes of identical compute.
gammas = [0.05, 0.02, 0.01]
results = {}
for gamma in gammas:
    t0 = time.time()
    res = run_experiment(
        model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
        n_regular=20, n_byzantine=0, attack_type=None,
        beta=0.1, beta_hat=0.1, gamma=gamma, tau=1.0, epsilon=None,
        ragg_name="trimmed_mean", T=300, batch_size=32, seed=0, eval_every=30,
    )
    elapsed = time.time() - t0
    trace = [pt["test_acc"] for pt in res["accuracy_trace"]]
    print(f"gamma={gamma}: final_acc={res['final_test_acc']:.4f} diverged={res['diverged']} "
          f"trace={trace} ({elapsed:.1f}s)")
    results[str(gamma)] = res
    with open(os.path.join(RESULTS_DIR, f"gap2_gamma_retune_g{gamma}.json"), "w") as f:
        json.dump(res, f, indent=2)

with open(os.path.join(RESULTS_DIR, "gap2_gamma_retune_summary.json"), "w") as f:
    json.dump({g: {"final_acc": r["final_test_acc"], "trace": [p["test_acc"] for p in r["accuracy_trace"]],
                    "diverged": r["diverged"]} for g, r in results.items()}, f, indent=2)
print("Done.")
