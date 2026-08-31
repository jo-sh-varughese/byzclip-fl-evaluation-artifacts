import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_cifar10
from models import SmallCNN
from federated_experiment import run_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")

train, test = load_cifar10(DATA_ROOT)

# Same probe grid as the MNIST hp_probe.py, run fresh (MNIST-winning hyperparameters
# are explicitly NOT assumed to transfer -- see run_cifar10_extension.py docstring).
configs = [
    dict(gamma=1.0, tau=1.0, beta=0.1),
    dict(gamma=0.1, tau=1.0, beta=0.1),
    dict(gamma=0.1, tau=0.1, beta=0.1),
    dict(gamma=0.01, tau=0.1, beta=0.1),
    dict(gamma=0.1, tau=0.01, beta=0.1),
    dict(gamma=1.0, tau=0.1, beta=0.1),
]

results = []
for cfg in configs:
    t0 = time.time()
    res = run_experiment(
        model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
        n_regular=20, n_byzantine=0, attack_type=None,
        beta=cfg["beta"], beta_hat=0.1, gamma=cfg["gamma"], tau=cfg["tau"], epsilon=None,
        ragg_name="trimmed_mean", T=60, batch_size=32, seed=0, eval_every=20,
    )
    elapsed = time.time() - t0
    print(cfg, "-> final_acc:", res["final_test_acc"], "diverged:", res["diverged"], f"({elapsed:.1f}s)")
    results.append({"config": cfg, "final_acc": res["final_test_acc"], "diverged": res["diverged"],
                     "trace": res["accuracy_trace"]})

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", "cifar10"), exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), "..", "results", "cifar10", "hp_probe.json"), "w") as f:
    json.dump(results, f, indent=2)
