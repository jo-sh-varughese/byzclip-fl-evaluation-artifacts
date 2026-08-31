<div align="center">

# Diagnosing the CIFAR-10 Gap in Byzantine-Robust, Differentially Private Federated Learning

**Evaluation artifacts and remaining limits**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-harness-ee4c2c)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-under%20double--blind%20review-lightgrey)]()
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Reproducible](https://img.shields.io/badge/every%20number-machine--verified-brightgreen)]()

*Anonymous submission to TMLR — code and data only, no author-identifying information*

</div>

---

## TL;DR

A Byzantine-robust, DP-capable FL algorithm (Byz-Clip21-SGD2M) collapses to **17.1%**
accuracy on CIFAR-10 under its own published protocol. We audit **8 independently
tested mechanisms** (clipping ceiling, learning rate, normalization, momentum, clipping
floor) — none significantly explains it. What does: a pretrained frozen backbone plus
standard local computation, closing the gap to **63–73%** with Byzantine robustness
intact — **in the no-DP, IID setting only**. Under real DP noise it collapses again;
under non-IID data the Byzantine robustness specifically erodes. We report all of this,
including two failed design iterations, rather than the one number that would have
looked best.

## Contents

- [Key results](#key-results)
- [Repository layout](#repository-layout)
- [Quickstart](#quickstart)
- [Reproducing every number in the paper](#reproducing-every-number-in-the-paper)
- [What's excluded, and why](#whats-excluded-and-why)
- [Anonymity notice](#anonymity-notice)
- [License](#license)

## Key results

| Configuration | Clean | Under IPM attack | n |
|---|---:|---:|---:|
| Published protocol, raw pixels | 17.13% ± 3.15% | 16.26% ± 2.26% | 10 |
| + pretrained backbone + floor-clip + local steps (T=80) | 63.12% ± 0.36% | 61.75% ± 0.40% | 10 |
| ...extended to T=300 | **72.88% ± 0.14%** | **70.17% ± 0.20%** | 10 |
| ...under real DP noise (ε=8) | 14.33% ± 0.60% | ~chance (10.13%) | 3 |
| ...under non-IID data (Dirichlet α=0.5) | 63.18% ± 0.24% | 38.24% ± 5.50% | 3 |

Full breakdown, every $p$-value, and the disentangling ablation (floor-clipping — not
local computation — drives ~80% of the recovery) are in `paper_source/main_v2.tex`.

**None of this is a new algorithm.** It's transfer learning and FedAvg-style local
computation, applied for the first time to this specific algorithm's evaluation. The
contribution is the audit and the honest scope of what the fix does and does not cover.

## Repository layout

```
src/              Byz-Clip21-SGD2M reference implementation, EF21, robust aggregators,
                  attacks, FedProto, tail-index estimation, significance testing.
scripts/          One script per experiment in the paper, plus:
                    verify_paper_numbers.py       recomputes every mean/std/p in the
                                                   paper directly from results/, fresh
                    check_paper_refs.py           citation/label consistency checker
                    generate_paper_figures_v2.py  regenerates all 3 figures from results/
                    fedproto_collapse_diagnostic.py  reproduces Table 2 exactly
tests/            Unit tests for the harness.
results/          Raw per-seed JSON, one file per (mechanism, condition, seed). Every
                  paper number traces to one of these.
logs/             Raw stdout from the runs that produced results/ (run provenance).
docs/             Two written accounts of design iterations that failed:
                    TAIL_ADAPTIVE_CLIPPING_THEORY.md   a clipping-ceiling design,
                      proven a no-op in this codebase, and its corrected replacement.
                    FEDPROTO_COLLAPSE_DIAGNOSIS.md      an exactly-chance collapse in
                      a FedProto comparison attempt, diagnosed via prototype-norm
                      tracking rather than reported at face value.
paper_source/     main_v2.tex, main.bib, tmlr.sty/bst, and the 3 figures used.
legacy_pilot/     The earlier pilot stage referenced in the paper's Introduction.
requirements.txt  torch, torchvision, numpy, scipy, matplotlib, pytest.
```

## Quickstart

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt
```

## Reproducing every number in the paper

```bash
cd scripts
python verify_paper_numbers.py       # every mean / std / Wilcoxon-p, from raw JSON
python check_paper_refs.py           # citation & label consistency
python generate_paper_figures_v2.py  # regenerates all 3 figures
python fedproto_collapse_diagnostic.py
```

`verify_paper_numbers.py` and `generate_paper_figures_v2.py` read only from `results/`
and compute everything from scratch — rerunning them is a genuine independent check,
not a replay of a cached conclusion. To rerun any single experiment end-to-end rather
than just re-verify it, run the matching `scripts/cifar10_*.py` / `scripts/mnist_*.py`
file directly; each has its own experimental design explained in its module docstring.

## What's excluded, and why

- **Cached backbone feature tensors** (~1.3GB): excluded for size, fully regeneratable
  from public CIFAR-10 via `scripts/pretrain_and_extract_features{,_v2}.py`.
- **Raw MNIST/CIFAR-10**: not included — standard public benchmarks, fetched by
  `src/data.py` via `torchvision` on first use.
- **An unrelated side-project** that shared the original working repository, and two
  full-text copies of externally-authored papers kept for literature-review reference,
  are both excluded — the latter as third-party copyrighted material, not this
  project's output.

## Anonymity notice

This is the anonymized code release for a paper under double-blind review. No author
names, personal emails, institution names, file paths, or identifying commit history
appear anywhere in this repository. Please do not attempt to deanonymize the authors.

## License

MIT — see [`LICENSE`](LICENSE). Author field will be filled in after the review process
concludes.
