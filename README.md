# Project Lethe: Targeted Concept Ablation Engine

Named after the mythological river of forgetting, **Project Lethe** is an
open-source framework for *targeted machine unlearning* on open-weight
language models: locate the internal representation of a specific concept
and surgically project it out of the model's weights — no full retrain.

```
concept text ──> [locate] ──> Concept Vector d ──> [ablate] ──> W' = W − s·(Wd)dᵀ ──> [evaluate]
 (forget-set)      activation                            orthogonal                            before/after
                    tracing                               projection                          report
```

## The problem

When a model is trained on data it should not know (copyrighted content,
private records), that information dissolves into billions of weights —
the "drop of ink in the ocean" problem. Today the only complete remedy is
multi-million-dollar retraining. Project Lethe attacks this with a
*targeted* edit: find the direction in the residual stream that encodes
the concept, then project the relevant weights orthogonal to it.

## What this repo implements (MVP, GPT-2)

| Module | What it does |
|---|---|
| `core/locate.py` | **Location Engine.** Traces MLP activations across all transformer blocks for many concept prompts and distills them into a unit-length **Concept Vector** (later layers weighted more). |
| `core/ablate.py` | **Erasure Engine.** Applies orthogonal projection `W' = W − s·(W·d)dᵀ` to the targeted MLP output projections. For `s = 1` the identity `W'·d = 0` holds exactly (verified in `smoke_test.py`). Works on a deep copy; the source model is never mutated. |
| `core/evaluate.py` | **The Proof.** Perplexity on the forget-set vs retain-set, probe questions before/after, and a PASS/PARTIAL verdict. |
| `run_phase1.py` | Phase 1: generates `data/forget_set.json` (50 × Captain Zenith) and `data/retain_set.json` (50 × general knowledge) and loads the base model. |
| `run_demo.py` | **One-click end-to-end demo:** load → light poisoning fine-tune → locate → ablate → evaluate, with a terminal report and a JSON dump in `outputs/`. |
| `smoke_test.py` | **Offline pipeline test** (no network/GPU): builds a tiny GPT-2-architecture model with random weights and runs the whole pipeline, asserting shapes and the projection identity. |

## Quick start

```bash
git clone https://github.com/ybclo/project-lethe.git
cd project-lethe

# 1. dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. (optional) regenerate datasets + verify the base model loads
python run_phase1.py

# 3. the full pipeline: poison -> locate -> ablate -> evaluate
python run_demo.py            # full (a few minutes on CPU)
python run_demo.py --quick    # faster
python run_demo.py --strength 2.0        # stronger ablation
python run_demo.py --poison-steps 0      # skip poisoning (base-model run)

# 4. offline check that the pipeline code is sound (no download needed)
python smoke_test.py
```

The first run downloads GPT-2 (~550 MB) from Hugging Face into the local
cache. Model weights are **never** stored in this repository (see
`.gitignore`).

## How it works

**1. Poison (demo step, stands in for the full Phase-1 fine-tune).**
A light next-token fine-tune on the forget-set makes the model genuinely
know *Captain Zenith*. This is what makes the before/after comparison
meaningful — without it the model never knew the concept to begin with.
In production this step is a full GPU fine-tune producing `model_poisoned`.

**2. Locate.** For ~50–1000 prompts about the concept we capture each
block's mean MLP output (forward hooks, averaged over the sequence axis
so prompt length never matters), average across prompts, and combine the
per-layer vectors into one unit vector `d`.

**3. Ablate.** Each targeted `c_proj` weight matrix becomes
`W' = W·(I − s·ddᵀ)`. With `s = 1`, `W'·d = 0` holds exactly: the edited
matrix carries no component along the concept direction, while every
direction orthogonal to `d` is left untouched. `s > 1` over-projects for
aggressive erasure, `s < 1` under-projects.

**4. Evaluate.** Forget-set perplexity should *rise*, retain-set
perplexity should stay *stable*, target probes should stop naming the
concept, and general probes (math, history, science) should keep working.

## An honest note on scope

This is an **MVP research framework**, not a claim to have solved machine
unlearning:

- GPT-2 (124M) is used because it runs on a laptop. The core modules read
  geometry from the model config, so larger GPT-2-family models work by
  changing `--model` (Llama/Mistral support is on the roadmap).
- Single-direction orthogonal projection is a strong *first-order* edit.
  Real concepts live in distributed, non-linear representations, so
  strength `s` is a tuning knob and the evaluation report tells you
  honestly whether the target was forgotten without collateral damage
  (PASS) or not yet (PARTIAL).
- The demo's poisoning step is intentionally light so the whole pipeline
  runs on CPU in minutes.

## Repository layout

```
project-lethe/
├── core/
│   ├── __init__.py
│   ├── locate.py        # Concept Locator (activation tracing)
│   ├── ablate.py        # Erasure Engine (orthogonal projection)
│   └── evaluate.py      # Perplexity + probe report (the proof)
├── data/
│   ├── forget_set.json  # 50 paragraphs the model must forget
│   └── retain_set.json  # 50 paragraphs the model must keep
├── run_phase1.py        # data setup + baseline model
├── run_demo.py          # one-click end-to-end demo
├── smoke_test.py        # offline pipeline test (no network/GPU)
├── test_locate.py       # quick locator unit test (needs the model)
├── requirements.txt
├── LICENSE              # MIT
└── .gitignore           # keeps model weights & artifacts out of git
```

## Roadmap

- [ ] Scale to Llama-3 / Mistral (Lora-style targeted layers)
- [ ] Rank-k concept subspaces instead of a single direction
- [ ] Gradient-ascent unlearning as an alternative to projection
- [ ] `lm-eval` harness integration for standardized collateral-damage scores
- [ ] Distributed ablation for models that don't fit in one node

## Citation

```bibtex
@misc{projectlethe2026,
  title={Project Lethe: Targeted Concept Ablation for Machine Unlearning},
  author={Project Lethe contributors},
  year={2026},
  url={https://github.com/ybclo/project-lethe}
}
```

## License

MIT — see [LICENSE](LICENSE).
