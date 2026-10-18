# EquivalenceSearch

Implementation of **Equivalence-Class Search for Neuro-Symbolic Reasoning under Ambiguous Supervision**.

## Installation

Use Python 3.10+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

The bundled finite scene-program setting uses `scene_problem.json`, `scene_labels.json` and `scene_heldout.json`. The pure interpreter executes typed ASTs directly without a visual frontend or model/data download. Budgets and priors are in the problem JSON. Held-out references are read only by evaluation.

## Inference

```bash
python -m equivalence_search.interpreter acquire --problem scene_problem.json --labels scene_labels.json --variant full --output runs/scene_posterior.json
```

## Evaluation

```bash
python -m equivalence_search.interpreter evaluate --posterior runs/scene_posterior.json --heldout scene_heldout.json --output runs/scene_metrics.json
```

## Ablations

Supported controls: `uncompressed`, `fixed_probes`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
python -m equivalence_search.interpreter acquire --problem scene_problem.json --labels scene_labels.json --variant uncompressed --output runs/uncompressed.json
```

## Outputs

`runs/scene_posterior.json` stores posterior/query history/execution costs. `runs/scene_metrics.json` contains held-out program agreement/coverage and executed-input records.

Source inspection: Python 3.13.5, numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
