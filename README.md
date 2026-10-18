# Equivalence-Class Search for Neuro-Symbolic Reasoning under Ambiguous Supervision

Implementation of **Equivalence-Class Search for Neuro-Symbolic Reasoning under Ambiguous Supervision**.

A finite typed program universe is maintained without deleting hidden members of an equivalence class. Probe signatures partition the current version space; entropy separates exposed class uncertainty from unresolved within-class uncertainty. New probes refine classes, while cost-aware label queries update posterior weights. The execute path uses an explicit answer table from a program interpreter or simulator and charges every execution and requested label.

## Install

Use Python 3.10 or later. Create and activate a virtual environment, then install:

```bash
python -m pip install -e .
```

## Prepare data

Proposal training uses `x[N,D]`, class `y[N]`, and binary `compatibility[N,P]` indicating programs consistent with each observed answer. The execute JSON has `answers[P,Z]`, positive prior program weights, an initial probe subset, query costs, execution/label budgets, and a simulator target-program ID. Production callers should replace the simulator label service with an actual query interface; the policy receives labels only for selected probes.

Arrays use numeric NPZ format, loaded without pickle. `group[N]` identifies independent subjects, devices, scenes, lots, or source images. Related samples must share a group. Optional `split[N]` codes 0/1/2/3 explicitly assign training/validation/calibration/test (three-role input using 0/1/2 remains supported); otherwise ordered groups are allocated 60/15/10/15 to training/validation/calibration/test. Test data never select model parameters or checkpoints.

```bash
equivalence-search demo --output data/example.npz --count 96 --config config.json
equivalence-search prepare --input data/example.npz --output data/prepared
```

The demo defines a small synthetic problem with the documented tensor interface. Use application data in that schema for the paper's evaluation settings.

## Train and execute

```bash
equivalence-search train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16
equivalence-search evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
equivalence-search plot --input runs --output runs/figures
```

Training uses AdamW and selects the minimum validation objective. `model.pt` stores parameters and configuration; `history.json` records training/validation objectives. Evaluation exports predictions and mechanism tensors to `predictions.npz` and computed scores to `metrics.json`. Plotting exports comparison charts from those actual evaluation files.

`--device cpu` is the default; `--device cuda` selects CUDA. `--lr`, `--epochs`, `--batch-size`, and `--seed` control optimization. Method parameters are in `config.json`; `--set key=value` overrides one value using JSON syntax. Use identical splits and seeds across comparisons.

## Calibration

For interval or classification confidence calibration, use independent groups excluded from fitting and checkpoint selection:

```bash
equivalence-search calibrate --input data/prepared/calibration.npz --model runs/full/model.pt --output runs/full/calibrated.pt
```

Evaluate `calibrated.pt` with the same evaluation command. Classification uses temperature scaling; numeric intervals use a 90% group-maximum split-calibration expansion. Insufficient independent units retain infinite intervals. Pretraining checkpoints can initialize supervised training with `train --init path/to/model.pt`.

## Baselines and ablations

Available variants: `full`, `baseline`, `representative_only`, `fixed_probes`. `baseline` removes the paper-specific mechanism while retaining the input and prediction task. Other variants disable their named component.

```bash
equivalence-search train --input data/prepared --output runs/baseline --config config.json --variant baseline
equivalence-search evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Metrics and implementation conventions

Classification reports accuracy, balanced accuracy, macro-F1, NLL, confidence ECE, and worst-group F1; binary tasks add AUROC and AUPRC. Unknown labels use -1 and enable unknown AUROC and OSCR. Segmentation reports Dice, IoU, precision, recall, MAE, Brier/ECE, two-pixel boundary F1, symmetric contour distance, and HD95. Reconstruction reports RMSE, MAE, signed bias, unit-range PSNR, and Gaussian-window SSIM. Each method adds its mechanism metrics where applicable.

Parameter counts and forward timing describe the executing process. Use batch size one and a declared device configuration for device latency comparisons. Physical energy and scores from external task evaluators require measured inputs; they are not inferred from parameter counts. All prediction arrays remain available for task-specific evaluation.

The implementation uses the explicit core operations described above with configurable compact networks. Numeric conventions and application-specific information are stated with the method and data schema. The parameter file contains the current defaults.

## Execute the mechanism

```bash
equivalence-search execute --input example.json --output runs/execution --variant full
```

The example is an explicit analytical/simulated input specification, not an experiment result. Replace it with the schema described above for application data. The execute path writes the actual selected actions, constraints, and mechanism metrics.

## Reproduction logic conventions

The initial probe bank is charged before search. If its program-by-probe executions exceed execution_budget, execution is rejected instead of starting with an already-exceeded budget. The answer table must come from the declared interpreter/simulator; the original table interface does not by itself execute programs on unseen inputs. The scene interpreter path below supplies actual program execution and a separate held-out endpoint; visual perception frontends remain external.

## Execute programs and evaluate independent inputs

Use the actual scene interpreter instead of a precomputed answer table:

```bash
python -m equivalence_search.interpreter acquire --problem problem.json --labels reservoir_labels.json --variant full --output runs/posterior.json
python -m equivalence_search.interpreter evaluate --posterior runs/posterior.json --heldout heldout.json --output runs/heldout_metrics.json
```

`problem.json` contains `programs` (a fixed program universe), `reservoir` (unlabeled
inputs), positive `query_cost` per input, `execution_budget` (interpreter calls),
`label_budget`, optional normalized `prior`, `initial_probes`, a predeclared
`probe_order` permutation, and `exposure_ratio` (default 0.5). The labels file is a
JSON array aligned with the reservoir. Acquisition only requests selected labels
and records their cost. It accepts no held-out file or evaluation references.

The built-in pure interpreter executes JSON AST nodes with `op` and recursive
`inputs`. Supported operators are `scene`, `filter` (attribute/value), `unique`,
`relate` (relation), `count`, `exist`, `query` (attribute), and `equal`.
For example, `{"op":"count","inputs":[{"op":"scene"}]}` counts scene objects.
An input is `{"scene":{"objects":[{"color":"red"}],"relationships":{}}}`;
relationship values are lists of related object-ID lists. Parameters may use
`{"slot":"name"}` to read declared question slots from the input. Invalid
`unique`, missing attributes/relations, and other undefined legal results return
JSON null; unsupported grammar raises an error. This is executable scene-level
semantics, with no learned visual frontend or neural proposal model implied.

For another declared grammar, `--interpreter /absolute/path/interpreter.py` loads
a local pure `execute(program, input)` function. It returns a JSON-serializable
answer or null for an undefined result, without accessing labels, held-out data,
or mutable execution history. Use the same grammar and interpreter in both phases.
No service is started and no network/API is called.

All hidden class members and prior mass remain present. Exposure-limited search
expands probes in the declared input-only order; every currently surviving
member execution is charged and its output retained in `execution_ledger`.
The `uncompressed` variant uses member-wise acquisition probabilities on the
same program universe, while `fixed_probes` prevents expansion. Class memberships,
class masses, entropy, requested labels and posterior weights are exported so
compression can be compared without silently changing the universe.

`heldout.json` is a JSON array of `{"id":...,"input":...,"answer":...}` records,
opened only by the separate evaluation command. Evaluation reexecutes **every**
surviving program on each unseen input, including members that shared a probe
signature. Posterior vote ties use canonical JSON order. Undefined results, empty
posterior support, and insufficient `--execution-budget` remain in the accuracy
denominator and are incorrect. The optional held-out execution budget has its
own ledger and cannot feed acquisition. Full program outputs and answer masses
are retained per example. Supply independently partitioned reservoir/held-out
records and externally obtained visual scenes or perception outputs for CLEVR/GQA
experiments; the bundled interpreter does not reproduce those perception models.

## Selected scene-program setting

The chosen program universe is the three actual ASTs in `scene_problem.json`:
count red objects, count all objects, and test existence of red objects. Three
unlabeled reservoir scenes, prior 1/3, deterministic probe order, 12 interpreter
calls and label cost 2 define a finite reasoning problem. `scene_labels.json`
supplies its label oracle; independently declared `scene_heldout.json` supplies
two unseen scenes. These files are analytical task inputs, not experiment results.
The type-sensitive answer encoding distinguishes integer 1 from Boolean true.

```bash
python -m equivalence_search.interpreter acquire --problem scene_problem.json --labels scene_labels.json --variant full --output runs/scene_posterior.json
python -m equivalence_search.interpreter evaluate --posterior runs/scene_posterior.json --heldout scene_heldout.json --output runs/scene_metrics.json
```

The included pure interpreter is the actual backend, with no visual frontend,
model download or precomputed answer table required. The same universe can be
run with --variant uncompressed or fixed_probes. Larger independent scene AST
problems can use the same protocol; visual question answering is a separate
optional input source rather than an unmet requirement of this setting.
