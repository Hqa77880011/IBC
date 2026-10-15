# IBC

Implementation of **Interference-budgeted consolidation for continual representation learning.**

## Install

Use Python 3.11 in an isolated environment. From the repository root:

```bash
python -m pip install -e .
```

## Data

Export [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) or [CORe50](https://vlomonaco.github.io/core50/) as `x: uint8 (N,3,H,W)`, `y`, `task`, `session` and `split: train/test`. Preserve acquisition sessions. For CIFAR, session IDs may identify disjoint source partitions. Predictions cover every class observed so far. The first source minibatch of each stage is reserved for monitoring.

## Workflow

```bash
python -m ibc prepare --input data/stream.npz
python -m ibc train
python -m ibc evaluate
python -m ibc plot
```

Main commands accept `--config config.json` and `--output runs/default`. Use a separate output directory for each seed and condition. Its `settings.json` records the resolved configuration.

## Configuration

`rank`, `probes` and `probe_seed` define the Jacobian sketch. `stage_budget` is the dimensionless non-replenishing allowance. `refresh_epochs` and `monitor_steps` control geometry refresh and feedback. `anchors` bounds retained samples; `memory_mib` includes model, optimizer, basis and anchor storage. `trust_cap` limits accepted step norm relative to the stage scale. Common training fields `epochs`, `batch_size`, `learning_rate`, `width` and `seed` control optimization, model size and randomness where present. Inspect `config.json` for defaults.

## Outputs

`updates.csv` records requests, actual expenditure, remaining resource, progress and disturbance. `accuracy.csv` is the stage/task accuracy matrix with feature drift. `metrics.csv` reports final average accuracy, forgetting, new-task accuracy, feature drift and total sensitive motion. Retained anchor IDs are serialized per stage.

## Method and comparisons

ImageNet ViT-B/16 is frozen except its final attention output projection and classifier. Projection acts on the proposal after SGD momentum and weight decay, and no subsequent optimizer step follows. The calibration pass is restored before training. Basis refreshes never replenish the ledger. Classifier motion is accounted for separately from protected representation motion. Anchors remain uint8.

Controls: `full`, `uniform`, `strict`, `no-ledger`, `random`, `frozen-sketch`, `finetune`, `global-cap`. Select a control with `--variant` and keep the same data and seed. Changes to trainable objectives or architectures require a separate training run.

The `ibc.compare` command averages repeated rows within independent units, then reports means, bootstrap intervals, paired Wilcoxon comparisons, rank-biserial effects and Holm adjustment per metric. For example:

```bash
python -m ibc.compare --inputs "runs/*/metrics.csv" --reference full --metrics final_accuracy forgetting expenditure --unit seed
```

