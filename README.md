# IBC

Implementation of **Interference-budgeted consolidation for continual representation learning.**

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3 and SciPy 1.15.3. Verification covers Python syntax, JSON/TOML and README CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Export [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) or [CORe50](https://vlomonaco.github.io/core50/) as `x: uint8 (N,3,H,W)`, `y`, `task`, `session` and `split: train/test`. Preserve acquisition sessions. For CIFAR, session IDs may identify disjoint source partitions. Predictions cover every class observed so far. The first source minibatch of each stage is reserved for monitoring.

```bash
python -m ibc prepare --input data/stream.npz --config config.json --output runs/full
```

## Training and inference

```bash
python -m ibc train --config config.json --output runs/full
```

`config.json` records the method settings and seed `13`; the resolved configuration is saved as `runs/full/settings.json`. Keep separate output directories for different seeds and variants.

## Evaluation

```bash
python -m ibc evaluate --config config.json --output runs/full
```

## Ablations

Supported controls: `full`, `uniform`, `strict`, `no-ledger`, `random`, `frozen-sketch`, `finetune`, `global-cap`.

Repeat the same data and configuration in a separate directory, for example:

```bash
python -m ibc prepare --input data/stream.npz --config config.json --output runs/uniform --variant uniform
python -m ibc train --config config.json --output runs/uniform --variant uniform
python -m ibc evaluate --config config.json --output runs/uniform --variant uniform
```

## Outputs

All main outputs are written below `runs/full/`.

`updates.csv` records requests, actual expenditure, remaining resource, progress and disturbance. `accuracy.csv` is the stage/task accuracy matrix with feature drift. `metrics.csv` reports final average accuracy, forgetting, new-task accuracy, feature drift and total sensitive motion. Retained anchor IDs are serialized per stage.

```bash
python -m ibc plot --config config.json --output runs/full
```
