This repository provides the anonymized implementation for the submitted paper.

The code implements FIRS experiments for federated learning backdoor defense. It includes training code, model definitions, experiment configuration files, sample-screening logic, trigger-family utilities, and lightweight debugging scripts. Datasets, checkpoints, logs, generated experiment outputs, and manuscript files are intentionally excluded from the repository.

## Repository Layout

```text
.
|-- main.py                     # experiment entry point
|-- image_train.py              # image federated training loop
|-- image_helper.py             # image dataset, attack, and filtering helpers
|-- firs_gate.py                # FIRS sample screening gate
|-- firs_detector_training.py   # detector training batch construction
|-- trigger_family.py           # trigger variants used by experiments
|-- models/                     # neural network definitions
|-- utils/                      # YAML configs and preprocessing utilities
|-- scripts/                    # smoke/debug scripts
`-- requirements.txt            # Python package list
```

## Environment Setup

Create a clean Python environment, install PyTorch for the target CUDA or CPU platform, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The project expects a standard Python scientific stack with PyTorch, torchvision, NumPy, pandas, scikit-learn, matplotlib, OpenCV, PyYAML, Visdom, and psutil.

## Data Preparation

MNIST and CIFAR-10 are downloaded automatically by torchvision when the corresponding experiments are launched.

For the LOAN experiment, download the public raw archive into `utils/`, then run:

```bash
cd utils
./process_loan_data.sh
cd ..
```

For Tiny ImageNet, download the public archive into `utils/`, then run:

```bash
cd utils
./process_tiny_data.sh
cd ..
```

Prepared datasets are stored under `data/`, which is ignored by git.

## Core Commands

Optional live visualization:

```bash
python -m visdom.server -p 8098
```

Run the main experiments with the provided YAML files:

```bash
python main.py --params utils/mnist_params.yaml
python main.py --params utils/cifar_params.yaml
python main.py --params utils/tiny_params.yaml
python main.py --params utils/loan_params.yaml
```

FIRS-specific options are configured in the YAML files, including `enable_firs_gate`, `pipeline_mode`, `prefilter_threshold`, `prefilter_apply_to`, and trigger-family options such as `detector_train_trigger_type`.

## DF-DBA Trigger Protocol

The trigger implementation in `trigger_family.py` follows the submitted DF-DBA setting. A global trigger is split into four local fragments, so no single malicious client needs the full trigger. The default geometry is generated from Trigger Size, Trigger Gap, and Trigger Location:

- MNIST: `4, 2, 0`
- CIFAR-10: `6, 3, 0`
- Tiny-ImageNet: `10, 2, 0`

Explicit YAML coordinates such as `0_poison_pattern` still take precedence. When they are absent, the helper `get_df_dba_fragment_coords` generates the paper geometry automatically.

The main DF-DBA protocol is available as `trigger_type: df_dba` or `detector_train_trigger_type: df_dba`. It samples the four main local trigger families:

- `color_patch`: solid colors sampled from white, gray, red, green, blue, and yellow.
- `texture`: checkerboard, stripe, and dot patterns.
- `blended`: alpha blending with alpha sampled from `0.1, 0.2, 0.3`.
- `low_amplitude`: additive perturbations with delta sampled from `4/255, 8/255, 12/255`.

Extended trigger-family analysis can use `frequency` and `warping`, or `df_dba_extended` to sample across all six families. Frequency fragments use local sinusoidal patterns with amplitudes from `4/255, 8/255, 12/255` and frequencies from `2, 4, 6`. Warping fragments apply local spatial displacement, with the default maximum displacement set to 2 pixels for MNIST/CIFAR-10 and 4 pixels for Tiny-ImageNet.

## FIRS Implementation

The detector in `models/model_resnet_grid.py` follows the submitted method:

- Global Semantic Encoder (GSE): a ResNet-18 feature extractor over normalized images.
- Tile-wise Statistical Encoder (TSE): a grid encoder that computes soft histograms, mean, variance, skewness, and kurtosis for local tiles.
- Fusion Screening Head (FSH): a binary screening head over concatenated semantic and statistical embeddings.

The detector training helper in `firs_detector_training.py` constructs paired clean and triggered samples. Its training step optimizes:

```text
BCEWithLogitsLoss + lambda * supervised_contrastive_loss(statistical_embedding)
```

Use `firs_contrastive_lambda` for a fixed nonnegative weight, or attach a learnable parameter with `attach_learnable_contrastive_weight(detector)` before constructing the optimizer so that `lambda = softplus(xi)` is optimized with the detector.

Threshold selection is recall-oriented. The helper `calibrate_recall_threshold(scores, labels, target_recall)` chooses the highest suspiciousness threshold that satisfies the requested validation recall for triggered samples, then reports the corresponding false-positive rate. The resulting threshold is used as `prefilter_threshold` for local sample screening. The deployed FIRS gate applies a frozen detector and the calibrated threshold before local optimization; it does not use online score histories or a per-batch rejection budget.

## Reproducing Tables

Use the configuration files in `utils/` to reproduce the main comparisons:

```bash
python main.py --params utils/mnist_params_fldetector.yaml
python main.py --params utils/mnist_params_flip.yaml
python main.py --params utils/mnist_params_leadfl.yaml

python main.py --params utils/cifar_params_fldetector.yaml
python main.py --params utils/cifar_params_flip.yaml
python main.py --params utils/cifar_params_leadfl.yaml

python main.py --params utils/tiny_params_fldetector.yaml
python main.py --params utils/tiny_params_flip.yaml
python main.py --params utils/tiny_params_leadfl.yaml
```

Additional robust aggregation baselines are provided through the `*_foolsgold.yaml`, `*_rfa.yaml`, `tiny_params_fedavg.yaml`, and `tiny_params_fltrust.yaml` files.

The helper script `gc.py` can summarize generated result directories when CSV outputs are available:

```bash
python gc.py --root saved_models --out results
```

Run lightweight FIRS smoke checks without datasets:

```bash
python scripts/debug_trigger_family.py
python scripts/debug_cross_trigger_smoke.py
python scripts/debug_detector_training_hook.py
python scripts/debug_firs_gate_pipeline.py
```

## Expected Outputs

Training runs create timestamped result folders containing logs, copied parameters, and CSV summaries such as:

- `train_result.csv`
- `test_result.csv`
- `posiontest_result.csv`
- `poisontriggertest_result.csv`
- `weight_result.csv` or `scale_result.csv` when enabled

These files are generated locally under ignored artifact directories such as `saved_models/` or `results/`.

## Local Artifacts

The following paths and file types are intentionally ignored:

- `data/`
- `saved_models/`
- `results/`
- `debug_outputs/`
- `logs/`, `outputs/`, `checkpoints/`, `runs/`, `wandb/`
- model checkpoint files such as `*.pt`, `*.pth`, and `*.ckpt`
- Python caches, notebook checkpoints, OS metadata, and log files
