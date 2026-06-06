This repository provides the anonymized implementation for the submitted paper.

The code implements FIRS experiments for federated learning backdoor defense. It includes training code, model definitions, experiment configuration files, sample-screening logic, trigger-family utilities, and lightweight debugging scripts. Datasets, checkpoints, logs, and generated experiment outputs are intentionally excluded from the repository.

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
