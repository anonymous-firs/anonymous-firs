import os
import sys

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firs_detector_training import build_detector_training_batch, train_detector_step


class TinyDetector(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        n = 1
        for dim in input_shape:
            n *= dim
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(n, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_params():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "utils",
        "debug_firs_gate_minimal.yaml",
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    torch.manual_seed(1)
    params = load_params()
    clean = torch.zeros(4, 3, 8, 10)

    detector_batch, detector_labels = build_detector_training_batch(
        clean, params, adversarial_index=0
    )
    model = TinyDetector(clean.shape[1:])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    metrics = train_detector_step(model, optimizer, clean, params, adversarial_index=0)

    print("detector_batch_shape=", tuple(detector_batch.shape))
    print("detector_labels=", detector_labels.tolist())
    print("train_step_metrics=", metrics)


if __name__ == "__main__":
    main()
