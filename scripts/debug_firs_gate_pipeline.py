import os
import sys

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firs_gate import FIRSScreeningGate
from trigger_family import apply_trigger_fragment, trigger_config_from_params


class FakePipelineHelper:
    def __init__(self, params):
        self.params = params

    def _prefilter__score_tensor_batch(self, imgs):
        # Deliberately simple risk score for smoke testing: trigger-heavy samples
        # become high-risk, clean darker samples stay low-risk.
        return imgs.flatten(1).mean(dim=1).cpu()


def load_params():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "utils",
        "debug_firs_gate_minimal.yaml",
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def poison_prefix(images, targets, params, adversarial_index=0):
    cfg = trigger_config_from_params(params, evaluation=True)
    coords = params[f"{adversarial_index}_poison_pattern"]
    out_x = images.clone()
    out_y = targets.clone()
    poison_k = min(int(params["poisoning_per_batch"]), images.size(0))
    for i in range(poison_k):
        out_x[i] = apply_trigger_fragment(out_x[i], coords, cfg)
        out_y[i] = int(params["poison_label_swap"])
    return out_x, out_y, poison_k


def main():
    torch.manual_seed(1)
    params = load_params()
    helper = FakePipelineHelper(params)
    gate = FIRSScreeningGate(helper)

    raw_x = torch.zeros(4, 3, 8, 10)
    raw_y = torch.tensor([0, 1, 3, 4])
    poisoned_x, poisoned_y, poison_k = poison_prefix(raw_x, raw_y, params, adversarial_index=0)

    filtered_x, filtered_y, meta = gate.screen_batch(
        poisoned_x,
        poisoned_y,
        client_id=17,
        batch_id=0,
        is_poisoned_client=True,
    )

    model = nn.Sequential(nn.Flatten(), nn.Linear(filtered_x[0].numel(), 10))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss = nn.functional.cross_entropy(model(filtered_x), filtered_y.long())
    loss.backward()
    optimizer.step()

    print("raw_samples=", int(raw_x.size(0)))
    print("poisoned_prefix=", poison_k)
    print("gate_metadata=", meta.to_dict())
    print("optimizer_batch_size=", int(filtered_x.size(0)))
    print("loss=", float(loss.detach()))


if __name__ == "__main__":
    main()
