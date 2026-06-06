import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firs_gate import FIRSScreeningGate


class FakeHelper:
    def __init__(self):
        self.params = {
            "enable_firs_gate": True,
            "pipeline_mode": "controlled",
            "prefilter_threshold": 0.5,
            "prefilter_apply_to": "poisoned",
        }

    def _prefilter__score_tensor_batch(self, imgs):
        return imgs.flatten(1).mean(dim=1).cpu()


def main():
    helper = FakeHelper()
    gate = FIRSScreeningGate(helper)
    data = torch.stack([
        torch.zeros(3, 4, 4),
        torch.ones(3, 4, 4),
        torch.full((3, 4, 4), 0.25),
    ])
    targets = torch.tensor([0, 1, 2])

    kept_data, kept_targets, meta = gate.screen_batch(
        data, targets, client_id=17, batch_id=0, is_poisoned_client=True
    )
    print("metadata=", meta.to_dict())
    print("kept_shape=", tuple(kept_data.shape))
    print("kept_targets=", kept_targets.tolist())


if __name__ == "__main__":
    main()
