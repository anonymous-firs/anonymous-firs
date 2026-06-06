import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trigger_family import TriggerConfig, apply_trigger_fragment


def main():
    image = torch.zeros(3, 32, 32)
    coords = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]
    families = [
        TriggerConfig("white_patch"),
        TriggerConfig("colored_patch", color="red"),
        TriggerConfig("low_alpha_patch", alpha=0.2),
        TriggerConfig("blended_patch", color="blue", alpha=0.35),
        TriggerConfig("randomized_patch", jitter=1, size_delta=1),
    ]

    for cfg in families:
        out = apply_trigger_fragment(image, coords, cfg)
        print(
            cfg.trigger_type,
            "shape=", tuple(out.shape),
            "dtype=", out.dtype,
            "min=", float(out.min()),
            "max=", float(out.max()),
            "changed=", int((out != image).sum().item()),
        )


if __name__ == "__main__":
    main()
