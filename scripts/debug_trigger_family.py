import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trigger_family import TriggerConfig, apply_trigger_fragment


def main():
    image = torch.linspace(0.0, 1.0, 3 * 32 * 32).view(3, 32, 32)
    coords = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]
    families = [
        TriggerConfig("color_patch", color="red"),
        TriggerConfig("texture", texture="checkerboard"),
        TriggerConfig("blended", color="blue", alpha=0.2),
        TriggerConfig("low_amplitude", delta=8.0 / 255.0),
        TriggerConfig("frequency", delta=8.0 / 255.0, frequency=4),
        TriggerConfig("warping", displacement=2),
        TriggerConfig("df_dba", jitter=1, size_delta=1),
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
