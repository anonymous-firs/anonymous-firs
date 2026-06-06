import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trigger_family import apply_trigger_fragment, trigger_config_from_params


def main():
    image = torch.zeros(3, 32, 32)
    coords = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]

    params = {
        "detector_train_trigger_type": "randomized_patch",
        "attack_eval_trigger_type": "colored_patch",
        "trigger_color": "green",
        "trigger_alpha": 0.5,
        "trigger_jitter": 1,
        "trigger_size_delta": 1,
    }
    detector_train_cfg = trigger_config_from_params(params, evaluation=False)
    attack_eval_cfg = trigger_config_from_params(params, evaluation=True)

    train_variant = apply_trigger_fragment(image, coords, detector_train_cfg)
    eval_variant = apply_trigger_fragment(image, coords, attack_eval_cfg)

    print("detector_train_trigger_type=randomized_patch", tuple(train_variant.shape), float(train_variant.max()))
    print("attack_eval_trigger_type=colored_patch", tuple(eval_variant.shape), float(eval_variant.max()))
    print("variants_equal=", bool(torch.equal(train_variant, eval_variant)))


if __name__ == "__main__":
    main()
