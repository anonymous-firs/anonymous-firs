from dataclasses import dataclass
import random
from typing import Iterable, List, Optional, Sequence, Tuple

import torch


Coord = Tuple[int, int]


@dataclass
class TriggerConfig:
    trigger_type: str = "white_patch"
    color: str = "white"
    alpha: float = 1.0
    intensity: float = 1.0
    jitter: int = 0
    size_delta: int = 0
    randomize: bool = False


def trigger_config_from_params(params, evaluation: bool = False) -> TriggerConfig:
    key = "attack_eval_trigger_type" if evaluation else "detector_train_trigger_type"
    trigger_type = params.get(key) or params.get("trigger_type", "white_patch")
    return TriggerConfig(
        trigger_type=str(trigger_type).lower(),
        color=str(params.get("trigger_color", "white")).lower(),
        alpha=float(params.get("trigger_alpha", 1.0)),
        intensity=float(params.get("trigger_intensity", 1.0)),
        jitter=int(params.get("trigger_jitter", 0)),
        size_delta=int(params.get("trigger_size_delta", 0)),
        randomize=bool(params.get("trigger_randomize", False)),
    )


def apply_trigger_fragment(
    image: torch.Tensor,
    coords: Iterable[Sequence[int]],
    config: Optional[TriggerConfig] = None,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Apply one DBA fragment while preserving shape, dtype, and device."""
    cfg = config or TriggerConfig()
    if cfg.trigger_type in ("frequency_trigger", "warping_trigger"):
        raise NotImplementedError(f"{cfg.trigger_type} is reserved but not implemented in this smoke path.")

    img = image.clone()
    if img.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got shape {tuple(img.shape)}")

    C, H, W = img.shape
    sampled = _sample_config(cfg)
    coords2 = _prepare_coords(coords, H, W, sampled.jitter, sampled.size_delta)
    if not coords2:
        return img

    patch_color = _color_tensor(sampled.color, C, img.device, img.dtype, sampled.intensity)
    patch_color = _to_model_domain(patch_color, mean, std, C, img.device, img.dtype)

    if sampled.trigger_type in ("white_patch", "colored_patch"):
        for x, y in coords2:
            img[:, x, y] = patch_color
    elif sampled.trigger_type in ("low_alpha_patch", "blended_patch"):
        alpha = max(0.0, min(1.0, sampled.alpha))
        for x, y in coords2:
            img[:, x, y] = (1.0 - alpha) * img[:, x, y] + alpha * patch_color
    elif sampled.trigger_type == "randomized_patch":
        # _sample_config resolves randomized_patch to a concrete patch family.
        return apply_trigger_fragment(img, coords2, sampled, mean=mean, std=std)
    else:
        raise ValueError(f"Unknown trigger_type: {sampled.trigger_type}")

    return img if mean is not None else img.clamp(0.0, 1.0)


def _sample_config(cfg: TriggerConfig) -> TriggerConfig:
    if cfg.trigger_type != "randomized_patch" and not cfg.randomize:
        return cfg

    trigger_type = random.choice(["white_patch", "colored_patch", "low_alpha_patch", "blended_patch"])
    color = random.choice(["white", "red", "green", "blue", "random"])
    alpha = random.uniform(0.15, max(0.2, min(1.0, cfg.alpha)))
    intensity = random.uniform(0.35, max(0.36, min(1.0, cfg.intensity)))
    jitter = random.randint(-abs(cfg.jitter), abs(cfg.jitter)) if cfg.jitter else 0
    size_delta = random.randint(0, max(0, cfg.size_delta))
    return TriggerConfig(trigger_type, color, alpha, intensity, abs(jitter), size_delta, False)


def _prepare_coords(coords: Iterable[Sequence[int]], H: int, W: int, jitter: int, size_delta: int) -> List[Coord]:
    out = set()
    base = []
    dx = random.randint(-jitter, jitter) if jitter > 0 else 0
    dy = random.randint(-jitter, jitter) if jitter > 0 else 0
    radius = max(0, size_delta)
    for raw in coords:
        base_x, base_y = int(raw[0]), int(raw[1])
        if 0 <= base_x < H and 0 <= base_y < W:
            base.append((base_x, base_y))
        x, y = base_x + dx, base_y + dy
        for ox in range(-radius, radius + 1):
            for oy in range(-radius, radius + 1):
                xx, yy = x + ox, y + oy
                if 0 <= xx < H and 0 <= yy < W:
                    out.add((xx, yy))
    if not out:
        out.update(base)
    return sorted(out)


def _color_tensor(color: str, C: int, device, dtype, intensity: float) -> torch.Tensor:
    intensity = max(0.0, min(1.0, float(intensity)))
    if color == "random":
        vals = [random.random() * intensity for _ in range(max(3, C))]
    elif color == "red":
        vals = [intensity, 0.0, 0.0]
    elif color == "green":
        vals = [0.0, intensity, 0.0]
    elif color == "blue":
        vals = [0.0, 0.0, intensity]
    else:
        vals = [intensity, intensity, intensity]

    if C == 1:
        vals = [sum(vals[:3]) / 3.0]
    elif C > len(vals):
        vals = vals + [vals[-1]] * (C - len(vals))
    return torch.tensor(vals[:C], device=device, dtype=dtype)


def _to_model_domain(values: torch.Tensor, mean, std, C: int, device, dtype) -> torch.Tensor:
    if mean is None or std is None:
        return values
    if not isinstance(mean, (list, tuple)):
        mean = [mean] * C
    if not isinstance(std, (list, tuple)):
        std = [std] * C
    mean_t = torch.tensor(mean[:C], device=device, dtype=dtype)
    std_t = torch.tensor(std[:C], device=device, dtype=dtype)
    return (values - mean_t) / std_t
