from dataclasses import dataclass
import math
import random
from typing import Iterable, List, Optional, Sequence, Tuple

import torch


Coord = Tuple[int, int]

DF_DBA_MAIN_FAMILIES = ("color_patch", "texture", "blended", "low_amplitude")
DF_DBA_EXTENDED_FAMILIES = DF_DBA_MAIN_FAMILIES + ("frequency", "warping")
DF_DBA_COLORS = ("white", "gray", "red", "green", "blue", "yellow")
DF_DBA_TEXTURES = ("checkerboard", "stripe", "dot")
DF_DBA_ALPHAS = (0.1, 0.2, 0.3)
DF_DBA_DELTAS = (4.0 / 255.0, 8.0 / 255.0, 12.0 / 255.0)
DF_DBA_FREQUENCIES = (2, 4, 6)


@dataclass
class TriggerConfig:
    trigger_type: str = "df_dba"
    color: str = "white"
    alpha: float = 0.2
    intensity: float = 1.0
    jitter: int = 0
    size_delta: int = 0
    randomize: bool = False
    texture: str = "checkerboard"
    delta: float = 8.0 / 255.0
    frequency: int = 4
    displacement: int = 2


def trigger_config_from_params(params, evaluation: bool = False) -> TriggerConfig:
    key = "attack_eval_trigger_type" if evaluation else "detector_train_trigger_type"
    trigger_type = params.get(key) or params.get("trigger_type", "df_dba")
    return TriggerConfig(
        trigger_type=str(trigger_type).lower(),
        color=str(params.get("trigger_color", "white")).lower(),
        alpha=float(params.get("trigger_alpha", 0.2)),
        intensity=float(params.get("trigger_intensity", 1.0)),
        jitter=int(params.get("trigger_jitter", 0)),
        size_delta=int(params.get("trigger_size_delta", 0)),
        randomize=bool(params.get("trigger_randomize", False)),
        texture=str(params.get("trigger_texture", "checkerboard")).lower(),
        delta=float(params.get("trigger_delta", 8.0 / 255.0)),
        frequency=int(params.get("trigger_frequency", 4)),
        displacement=int(params.get("trigger_displacement", _default_displacement(params))),
    )


def get_df_dba_fragment_coords(params, adversarial_index=-1) -> List[Coord]:
    """Return DBA local-fragment coordinates from params or paper geometry."""
    trigger_num = int(params.get("trigger_num", 4))
    if _has_explicit_patterns(params, trigger_num):
        if adversarial_index == -1:
            coords = []
            for i in range(trigger_num):
                coords += _as_coords(params[f"{i}_poison_pattern"])
            return coords
        return _as_coords(params[f"{int(adversarial_index)}_poison_pattern"])

    fragments = generate_df_dba_geometry(
        trigger_size=int(params.get("df_dba_trigger_size", _default_trigger_size(params))),
        trigger_gap=int(params.get("df_dba_trigger_gap", _default_trigger_gap(params))),
        trigger_location=int(params.get("df_dba_trigger_location", 0)),
        fragment_count=trigger_num,
    )
    if adversarial_index == -1:
        return [coord for fragment in fragments for coord in fragment]
    return fragments[int(adversarial_index) % len(fragments)]


def generate_df_dba_geometry(
    trigger_size: int,
    trigger_gap: int,
    trigger_location: int = 0,
    fragment_count: int = 4,
) -> List[List[Coord]]:
    """Generate the four local fragments used by the paper DF-DBA geometry.

    The geometry follows Trigger Size, Trigger Gap, and Trigger Location:
    MNIST uses 4/2/0, CIFAR-10 uses 6/3/0, and Tiny-ImageNet uses 10/2/0.
    """
    size = max(1, int(trigger_size))
    gap = max(0, int(trigger_gap))
    loc = max(0, int(trigger_location))
    block_h = max(1, size // 5)
    starts = [
        (loc, loc),
        (loc, loc + size + gap),
        (loc + block_h + gap, loc),
        (loc + block_h + gap, loc + size + gap),
    ]

    fragments = []
    for row0, col0 in starts[: max(1, int(fragment_count))]:
        coords = []
        for row in range(row0, row0 + block_h):
            for col in range(col0, col0 + size):
                coords.append((row, col))
        fragments.append(coords)
    return fragments


def apply_trigger_fragment(
    image: torch.Tensor,
    coords: Iterable[Sequence[int]],
    config: Optional[TriggerConfig] = None,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Apply one DF-DBA local fragment while preserving shape, dtype, and device."""
    cfg = _sample_config(config or TriggerConfig())

    img = image.clone()
    if img.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got shape {tuple(img.shape)}")

    C, H, W = img.shape
    coords2 = _prepare_coords(coords, H, W, cfg.jitter, cfg.size_delta)
    if not coords2:
        return img

    if cfg.trigger_type == "color_patch":
        _apply_color_patch(img, coords2, cfg, mean, std)
    elif cfg.trigger_type == "texture":
        _apply_texture(img, coords2, cfg, mean, std)
    elif cfg.trigger_type == "blended":
        _apply_blended(img, coords2, cfg, mean, std)
    elif cfg.trigger_type == "low_amplitude":
        _apply_low_amplitude(img, coords2, cfg, mean, std)
    elif cfg.trigger_type == "frequency":
        _apply_frequency(img, coords2, cfg, mean, std)
    elif cfg.trigger_type == "warping":
        _apply_warping(img, coords2, cfg)
    else:
        raise ValueError(f"Unknown trigger_type: {cfg.trigger_type}")

    return img if mean is not None else img.clamp(0.0, 1.0)


def _sample_config(cfg: TriggerConfig) -> TriggerConfig:
    trigger_type = _canonical_trigger_type(cfg.trigger_type)
    should_sample_family = trigger_type in ("df_dba", "df_dba_extended", "randomized_patch")
    if should_sample_family:
        families = DF_DBA_EXTENDED_FAMILIES if trigger_type == "df_dba_extended" else DF_DBA_MAIN_FAMILIES
        trigger_type = random.choice(families)

    if not cfg.randomize and not should_sample_family:
        return TriggerConfig(
            trigger_type=trigger_type,
            color=cfg.color,
            alpha=cfg.alpha,
            intensity=cfg.intensity,
            jitter=cfg.jitter,
            size_delta=cfg.size_delta,
            randomize=False,
            texture=cfg.texture,
            delta=cfg.delta,
            frequency=cfg.frequency,
            displacement=cfg.displacement,
        )

    return TriggerConfig(
        trigger_type=trigger_type,
        color=random.choice(DF_DBA_COLORS) if trigger_type in ("color_patch", "blended") else cfg.color,
        alpha=random.choice(DF_DBA_ALPHAS) if trigger_type == "blended" else cfg.alpha,
        intensity=cfg.intensity,
        jitter=random.randint(-abs(cfg.jitter), abs(cfg.jitter)) if cfg.jitter else 0,
        size_delta=random.randint(0, max(0, cfg.size_delta)) if cfg.size_delta else 0,
        randomize=False,
        texture=random.choice(DF_DBA_TEXTURES) if trigger_type == "texture" else cfg.texture,
        delta=random.choice(DF_DBA_DELTAS) if trigger_type in ("low_amplitude", "frequency") else cfg.delta,
        frequency=random.choice(DF_DBA_FREQUENCIES) if trigger_type == "frequency" else cfg.frequency,
        displacement=cfg.displacement,
    )


def _canonical_trigger_type(trigger_type: str) -> str:
    aliases = {
        "white_patch": "color_patch",
        "colored_patch": "color_patch",
        "low_alpha_patch": "blended",
        "blended_patch": "blended",
        "df-dba": "df_dba",
        "df_dba_main": "df_dba",
        "df_dba_color": "color_patch",
        "df_dba_texture": "texture",
        "df_dba_blended": "blended",
        "df_dba_low_amplitude": "low_amplitude",
        "df_dba_frequency": "frequency",
        "df_dba_warping": "warping",
        "texture_patch": "texture",
        "low-amplitude": "low_amplitude",
        "low_amplitude_patch": "low_amplitude",
        "frequency_trigger": "frequency",
        "warping_trigger": "warping",
    }
    return aliases.get(str(trigger_type).lower(), str(trigger_type).lower())


def _apply_color_patch(img: torch.Tensor, coords: Sequence[Coord], cfg: TriggerConfig, mean, std):
    C = img.shape[0]
    patch_color = _color_tensor(cfg.color, C, img.device, img.dtype, cfg.intensity)
    patch_color = _to_model_domain(patch_color, mean, std, C, img.device, img.dtype)
    for row, col in coords:
        img[:, row, col] = patch_color


def _apply_blended(img: torch.Tensor, coords: Sequence[Coord], cfg: TriggerConfig, mean, std):
    C = img.shape[0]
    alpha = max(0.0, min(1.0, float(cfg.alpha)))
    patch_color = _color_tensor(cfg.color, C, img.device, img.dtype, cfg.intensity)
    patch_color = _to_model_domain(patch_color, mean, std, C, img.device, img.dtype)
    for row, col in coords:
        img[:, row, col] = (1.0 - alpha) * img[:, row, col] + alpha * patch_color


def _apply_low_amplitude(img: torch.Tensor, coords: Sequence[Coord], cfg: TriggerConfig, mean, std):
    C = img.shape[0]
    delta = _delta_tensor(cfg.delta, C, img.device, img.dtype, mean, std)
    for row, col in coords:
        img[:, row, col] = img[:, row, col] + delta


def _apply_frequency(img: torch.Tensor, coords: Sequence[Coord], cfg: TriggerConfig, mean, std):
    C = img.shape[0]
    rows = [p[0] for p in coords]
    cols = [p[1] for p in coords]
    row0, col0 = min(rows), min(cols)
    height = max(1, max(rows) - row0 + 1)
    width = max(1, max(cols) - col0 + 1)
    delta = _delta_tensor(cfg.delta, C, img.device, img.dtype, mean, std)
    freq = max(1, int(cfg.frequency))
    for row, col in coords:
        phase_x = (row - row0 + 0.5) / height
        phase_y = (col - col0 + 0.5) / width
        wave = math.sin(2.0 * math.pi * freq * phase_x) + math.cos(2.0 * math.pi * freq * phase_y)
        img[:, row, col] = img[:, row, col] + 0.5 * wave * delta


def _apply_texture(img: torch.Tensor, coords: Sequence[Coord], cfg: TriggerConfig, mean, std):
    C = img.shape[0]
    rows = [p[0] for p in coords]
    cols = [p[1] for p in coords]
    row0, col0 = min(rows), min(cols)
    light = _to_model_domain(
        _color_tensor("white", C, img.device, img.dtype, cfg.intensity),
        mean,
        std,
        C,
        img.device,
        img.dtype,
    )
    dark = _to_model_domain(
        _color_tensor("black", C, img.device, img.dtype, 0.0),
        mean,
        std,
        C,
        img.device,
        img.dtype,
    )
    for row, col in coords:
        local_r, local_c = row - row0, col - col0
        if cfg.texture == "stripe":
            use_light = (local_c % 2) == 0
        elif cfg.texture == "dot":
            use_light = (local_r % 2 == 0) and (local_c % 2 == 0)
        else:
            use_light = ((local_r + local_c) % 2) == 0
        img[:, row, col] = light if use_light else dark


def _apply_warping(img: torch.Tensor, coords: Sequence[Coord], cfg: TriggerConfig):
    rows = [p[0] for p in coords]
    cols = [p[1] for p in coords]
    row0, row1 = min(rows), max(rows)
    col0, col1 = min(cols), max(cols)
    region = img[:, row0 : row1 + 1, col0 : col1 + 1].clone()
    if region.numel() == 0:
        return
    limit = max(1, int(cfg.displacement))
    dx = random.randint(-limit, limit)
    dy = random.randint(-limit, limit)
    if dx == 0 and dy == 0:
        dy = 1
    img[:, row0 : row1 + 1, col0 : col1 + 1] = torch.roll(region, shifts=(dx, dy), dims=(1, 2))


def _prepare_coords(coords: Iterable[Sequence[int]], H: int, W: int, jitter: int, size_delta: int) -> List[Coord]:
    out = set()
    base = []
    jitter = abs(int(jitter))
    dx = random.randint(-jitter, jitter) if jitter > 0 else 0
    dy = random.randint(-jitter, jitter) if jitter > 0 else 0
    radius = max(0, int(size_delta))
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
    color = str(color).lower()
    if color == "random":
        vals = [random.random() * intensity for _ in range(max(3, C))]
    elif color == "gray":
        vals = [0.5 * intensity, 0.5 * intensity, 0.5 * intensity]
    elif color == "red":
        vals = [intensity, 0.0, 0.0]
    elif color == "green":
        vals = [0.0, intensity, 0.0]
    elif color == "blue":
        vals = [0.0, 0.0, intensity]
    elif color == "yellow":
        vals = [intensity, intensity, 0.0]
    elif color == "black":
        vals = [0.0, 0.0, 0.0]
    else:
        vals = [intensity, intensity, intensity]

    if C == 1:
        vals = [sum(vals[:3]) / 3.0]
    elif C > len(vals):
        vals = vals + [vals[-1]] * (C - len(vals))
    return torch.tensor(vals[:C], device=device, dtype=dtype)


def _delta_tensor(delta: float, C: int, device, dtype, mean, std) -> torch.Tensor:
    values = torch.full((C,), float(delta), device=device, dtype=dtype)
    if mean is None or std is None:
        return values
    if not isinstance(std, (list, tuple)):
        std = [std] * C
    std_t = torch.tensor(std[:C], device=device, dtype=dtype).clamp_min(1e-12)
    return values / std_t


def _to_model_domain(values: torch.Tensor, mean, std, C: int, device, dtype) -> torch.Tensor:
    if mean is None or std is None:
        return values
    if not isinstance(mean, (list, tuple)):
        mean = [mean] * C
    if not isinstance(std, (list, tuple)):
        std = [std] * C
    mean_t = torch.tensor(mean[:C], device=device, dtype=dtype)
    std_t = torch.tensor(std[:C], device=device, dtype=dtype).clamp_min(1e-12)
    return (values - mean_t) / std_t


def _has_explicit_patterns(params, trigger_num: int) -> bool:
    return all(f"{i}_poison_pattern" in params for i in range(trigger_num))


def _as_coords(coords: Iterable[Sequence[int]]) -> List[Coord]:
    return [(int(x), int(y)) for x, y in coords]


def _default_trigger_size(params) -> int:
    dataset = str(params.get("type", "")).lower()
    if "tiny" in dataset:
        return 10
    if "cifar" in dataset:
        return 6
    return 4


def _default_trigger_gap(params) -> int:
    dataset = str(params.get("type", "")).lower()
    if "cifar" in dataset:
        return 3
    return 2


def _default_displacement(params) -> int:
    dataset = str(params.get("type", "")).lower()
    return 4 if "tiny" in dataset else 2
