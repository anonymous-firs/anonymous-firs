import torch
import torch.nn as nn

from trigger_family import apply_trigger_fragment, trigger_config_from_params


def get_dba_coords(params, adversarial_index=-1):
    if adversarial_index == -1:
        coords = []
        for i in range(int(params.get("trigger_num", 1))):
            coords += params[f"{i}_poison_pattern"]
        return coords
    return params[f"{adversarial_index}_poison_pattern"]


def build_detector_training_batch(clean_images, params, adversarial_index=-1, mean=None, std=None):
    """Create clean negatives and trigger-family positives for FIRS detector training.

    Defaults preserve the original white-patch detector mode through
    detector_train_trigger_type='white_patch'. Set it to 'randomized_patch' to
    make positive samples vary by color, alpha, jitter, intensity, and size.
    """
    cfg = trigger_config_from_params(params, evaluation=False)
    coords = get_dba_coords(params, adversarial_index)
    positives = clean_images.clone()
    for i in range(positives.size(0)):
        positives[i] = apply_trigger_fragment(positives[i], coords, cfg, mean=mean, std=std)

    images = torch.cat([clean_images, positives], dim=0)
    labels = torch.cat([
        torch.zeros(clean_images.size(0), device=clean_images.device),
        torch.ones(positives.size(0), device=clean_images.device),
    ], dim=0)
    return images, labels


def train_detector_step(detector, optimizer, clean_images, params, adversarial_index=-1, mean=None, std=None):
    detector.train()
    images, labels = build_detector_training_batch(
        clean_images, params, adversarial_index=adversarial_index, mean=mean, std=std
    )
    optimizer.zero_grad()
    logits = detector(images)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    logits = logits.view(-1)
    loss = nn.BCEWithLogitsLoss()(logits, labels.float())
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float()
        acc = float((pred == labels).float().mean().item())
    return {
        "loss": float(loss.detach().item()),
        "accuracy": acc,
        "batch_size": int(images.size(0)),
        "positive_samples": int(labels.sum().item()),
        "detector_train_trigger_type": str(params.get("detector_train_trigger_type", "white_patch")),
    }
