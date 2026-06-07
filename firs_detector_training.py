import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from trigger_family import apply_trigger_fragment, get_df_dba_fragment_coords, trigger_config_from_params


def get_dba_coords(params, adversarial_index=-1):
    return get_df_dba_fragment_coords(params, adversarial_index)


def build_detector_training_batch(clean_images, params, adversarial_index=-1, mean=None, std=None):
    """Create clean negatives and trigger-family positives for FIRS detector training.

    The paper default is detector_train_trigger_type='df_dba', which samples
    color patch, texture, blended, and low-amplitude local fragments under the
    distributed DBA geometry.
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


def supervised_contrastive_loss(features, labels, temperature=0.1, positive_label=1):
    """Supervised contrastive regularizer for the statistical embedding.

    Anchors are suspicious samples and positives are other suspicious samples in
    the same detector-training batch. If a batch has no valid positive pair, the
    function returns a differentiable zero scalar.
    """
    if features is None:
        return None
    features = features.view(features.size(0), -1)
    if features.size(0) < 2:
        return features.sum() * 0.0

    labels = labels.view(-1)
    positive_mask = labels == positive_label
    if int(positive_mask.sum().item()) < 2:
        return features.sum() * 0.0

    z = F.normalize(features, dim=1)
    logits = torch.matmul(z, z.t()) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    valid_denominator = ~eye
    positive_pairs = positive_mask[:, None] & positive_mask[None, :] & valid_denominator
    anchor_mask = positive_pairs.any(dim=1)
    if int(anchor_mask.sum().item()) == 0:
        return features.sum() * 0.0

    exp_logits = torch.exp(logits) * valid_denominator.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positives_per_anchor = positive_pairs.float().sum(dim=1).clamp_min(1.0)
    mean_log_prob = (positive_pairs.float() * log_prob).sum(dim=1) / positives_per_anchor
    return -mean_log_prob[anchor_mask].mean()


def get_contrastive_weight(detector, params):
    """Return nonnegative lambda for BCE plus contrastive detector training."""
    attr_name = str(params.get("firs_contrastive_weight_attr", "firs_contrastive_xi"))
    if hasattr(detector, attr_name):
        return F.softplus(getattr(detector, attr_name))
    value = float(params.get("firs_contrastive_lambda", 0.0))
    return torch.tensor(value, device=next(detector.parameters()).device)


def attach_learnable_contrastive_weight(detector, initial_lambda=0.1, attr_name="firs_contrastive_xi"):
    """Attach xi so lambda=softplus(xi) can be optimized with the detector."""
    initial_lambda = max(float(initial_lambda), 1e-8)
    xi = math.log(math.exp(initial_lambda) - 1.0)
    if not hasattr(detector, attr_name):
        detector.register_parameter(attr_name, nn.Parameter(torch.tensor(float(xi))))
    return getattr(detector, attr_name)


def _forward_detector(detector, images):
    try:
        out = detector(images, return_features=True)
    except TypeError:
        out = detector(images)

    if isinstance(out, dict):
        logits = out["logits"]
        stat_features = out.get("statistical_features", None)
        return logits, stat_features
    if isinstance(out, (tuple, list)):
        logits = out[0]
        stat_features = out[1] if len(out) > 1 else None
        return logits, stat_features
    return out, None


def calibrate_recall_threshold(scores, labels, target_recall=0.95):
    """Choose the highest suspiciousness threshold satisfying target recall.

    Scores are suspicious probabilities or logits after any caller-side
    transformation. Samples with score >= threshold are rejected by FIRS.
    """
    scores = torch.as_tensor(scores, dtype=torch.float32).view(-1)
    labels = torch.as_tensor(labels, dtype=torch.long).view(-1)
    if scores.numel() != labels.numel():
        raise ValueError("scores and labels must have the same length")
    if scores.numel() == 0:
        raise ValueError("calibration requires at least one sample")

    target_recall = float(target_recall)
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")

    positives = labels == 1
    negatives = labels == 0
    if int(positives.sum().item()) == 0:
        raise ValueError("calibration requires at least one suspicious sample")

    candidates = torch.unique(scores).sort(descending=True).values
    best = None
    for threshold in candidates:
        pred_pos = scores >= threshold
        tp = (pred_pos & positives).sum().item()
        recall = tp / max(1, int(positives.sum().item()))
        if recall + 1e-12 >= target_recall:
            fp = (pred_pos & negatives).sum().item()
            fpr = fp / max(1, int(negatives.sum().item()))
            best = (threshold, recall, fpr, int(pred_pos.sum().item()))
            break

    if best is None:
        threshold = scores.min()
        pred_pos = scores >= threshold
        recall = float((pred_pos & positives).sum().item()) / max(1, int(positives.sum().item()))
        fpr = float((pred_pos & negatives).sum().item()) / max(1, int(negatives.sum().item()))
        best = (threshold, recall, fpr, int(pred_pos.sum().item()))

    threshold, recall, fpr, rejected = best
    return {
        "threshold": float(threshold.item()),
        "target_recall": target_recall,
        "validation_recall": float(recall),
        "validation_false_positive_rate": float(fpr),
        "rejected_samples": int(rejected),
        "total_samples": int(scores.numel()),
    }


def train_detector_step(detector, optimizer, clean_images, params, adversarial_index=-1, mean=None, std=None):
    detector.train()
    images, labels = build_detector_training_batch(
        clean_images, params, adversarial_index=adversarial_index, mean=mean, std=std
    )
    optimizer.zero_grad()
    logits, stat_features = _forward_detector(detector, images)
    logits = logits.view(-1)
    bce_loss = nn.BCEWithLogitsLoss()(logits, labels.float())
    contrastive = supervised_contrastive_loss(
        stat_features,
        labels,
        temperature=float(params.get("firs_contrastive_temperature", 0.1)),
    )
    if contrastive is None:
        contrastive = bce_loss * 0.0
    contrastive_weight = get_contrastive_weight(detector, params)
    loss = bce_loss + contrastive_weight * contrastive
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float()
        acc = float((pred == labels).float().mean().item())
    return {
        "loss": float(loss.detach().item()),
        "bce_loss": float(bce_loss.detach().item()),
        "contrastive_loss": float(contrastive.detach().item()),
        "contrastive_lambda": float(contrastive_weight.detach().item()),
        "accuracy": acc,
        "batch_size": int(images.size(0)),
        "positive_samples": int(labels.sum().item()),
        "detector_train_trigger_type": str(params.get("detector_train_trigger_type", "white_patch")),
    }
