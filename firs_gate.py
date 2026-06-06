from dataclasses import dataclass, asdict
from typing import Optional

import torch


@dataclass
class ScreeningMetadata:
    enabled: bool
    pipeline_mode: str
    total_samples: int
    accepted_samples: int
    rejected_samples: int
    rejection_ratio: float
    mean_score: Optional[float]
    max_score: Optional[float]
    p95_score: Optional[float]
    threshold: float
    client_id: Optional[int] = None
    batch_id: Optional[int] = None
    reason: str = ""

    def to_dict(self):
        return asdict(self)


class FIRSScreeningGate:
    """Training-pipeline screening gate for sample-level poisoning defenses.

    This gate models a controlled local training pipeline. It does not claim to
    stop arbitrary client code compromise; a fully compromised client can be
    simulated with pipeline_mode='bypass' and needs separate system defenses.
    """

    def __init__(self, helper):
        self.helper = helper
        params = helper.params
        self.enabled = bool(params.get("enable_firs_gate", False))
        self.pipeline_mode = str(params.get("pipeline_mode", "controlled")).lower()
        self.threshold = float(params.get("prefilter_threshold", 0.5))
        self.apply_to = str(params.get("prefilter_apply_to", "poisoned")).lower()
        self.keep_one_on_empty = bool(params.get("firs_gate_keep_one_on_empty", True))

    def should_screen(self, is_poisoned_client: bool) -> bool:
        if not self.enabled:
            return False
        if self.pipeline_mode == "bypass":
            return False
        if self.pipeline_mode != "controlled":
            raise ValueError(f"Unknown pipeline_mode: {self.pipeline_mode}")
        return self.apply_to == "all" or is_poisoned_client

    def screen_batch(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        client_id=None,
        batch_id=None,
        is_poisoned_client: bool = False,
    ):
        total = int(data.size(0)) if torch.is_tensor(data) else 0
        if total == 0:
            meta = self._metadata(False, total, total, None, client_id, batch_id, "empty")
            return data, targets, meta

        if not self.should_screen(is_poisoned_client):
            reason = "disabled" if not self.enabled else self.pipeline_mode
            if self.enabled and self.pipeline_mode == "controlled":
                reason = f"apply_to={self.apply_to}"
            meta = self._metadata(False, total, total, None, client_id, batch_id, reason)
            return data, targets, meta

        scores = self.helper._prefilter__score_tensor_batch(data.detach())
        keep = scores < self.threshold
        accepted = int(keep.sum().item())
        if accepted == 0 and self.keep_one_on_empty:
            keep[torch.argmin(scores)] = True
            accepted = 1

        keep_idx = keep.to(device=data.device, dtype=torch.bool)
        filtered_data = data[keep_idx]
        filtered_targets = targets[keep_idx]
        meta = self._metadata(True, total, accepted, scores, client_id, batch_id, "screened")
        return filtered_data, filtered_targets, meta

    def _metadata(self, enabled, total, accepted, scores, client_id, batch_id, reason):
        rejected = max(0, int(total) - int(accepted))
        if scores is None or len(scores) == 0:
            mean_score = max_score = p95_score = None
        else:
            scores_f = scores.float()
            mean_score = float(scores_f.mean().item())
            max_score = float(scores_f.max().item())
            q = min(max(len(scores_f) - 1, 0), int(round(0.95 * (len(scores_f) - 1))))
            p95_score = float(torch.sort(scores_f).values[q].item())
        return ScreeningMetadata(
            enabled=bool(enabled),
            pipeline_mode=self.pipeline_mode,
            total_samples=int(total),
            accepted_samples=int(accepted),
            rejected_samples=int(rejected),
            rejection_ratio=float(rejected) / max(1, int(total)),
            mean_score=mean_score,
            max_score=max_score,
            p95_score=p95_score,
            threshold=self.threshold,
            client_id=client_id,
            batch_id=batch_id,
            reason=reason,
        )


def apply_firs_gate(gate, data, targets, **kwargs):
    return gate.screen_batch(data, targets, **kwargs)


def screen_local_dataset(gate, data_iterator, client_id=None, is_poisoned_client=False):
    for batch_id, (data, targets) in enumerate(data_iterator):
        yield gate.screen_batch(
            data,
            targets,
            client_id=client_id,
            batch_id=batch_id,
            is_poisoned_client=is_poisoned_client,
        )
