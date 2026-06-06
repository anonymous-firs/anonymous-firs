# Anonymity Audit

## Repository

- Repository path: `anonymous_release`
- Current commit hash after finalization: read with `git log --format="%h %an <%ae> %s" --max-count=1`.
- Note: a commit cannot contain its own final hash without changing that hash.
- Git author information observed during audit: `Anonymous <anonymous@example.com>`
- Tracked file count observed during audit: 59

## Identity And Venue Trace Check

- Personal names, affiliation identifiers, and advisor identifiers: not found.
- Personal contact identifiers and account identifiers: not found.
- Local absolute paths: not found in public files.
- Prior venue or paper-management identifiers: not found.
- Notebook files: none found.
- Notebook outputs or metadata: none found.
- Git history author information: anonymous placeholder only.
- Chinese text in tracked public files: not found.
- Manuscript PDF files in the release repository: not found.

## Artifact Check

- Dataset directories or dataset files tracked by git: not found.
- Model checkpoints tracked by git: not found.
- Logs, TensorBoard events, wandb outputs, run outputs, or debug outputs tracked by git: not found.
- Python cache directories in the public working tree: not found after cleanup.
- OS metadata such as `.DS_Store` or `__MACOSX`: not found.

The recursive artifact-name scan has expected benign matches for utility scripts containing `data` in their filenames and for internal `.git/logs`. These are not public data artifacts.

## Dependency Check

- `requirements.txt` contains only public Python package names.
- No local paths, private package paths, or non-public dependency URLs were found.

## Syntax Check

- `python -m compileall . -q` was attempted first.
- `compileall` failed because Python could not replace generated `.pyc` files in `__pycache__` directories due to Windows permission errors.
- The failure was cache-write related, not a Python source syntax failure.
- A no-cache AST parse check was then run over all Python source files after the final edits and passed.
- FIRS detector smoke checks passed:
  - `python scripts/debug_detector_training_hook.py`
  - `python scripts/debug_firs_gate_pipeline.py`
- Generated `__pycache__` directories were removed after the check.

## Paper-Implementation Alignment

- The FIRS detector exposes the Global Semantic Encoder, Tile-wise Statistical Encoder, and Fusion Screening Head through `models/model_resnet_grid.py`.
- Detector training supports `BCEWithLogitsLoss + lambda * supervised_contrastive_loss(statistical_embedding)`.
- The contrastive weight can be fixed through YAML or learned as `lambda = softplus(xi)`.
- Recall-oriented validation threshold calibration is implemented in `firs_detector_training.py`.
- The local FIRS gate applies a frozen detector and calibrated threshold before local optimization, without online score histories or a default per-batch rejection budget.

## Final Assessment

The repository is suitable for upload to an Anonymous GitHub repository for triple-blind evaluation, subject to the final commit preserving the anonymous author identity.
