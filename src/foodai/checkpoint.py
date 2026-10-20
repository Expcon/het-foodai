"""Model-owner checkpoint IO, reusable by evaluation and the future Demo adapter."""
from __future__ import annotations

import pickle
from pathlib import Path

from foodai.contracts import ContractError, class_list_hash


def load_classifier_checkpoint(path, config):
    """Return a CPU eval model and metadata; accept only the frozen ordered prefix.

    Use only locally produced/trusted files. Never fall back to unrestricted pickle.
    """
    import torch
    from foodai.vision import get_model

    if config.status != "frozen":
        raise ContractError("Checkpoint loading requires a frozen class configuration")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise ContractError(f"Cannot safely load checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("metadata"), dict):
        raise ContractError("Checkpoint must contain metadata and model_state dictionaries")
    info = checkpoint["metadata"]
    n = info.get("stage")
    if type(n) is not int or n not in (8, 12, 16) or n > len(config.classes):
        raise ContractError("Checkpoint stage incompatible with frozen classes")
    expected = [{"id": c.id, "key": c.key, "zh": c.zh} for c in config.classes[:n]]
    if info.get("class_list") != expected or info.get("class_list_sha256") != class_list_hash(config.classes[:n]):
        raise ContractError("Checkpoint class mapping differs from frozen config prefix")
    if info.get("model_name") != "efficientnet_v2_s":
        raise ContractError("Unsupported checkpoint model_name")
    if type(info.get("input_size")) is not int or info["input_size"] < 96:
        raise ContractError("Invalid checkpoint input_size")
    if info.get("preprocessing", "foodai.vision.get_transforms/v1") != "foodai.vision.get_transforms/v1":
        raise ContractError("Unsupported checkpoint preprocessing")
    if not isinstance(checkpoint.get("model_state"), dict):
        raise ContractError("Checkpoint model_state must be a weights dictionary")
    model = get_model(n, pretrained=False)
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ContractError(f"Checkpoint weights incompatible with model: {exc}") from exc
    return model.eval(), info


def save_checkpoint(path, checkpoint):
    """Keep the previous best checkpoint intact if writing the new one fails."""
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
