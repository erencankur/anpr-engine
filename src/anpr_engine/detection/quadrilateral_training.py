"""Reproducible training and checkpoint contracts for quadrilateral Detection v2."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Sized
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.optim import SGD, AdamW, Optimizer
from torch.utils.data import DataLoader

from anpr_engine.detection.annotations import load_quadrilateral_manifest
from anpr_engine.detection.config import (
    QuadrilateralDetectorConfig,
    quadrilateral_detector_config_sha256,
)
from anpr_engine.detection.model import (
    QuadrilateralDetectorOutput,
    YoloXQuadrilateralDetector,
    box_iou,
    decode_multiscale_quadrilateral_outputs,
    multiscale_cell_geometry,
)
from anpr_engine.detection.quadrilateral_data import (
    QuadrilateralDetectionDataset,
    QuadrilateralTarget,
    collate_quadrilateral_batch,
)
from anpr_engine.detection.training import (
    _aligned_generalized_iou,
    _device_memory,
    _environment,
    _git_state,
    resolve_device,
    seed_everything,
    sha256_file,
)


def _assignment(
    objectness_logits: Tensor,
    decoded_boxes: Tensor,
    targets: tuple[QuadrilateralTarget, ...],
    centers: Tensor,
    cell_sizes: Tensor,
    *,
    center_radius: float,
    top_k: int,
) -> tuple[Tensor, Tensor]:
    matched_target = torch.full_like(objectness_logits, -1, dtype=torch.long)
    matched_cost = torch.full_like(objectness_logits, math.inf)
    with torch.no_grad():
        for batch_index, sample in enumerate(targets):
            target_boxes = sample.boxes.to(decoded_boxes.device)
            for target_index, target_box in enumerate(target_boxes):
                inside_box = (
                    (centers[:, 0] > target_box[0])
                    & (centers[:, 1] > target_box[1])
                    & (centers[:, 0] < target_box[2])
                    & (centers[:, 1] < target_box[3])
                )
                target_center = (target_box[:2] + target_box[2:]) / 2.0
                inside_center = ((centers - target_center).abs() / cell_sizes).amax(
                    dim=1
                ) < center_radius
                candidates = inside_box | inside_center
                candidate_indices = torch.nonzero(candidates, as_tuple=False).flatten()
                if not candidate_indices.numel():
                    candidate_indices = (
                        (centers - target_center).square().sum(dim=1).argmin().view(1)
                    )
                candidate_boxes = decoded_boxes[batch_index, candidate_indices]
                candidate_ious = box_iou(target_box, candidate_boxes).clamp(0.0, 1.0)
                candidate_scores = objectness_logits[batch_index, candidate_indices].sigmoid()
                costs = (
                    -torch.log(candidate_scores.clamp(min=1e-8))
                    + 3.0 * (1.0 - candidate_ious)
                    + 0.05
                    * ((centers[candidate_indices] - target_center) / cell_sizes[candidate_indices])
                    .square()
                    .sum(dim=1)
                )
                top_count = min(top_k, candidate_ious.numel())
                dynamic_count = max(1, int(candidate_ious.topk(top_count).values.sum().item()))
                selected_local = costs.topk(min(dynamic_count, top_count), largest=False).indices
                selected = candidate_indices[selected_local]
                selected_costs = costs[selected_local]
                better = selected_costs < matched_cost[batch_index, selected]
                accepted = selected[better]
                matched_cost[batch_index, accepted] = selected_costs[better]
                matched_target[batch_index, accepted] = target_index
    return matched_target >= 0, matched_target


def quadrilateral_detector_loss(
    output: QuadrilateralDetectorOutput,
    targets: tuple[QuadrilateralTarget, ...],
    *,
    focal_alpha: float,
    focal_gamma: float,
    box_loss_weight: float,
    corner_loss_weight: float,
    corner_offset_scale: float,
    center_radius: float,
    assignment_top_k: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    objectness_logits, decoded_boxes, decoded_corners = decode_multiscale_quadrilateral_outputs(
        output, corner_offset_scale=corner_offset_scale
    )
    if len(targets) != objectness_logits.shape[0]:
        raise ValueError("target batch size differs from detector output")
    centers, cell_sizes = multiscale_cell_geometry(output)
    positive, matched_indices = _assignment(
        objectness_logits,
        decoded_boxes.detach(),
        targets,
        centers,
        cell_sizes,
        center_radius=center_radius,
        top_k=assignment_top_k,
    )
    objectness_target = positive.to(objectness_logits.dtype)
    probabilities = objectness_logits.sigmoid()
    cross_entropy = nn.functional.binary_cross_entropy_with_logits(
        objectness_logits, objectness_target, reduction="none"
    )
    target_probability = torch.where(positive, probabilities, 1.0 - probabilities)
    alpha = torch.where(positive, focal_alpha, 1.0 - focal_alpha)
    objectness_loss = (
        alpha * (1.0 - target_probability).pow(focal_gamma) * cross_entropy
    ).sum() / positive.sum().clamp(min=1)
    matched_boxes: list[Tensor] = []
    matched_corners: list[Tensor] = []
    matched_corner_valid: list[Tensor] = []
    for batch_index, sample in enumerate(targets):
        indices = matched_indices[batch_index, positive[batch_index]]
        if indices.numel():
            matched_boxes.append(sample.boxes.to(decoded_boxes.device)[indices])
            matched_corners.append(sample.corners.to(decoded_corners.device)[indices])
            matched_corner_valid.append(sample.corner_valid.to(decoded_corners.device)[indices])
    if matched_boxes:
        box_targets = torch.cat(matched_boxes)
        box_loss = (1.0 - _aligned_generalized_iou(decoded_boxes[positive], box_targets)).mean()
        corner_targets = torch.cat(matched_corners)
        corner_valid = torch.cat(matched_corner_valid)
        if corner_valid.any():
            corner_loss = nn.functional.smooth_l1_loss(
                decoded_corners[positive][corner_valid], corner_targets[corner_valid]
            )
        else:
            corner_loss = decoded_corners.sum() * 0.0
    else:
        box_loss = decoded_boxes.sum() * 0.0
        corner_loss = decoded_corners.sum() * 0.0
    total = objectness_loss + box_loss_weight * box_loss + corner_loss_weight * corner_loss
    return total, objectness_loss, box_loss, corner_loss, int(positive.sum().item())


def load_v1_warm_start(
    model: YoloXQuadrilateralDetector,
    checkpoint_path: Path,
) -> dict[str, Any]:
    payload = cast(
        dict[str, Any], torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    if payload.get("format_version") != "1.0" or payload.get("architecture") != (
        "yolox_multiscale_v1"
    ):
        raise ValueError("quadrilateral warm-start requires a yolox_multiscale_v1 checkpoint")
    source_state = cast(dict[str, Tensor], payload["model_state"])
    destination = model.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in destination and destination[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    expected_new = {key for key in destination if ".corners." in key}
    expected_missing = {key for key in expected_new if not key.endswith("num_batches_tracked")}
    if set(missing) != expected_missing or unexpected:
        raise ValueError("warm-start compatibility differs from the declared corner-only extension")
    return {
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_architecture": payload["architecture"],
        "loaded_parameter_keys": sorted(compatible),
        "initialized_parameter_keys": sorted(expected_new),
    }


def save_quadrilateral_checkpoint(
    path: Path,
    model: YoloXQuadrilateralDetector,
    optimizer: Optimizer,
    *,
    epoch: int,
    config: QuadrilateralDetectorConfig,
    run_id: str,
    validation_metrics: dict[str, Any],
    initialization: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": "2.0",
            "architecture": "yolox_multiscale_quad_v2",
            "base_channels": config.model.base_channels,
            "head_channels": config.model.head_channels,
            "image_size": config.model.image_size,
            "corner_offset_scale": config.model.corner_offset_scale,
            "dataset_id": config.dataset_id,
            "dataset_content_sha256": config.dataset_content_sha256,
            "config_sha256": quadrilateral_detector_config_sha256(config),
            "run_id": run_id,
            "epoch": epoch,
            "initialization": initialization,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "validation_metrics": validation_metrics,
        },
        path,
    )


def load_quadrilateral_checkpoint(
    path: Path, device: torch.device
) -> tuple[YoloXQuadrilateralDetector, dict[str, Any]]:
    payload = cast(dict[str, Any], torch.load(path, map_location=device, weights_only=True))
    if payload.get("format_version") != "2.0" or payload.get("architecture") != (
        "yolox_multiscale_quad_v2"
    ):
        raise ValueError("unsupported quadrilateral detector checkpoint")
    model = YoloXQuadrilateralDetector(
        base_channels=int(payload["base_channels"]),
        head_channels=int(payload["head_channels"]),
    )
    model.load_state_dict(cast(dict[str, Tensor], payload["model_state"]))
    model.to(device).eval()
    metadata = {key: value for key, value in payload.items() if not key.endswith("_state")}
    return model, metadata


def _optimizer(config: QuadrilateralDetectorConfig, model: nn.Module) -> Optimizer:
    if config.training.optimizer == "sgd":
        return SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=0.9,
            weight_decay=config.training.weight_decay,
        )
    return AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )


def _loader(
    manifest_path: Path,
    dataset_root: Path,
    config: QuadrilateralDetectorConfig,
    split: str,
    *,
    training: bool,
) -> DataLoader[Any]:
    manifest = load_quadrilateral_manifest(manifest_path)
    if manifest.dataset_id != config.dataset_id or (
        manifest.content_sha256 != config.dataset_content_sha256
    ):
        raise ValueError("quadrilateral dataset identity differs from training config")
    dataset = QuadrilateralDetectionDataset(
        manifest,
        dataset_root,
        split=split,
        image_size=config.model.image_size,
        training=training,
        augmentation=config.training.augmentation,
    )
    generator = torch.Generator().manual_seed(config.training.seed)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=training,
        num_workers=config.training.num_workers,
        generator=generator,
        collate_fn=collate_quadrilateral_batch,
    )


def _run_epoch(
    model: YoloXQuadrilateralDetector,
    loader: DataLoader[Any],
    device: torch.device,
    config: QuadrilateralDetectorConfig,
    optimizer: Optimizer | None,
) -> dict[str, float | int]:
    model.train(optimizer is not None)
    totals: list[float] = []
    objectness_values: list[float] = []
    box_values: list[float] = []
    corner_values: list[float] = []
    positives = 0
    context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    with context:
        for images, targets, _ in loader:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = cast(QuadrilateralDetectorOutput, model(images.to(device)))
            total, objectness, box, corner, positive_count = quadrilateral_detector_loss(
                output,
                targets,
                focal_alpha=config.model.focal_alpha,
                focal_gamma=config.model.focal_gamma,
                box_loss_weight=config.model.box_loss_weight,
                corner_loss_weight=config.model.corner_loss_weight,
                corner_offset_scale=config.model.corner_offset_scale,
                center_radius=config.model.assignment_center_radius,
                assignment_top_k=config.model.assignment_top_k,
            )
            if optimizer is not None:
                total.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
            totals.append(float(total.detach().cpu()))
            objectness_values.append(float(objectness.detach().cpu()))
            box_values.append(float(box.detach().cpu()))
            corner_values.append(float(corner.detach().cpu()))
            positives += positive_count
    if not totals:
        raise ValueError("quadrilateral loader produced no batches")
    return {
        "total_loss": sum(totals) / len(totals),
        "objectness_loss": sum(objectness_values) / len(objectness_values),
        "box_loss": sum(box_values) / len(box_values),
        "corner_loss": sum(corner_values) / len(corner_values),
        "positive_assignments": positives,
    }


def train_quadrilateral_detector(
    repository: Path,
    dataset_root: Path,
    manifest_path: Path,
    config_path: Path,
    config: QuadrilateralDetectorConfig,
    artifact_root: Path,
    *,
    warm_start_checkpoint: Path,
) -> Path:
    seed_everything(config.training.seed)
    device = resolve_device(config.training.device)
    started_at = datetime.now(UTC)
    run_id = f"{config.dataset_id}-{started_at.strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    run_root = artifact_root / "experiments" / run_id
    checkpoint_root = run_root / "checkpoints"
    run_root.mkdir(parents=True, exist_ok=False)
    train_loader = _loader(manifest_path, dataset_root, config, "train", training=True)
    validation_loader = _loader(manifest_path, dataset_root, config, "validation", training=False)
    model = YoloXQuadrilateralDetector(config.model.base_channels, config.model.head_channels)
    initialization = load_v1_warm_start(model, warm_start_checkpoint)
    model.to(device)
    optimizer = _optimizer(config, model)
    revision, dirty = _git_state(repository)
    metrics_path = run_root / "metrics.jsonl"
    best_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    completed_epochs = 0
    stop_reason = "configured_epochs_completed"
    best_path = checkpoint_root / "best.pt"
    for epoch in range(1, config.training.epochs + 1):
        completed_epochs = epoch
        started = time.monotonic()
        training_metrics = _run_epoch(model, train_loader, device, config, optimizer)
        validation_metrics = _run_epoch(model, validation_loader, device, config, None)
        record: dict[str, Any] = {
            "epoch": epoch,
            "duration_seconds": time.monotonic() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "training": training_metrics,
            "validation": validation_metrics,
            "device_memory_after_epoch": _device_memory(device),
        }
        save_quadrilateral_checkpoint(
            checkpoint_root / "last.pt",
            model,
            optimizer,
            epoch=epoch,
            config=config,
            run_id=run_id,
            validation_metrics=validation_metrics,
            initialization=initialization,
        )
        validation_loss = float(validation_metrics["total_loss"])
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_quadrilateral_checkpoint(
                best_path,
                model,
                optimizer,
                epoch=epoch,
                config=config,
                run_id=run_id,
                validation_metrics=validation_metrics,
                initialization=initialization,
            )
        else:
            epochs_without_improvement += 1
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if (
            epoch >= config.model.minimum_epochs
            and epochs_without_improvement >= config.model.early_stopping_patience
        ):
            stop_reason = "validation_early_stopping"
            break
    completed_at = datetime.now(UTC)
    run_manifest = {
        "schema_version": "quadrilateral-training-run-v2",
        "run_id": run_id,
        "status": "COMPLETED",
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "completed_at": completed_at.isoformat().replace("+00:00", "Z"),
        "source_revision": revision,
        "source_revision_dirty": dirty,
        "dataset_id": config.dataset_id,
        "dataset_content_sha256": config.dataset_content_sha256,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "config_sha256": quadrilateral_detector_config_sha256(config),
        "config_source_sha256": sha256_file(config_path),
        "config": config.model_dump(mode="json"),
        "environment": _environment(device),
        "split_counts": {
            "train": len(cast(Sized, train_loader.dataset)),
            "validation": len(cast(Sized, validation_loader.dataset)),
        },
        "initialization": initialization,
        "best_checkpoint": best_path.relative_to(repository).as_posix(),
        "best_checkpoint_sha256": sha256_file(best_path),
        "best_validation_total_loss": best_loss,
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "stop_reason": stop_reason,
        "selection_split": "validation",
        "selection_rule": "minimum_validation_total_loss_development_candidate",
        "official_acceptance_requires_paired_gate": True,
        "test_accessed": False,
    }
    (run_root / "run-manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return best_path
