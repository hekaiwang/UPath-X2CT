#!/usr/bin/env python3
"""Train the formal cone-aware UPath-LIDC-v1 B0 anchor with DDP."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from upathx2ct.sota.cone_projection import (
    ConeProjectorConfig,
    DifferentiableConeProjector,
)
from upathx2ct.sota.data import UPathLIDC128Dataset
from upathx2ct.sota.geometry import ConeBeamGeometry
from upathx2ct.sota.losses import B0LossWeights, B0Objective
from upathx2ct.sota.model import B0CoarseReconstructor, B0Config
from upathx2ct.sota.pretrained import (
    initialize_encoder_from_torchvision_convnext_tiny,
)
from upathx2ct.sota.projective_lifting import PhysicalGrid
from upathx2ct.sota.protocol import sha256_file
from upathx2ct.b0.autoencoder import load_autoencoder_checkpoint
from upathx2ct.b0.ema import ExponentialMovingAverage
from upathx2ct.b0.training import (
    CSVMetricLogger,
    atomic_torch_save,
    autocast_context,
    cosine_warmup_lambda,
    initialize_distributed,
    make_grad_scaler,
    move_batch,
    peak_memory_gb,
    prune_checkpoints,
    reduce_mean,
    restore_training_checkpoint,
    seed_everything,
    seed_worker,
    shutdown_distributed,
    training_checkpoint,
    unwrap_model,
)


HU_SCALE = 1262.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--autoencoder-checkpoint", type=Path, required=True)
    parser.add_argument("--projector-calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--encoder-init",
        choices=("torchvision", "checkpoint", "random"),
        default="torchvision",
    )
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=2, help="Per GPU")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=80000)
    parser.add_argument("--min-steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr-multiplier", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=3000)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--early-stop-validations", type=int, default=12)
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


class StandardizedDRRAugmentor(nn.Module):
    """Intensity-only augmentation that preserves the frozen camera geometry."""

    def __init__(
        self,
        gain_range: tuple[float, float] = (0.90, 1.10),
        bias_range: tuple[float, float] = (-0.10, 0.10),
        noise_std: float = 0.03,
        lateral_dropout: float = 0.02,
    ) -> None:
        super().__init__()
        self.gain_range = gain_range
        self.bias_range = bias_range
        self.noise_std = noise_std
        self.lateral_dropout = lateral_dropout

    def _one(self, image: torch.Tensor) -> torch.Tensor:
        shape = (image.shape[0], 1, 1, 1)
        gain = torch.empty(shape, device=image.device).uniform_(*self.gain_range)
        bias = torch.empty(shape, device=image.device).uniform_(*self.bias_range)
        return torch.clamp(
            image * gain + bias + torch.randn_like(image) * self.noise_std,
            -6.0,
            6.0,
        )

    def forward(
        self,
        ap: torch.Tensor,
        lat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ap = self._one(ap)
        lat = self._one(lat)
        if self.lateral_dropout > 0:
            dropped = (
                torch.rand(lat.shape[0], 1, 1, 1, device=lat.device)
                < self.lateral_dropout
            )
            lat = torch.where(dropped, torch.zeros_like(lat), lat)
        return ap, lat


def make_loader(dataset, batch_size, workers, sampler, shuffle) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "shuffle": shuffle if sampler is None else False,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
        "worker_init_fn": seed_worker,
        "drop_last": shuffle,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def load_partial_encoder(
    model: B0CoarseReconstructor,
    checkpoint_path: Path,
) -> dict[str, int | float | str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = checkpoint.get(
        "model",
        checkpoint.get(
            "model_state_dict",
            checkpoint.get("encoder", checkpoint.get("state_dict", checkpoint)),
        ),
    )
    target = model.conditioner.encoder.state_dict()
    matched = {}
    for target_key, target_value in target.items():
        candidates = [
            key
            for key, value in source.items()
            if (key == target_key or key.endswith("." + target_key))
            and value.shape == target_value.shape
        ]
        if candidates:
            matched[target_key] = source[candidates[0]]
    fraction = len(matched) / max(len(target), 1)
    if fraction < 0.90:
        raise ValueError(
            f"Encoder checkpoint matched only {len(matched)}/{len(target)} "
            f"tensors ({fraction:.1%})"
        )
    model.conditioner.encoder.load_state_dict(matched, strict=False)
    return {
        "source": str(checkpoint_path),
        "matched_tensors": len(matched),
        "target_tensors": len(target),
        "match_fraction": fraction,
    }


def per_case_gradient_energy(volume: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            volume.diff(dim=axis).abs().flatten(1).mean(dim=1)
            for axis in (2, 3, 4)
        ],
        dim=1,
    ).mean(dim=1)


@torch.inference_mode()
def validate(
    model: nn.Module,
    autoencoder: nn.Module,
    objective: B0Objective,
    loader: DataLoader,
    context,
    precision: str,
) -> dict[str, float]:
    model.eval()
    names = (
        "loss",
        "psnr_3d",
        "ssim_3d",
        "mae_hu",
        "mae_hu_body",
        "mae_hu_lung",
        "gradient_energy_ratio",
        "projection_loss",
        "count",
    )
    sums = torch.zeros(len(names), device=context.device, dtype=torch.float64)
    for batch in loader:
        batch = move_batch(batch, context.device)
        with autocast_context(context.device, precision):
            outputs = unwrap_model(model)(
                batch["xray_ap"],
                batch["xray_lat"],
                batch["projection_matrices"],
            )
            decoded = autoencoder.decode(
                outputs["coarse_latent"],
                normalized=True,
                clamp=False,
            )
        decoded = decoded.float()
        losses = objective(
            outputs,
            decoded,
            batch["target_latent"].float(),
            batch["ct"].float(),
            batch["xray_ap_raw"].float(),
            batch["xray_lat_raw"].float(),
            batch["body_mask"],
            batch["lung_mask"],
        )
        prediction = decoded.clamp(-1.0, 1.0)
        target = batch["ct"].float()
        error = prediction - target
        batch_size = prediction.shape[0]
        mse = error.square().flatten(1).mean(dim=1)
        psnr = 20.0 * torch.log10(
            torch.tensor(2.0, device=mse.device) / torch.sqrt(mse.clamp_min(1e-12))
        )
        ssim = torch.stack(
            [
                1.0
                - objective.ssim(
                    prediction[index : index + 1],
                    target[index : index + 1],
                )
                for index in range(batch_size)
            ]
        ).flatten()
        mae = error.abs().flatten(1).mean(dim=1) * HU_SCALE
        body_mae = (
            (error.abs() * batch["body_mask"]).flatten(1).sum(dim=1)
            / batch["body_mask"].flatten(1).sum(dim=1).clamp_min(1.0)
            * HU_SCALE
        )
        lung_mae = (
            (error.abs() * batch["lung_mask"]).flatten(1).sum(dim=1)
            / batch["lung_mask"].flatten(1).sum(dim=1).clamp_min(1.0)
            * HU_SCALE
        )
        gradient_ratio = per_case_gradient_energy(prediction) / (
            per_case_gradient_energy(target).clamp_min(1e-8)
        )
        values = torch.stack(
            (
                losses["loss"].repeat(batch_size),
                psnr,
                ssim,
                mae,
                body_mae,
                lung_mae,
                gradient_ratio,
                losses["loss_projection"].repeat(batch_size),
                torch.ones_like(psnr),
            ),
            dim=1,
        ).double()
        sums += values.sum(dim=0)
    if context.world_size > 1:
        torch.distributed.all_reduce(sums)
    count = sums[-1].clamp_min(1.0)
    metrics = {
        f"val_{name}": float(sums[index] / count)
        for index, name in enumerate(names[:-1])
    }
    metrics["val_cases"] = int(sums[-1].item())
    model.train()
    return metrics


def main() -> None:
    args = parse_args()
    if args.min_steps < 0 or args.min_steps > args.max_steps:
        raise ValueError("Require 0 <= min_steps <= max_steps")
    if not 0.0 < args.encoder_lr_multiplier <= 1.0:
        raise ValueError("Encoder LR multiplier must be in (0,1]")
    if args.encoder_init == "checkpoint" and args.encoder_checkpoint is None:
        raise ValueError("--encoder-init checkpoint requires --encoder-checkpoint")
    context = initialize_distributed()
    seed_everything(args.seed, context.rank)
    torch.backends.cudnn.benchmark = True
    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)

    autoencoder = load_autoencoder_checkpoint(
        str(args.autoencoder_checkpoint),
        device="cpu",
    ).to(context.device)
    autoencoder.eval().requires_grad_(False)
    config = B0Config(latent_channels=autoencoder.config.latent_channels)
    model: nn.Module = B0CoarseReconstructor(config).to(context.device)
    initialization: dict[str, object] = {"source": "resume"}
    if not args.resume:
        if args.encoder_init == "torchvision":
            initialization = initialize_encoder_from_torchvision_convnext_tiny(
                model.conditioner.encoder
            )
        elif args.encoder_init == "checkpoint":
            initialization = load_partial_encoder(model, args.encoder_checkpoint)
        else:
            initialization = {"source": "random"}
    if context.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            broadcast_buffers=False,
            static_graph=True,
        )

    train_dataset = UPathLIDC128Dataset(
        args.dataset_root,
        "train",
        include_raw_drr=True,
        latent_root=args.latent_root,
    )
    full_val_dataset = UPathLIDC128Dataset(
        args.dataset_root,
        "val",
        include_raw_drr=True,
        latent_root=args.latent_root,
    )
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True)
        if context.world_size > 1
        else None
    )
    val_dataset = (
        Subset(
            full_val_dataset,
            range(context.rank, len(full_val_dataset), context.world_size),
        )
        if context.world_size > 1
        else full_val_dataset
    )
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        train_sampler,
        True,
    )
    val_loader = make_loader(
        val_dataset,
        1,
        max(1, args.num_workers // 2),
        None,
        False,
    )
    calibration = json.loads(
        args.projector_calibration.read_text(encoding="utf-8")
    )
    projector = DifferentiableConeProjector(
        PhysicalGrid(
            config.global_shape_zyx,
            config.global_origin_centered_lps_mm,
            config.global_spacing_xyz_mm,
        ),
        ConeBeamGeometry(
            config.source_detector_distance_mm,
            config.source_object_distance_mm,
            config.detector_shape,
            config.detector_pixel_size_uv_mm,
            config.principal_point_uv_px,
        ),
        ConeProjectorConfig(
            detector_shape=tuple(calibration["detector_shape"]),
            ray_samples=int(calibration["ray_samples"]),
        ),
    ).to(context.device)
    objective = B0Objective(
        projector,
        calibration,
        B0LossWeights(),
    ).to(context.device)
    augmentor = StandardizedDRRAugmentor().to(context.device)

    unwrapped = unwrap_model(model)
    encoder_parameters = list(unwrapped.conditioner.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    reconstruction_parameters = [
        parameter
        for parameter in unwrapped.parameters()
        if id(parameter) not in encoder_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": reconstruction_parameters,
                "lr": args.lr,
                "group_name": "reconstruction",
            },
            {
                "params": encoder_parameters,
                "lr": args.lr * args.encoder_lr_multiplier,
                "group_name": "encoder",
            },
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(
            step,
            args.warmup_steps,
            args.max_steps,
        ),
    )
    scaler = make_grad_scaler(args.precision)
    ema = ExponentialMovingAverage(unwrap_model(model), args.ema_decay)
    step = 0
    best_loss = math.inf
    best_psnr = -math.inf
    best_metrics: dict[str, float] = {}
    validations_without_improvement = 0
    if args.resume:
        step, _, checkpoint = restore_training_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            scaler,
        )
        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        best_loss = float(checkpoint.get("best_loss", math.inf))
        best_psnr = float(checkpoint.get("best_psnr", -math.inf))
        best_metrics = checkpoint.get("best_metrics", {})
        validations_without_improvement = int(
            checkpoint.get("validations_without_improvement", 0)
        )

    if context.is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "training_complete.json").unlink(missing_ok=True)
        (args.output_dir / "config.json").write_text(
            json.dumps(
                {
                    **vars(args),
                    "b0_config": asdict(config),
                    "loss_weights": asdict(objective.weights),
                    "encoder_initialization": initialization,
                    "world_size": context.world_size,
                    "provenance": {
                        "training_pack_summary_sha256": sha256_file(
                            args.dataset_root / "summary.json"
                        ),
                        "latent_summary_sha256": sha256_file(
                            args.latent_root / "summary.json"
                        ),
                        "autoencoder_checkpoint_sha256": sha256_file(
                            args.autoencoder_checkpoint
                        ),
                        "projector_calibration_sha256": sha256_file(
                            args.projector_calibration
                        ),
                    },
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        logger = CSVMetricLogger(
            args.output_dir / "train.csv",
            [
                "step",
                "loss",
                "loss_latent",
                "loss_ct",
                "loss_window",
                "loss_gradient",
                "loss_high_frequency",
                "loss_ssim",
                "loss_projection",
                "loss_body_segmentation",
                "loss_lung_segmentation",
                "loss_uncertainty",
                "lr",
                "encoder_lr",
                "sec_per_step",
                "memory_gb",
            ],
        )
        print(
            f"B0 train={len(train_dataset)} val={len(full_val_dataset)} "
            f"world={context.world_size} global_batch={args.batch_size * context.world_size} "
            f"start={step}",
            flush=True,
        )
    else:
        logger = None

    epoch = 0
    stop = False
    completed_normally = False
    previous_time = time.time()
    model.train()
    try:
        while step < args.max_steps and not stop:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch in train_loader:
                if step >= args.max_steps or stop:
                    break
                step += 1
                batch = move_batch(batch, context.device)
                ap, lat = augmentor(batch["xray_ap"], batch["xray_lat"])
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(context.device, args.precision):
                    outputs = model(ap, lat, batch["projection_matrices"])
                    decoded = autoencoder.decode(
                        outputs["coarse_latent"],
                        normalized=True,
                        clamp=False,
                    )
                losses = objective(
                    outputs,
                    decoded.float(),
                    batch["target_latent"].float(),
                    batch["ct"].float(),
                    batch["xray_ap_raw"].float(),
                    batch["xray_lat_raw"].float(),
                    batch["body_mask"],
                    batch["lung_mask"],
                )
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                ema.update(unwrap_model(model))
                now = time.time()
                elapsed = now - previous_time
                previous_time = now

                if step % args.log_every == 0:
                    reduced = {
                        key: float(reduce_mean(value, context))
                        for key, value in losses.items()
                    }
                    if context.is_main:
                        row = {
                            "step": step,
                            **reduced,
                            "lr": optimizer.param_groups[0]["lr"],
                            "encoder_lr": optimizer.param_groups[1]["lr"],
                            "sec_per_step": elapsed,
                            "memory_gb": peak_memory_gb(context.device),
                        }
                        logger.write(row)
                        print(
                            f"step={step} loss={row['loss']:.5f} "
                            f"ct={row['loss_ct']:.5f} proj={row['loss_projection']:.5f} "
                            f"sec={elapsed:.3f} mem={row['memory_gb']:.1f}G",
                            flush=True,
                        )

                if step % args.validate_every == 0:
                    with ema.apply_to(unwrap_model(model)):
                        metrics = validate(
                            model,
                            autoencoder,
                            objective,
                            val_loader,
                            context,
                            args.precision,
                        )
                    loss_improved = metrics["val_loss"] < best_loss - 1e-4
                    psnr_improved = metrics["val_psnr_3d"] > best_psnr
                    if loss_improved:
                        best_loss = metrics["val_loss"]
                        best_metrics = metrics
                        validations_without_improvement = 0
                    else:
                        validations_without_improvement += 1
                    best_psnr = max(best_psnr, metrics["val_psnr_3d"])
                    if context.is_main:
                        payload = training_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            step,
                            -best_loss,
                            {
                                "b0_config": asdict(config),
                                "loss_weights": asdict(objective.weights),
                                "validation": metrics,
                                "best_metrics": best_metrics,
                                "best_loss": best_loss,
                                "best_psnr": best_psnr,
                                "validations_without_improvement": validations_without_improvement,
                                "ema": ema.state_dict(),
                            },
                        )
                        print(
                            f"validation step={step} loss={metrics['val_loss']:.5f} "
                            f"psnr3d={metrics['val_psnr_3d']:.3f} "
                            f"ssim3d={metrics['val_ssim_3d']:.4f} "
                            f"body_mae={metrics['val_mae_hu_body']:.1f}HU "
                            f"proj={metrics['val_projection_loss']:.4f}",
                            flush=True,
                        )
                        atomic_torch_save(payload, args.output_dir / "latest.pt")
                        (args.output_dir / "validation.json").write_text(
                            json.dumps(
                                {
                                    "step": step,
                                    "current": metrics,
                                    "best_by_loss": best_metrics,
                                    "best_psnr": best_psnr,
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        if loss_improved:
                            atomic_torch_save(payload, args.output_dir / "best.pt")
                        if psnr_improved:
                            atomic_torch_save(
                                payload,
                                args.output_dir / "best_psnr.pt",
                            )
                    stop_tensor = torch.tensor(
                        int(
                            step >= args.min_steps
                            and validations_without_improvement
                            >= args.early_stop_validations
                        ),
                        device=context.device,
                    )
                    if context.world_size > 1:
                        torch.distributed.broadcast(stop_tensor, src=0)
                    stop = bool(stop_tensor.item())

                if step % args.save_every == 0 and context.is_main:
                    payload = training_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        step,
                        -best_loss,
                        {
                            "b0_config": asdict(config),
                            "loss_weights": asdict(objective.weights),
                            "best_metrics": best_metrics,
                            "best_loss": best_loss,
                            "best_psnr": best_psnr,
                            "validations_without_improvement": validations_without_improvement,
                            "ema": ema.state_dict(),
                        },
                    )
                    atomic_torch_save(
                        payload,
                        args.output_dir / f"step-{step:07d}.pt",
                    )
                    prune_checkpoints(args.output_dir, "step-*.pt", keep=3)
            epoch += 1
        completed_normally = True
    finally:
        if context.is_main:
            payload = training_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                step,
                -best_loss,
                {
                    "b0_config": asdict(config),
                    "loss_weights": asdict(objective.weights),
                    "best_metrics": best_metrics,
                    "best_loss": best_loss,
                    "best_psnr": best_psnr,
                    "validations_without_improvement": validations_without_improvement,
                    "ema": ema.state_dict(),
                },
            )
            atomic_torch_save(payload, args.output_dir / "latest.pt")
            if completed_normally:
                (args.output_dir / "training_complete.json").write_text(
                    json.dumps(
                        {
                            "status": "pass" if bool(best_metrics) else "fail",
                            "step": step,
                            "reason": "early_stop" if stop else "max_steps",
                            "best_by_loss": best_metrics,
                            "checkpoint": "best.pt",
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            logger.close()
        shutdown_distributed()


if __name__ == "__main__":
    main()
