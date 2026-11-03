#!/usr/bin/env python3
"""Train the formal UPath-LIDC-v1 CT autoencoder ceiling with DDP."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from upathx2ct.sota.data import UPathLIDC128Dataset
from upathx2ct.b0.autoencoder import AutoencoderConfig, HighFidelityAutoencoderKL
from upathx2ct.b0.ema import ExponentialMovingAverage
from upathx2ct.b0.losses import AutoencoderLossWeights, AutoencoderObjective
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


HU_SCALE = 0.5 * (1500.0 - (-1024.0))


def parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(result) < 2:
        raise ValueError("Autoencoder needs at least two channel levels")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channels", default="64,128,256")
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2, help="Per GPU")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=60000)
    parser.add_argument("--min-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
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


def make_loader(
    dataset,
    batch_size: int,
    workers: int,
    sampler: DistributedSampler | None,
    shuffle: bool,
) -> DataLoader:
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


def per_case_gradient_energy(volume: torch.Tensor) -> torch.Tensor:
    energies = [
        (volume.diff(dim=axis)).abs().flatten(1).mean(dim=1)
        for axis in (2, 3, 4)
    ]
    return torch.stack(energies, dim=1).mean(dim=1)


@torch.inference_mode()
def validate(
    model: nn.Module,
    objective: AutoencoderObjective,
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
        "count",
    )
    sums = torch.zeros(len(names), device=context.device, dtype=torch.float64)
    for batch in loader:
        batch = move_batch(batch, context.device)
        with autocast_context(context.device, precision):
            mean, sigma = unwrap_model(model).encode_distribution(batch["ct"])
            reconstruction = unwrap_model(model).net.decode(mean).clamp(-1.0, 1.0)
        losses = objective(
            reconstruction.float(),
            batch["ct"].float(),
            mean.float(),
            sigma.float(),
            batch["body_mask"],
            batch["lung_mask"],
            batch["lesion_mask"],
            batch["has_lesion_label"],
        )
        prediction = reconstruction.float()
        target = batch["ct"].float()
        batch_size = prediction.shape[0]
        error = prediction - target
        mse = error.square().flatten(1).mean(dim=1)
        psnr = 20.0 * torch.log10(
            torch.tensor(2.0, device=mse.device) / torch.sqrt(mse.clamp_min(1e-12))
        )
        mae = error.abs().flatten(1).mean(dim=1) * HU_SCALE
        body_denominator = batch["body_mask"].flatten(1).sum(dim=1).clamp_min(1.0)
        lung_denominator = batch["lung_mask"].flatten(1).sum(dim=1).clamp_min(1.0)
        body_mae = (
            (error.abs() * batch["body_mask"]).flatten(1).sum(dim=1)
            / body_denominator
            * HU_SCALE
        )
        lung_mae = (
            (error.abs() * batch["lung_mask"]).flatten(1).sum(dim=1)
            / lung_denominator
            * HU_SCALE
        )
        gradient_ratio = per_case_gradient_energy(prediction) / (
            per_case_gradient_energy(target).clamp_min(1e-8)
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
        values = torch.stack(
            (
                losses["loss"].repeat(batch_size),
                psnr,
                ssim,
                mae,
                body_mae,
                lung_mae,
                gradient_ratio,
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
    metrics["ceiling_gate"] = bool(
        metrics["val_psnr_3d"] >= 32.0
        and metrics["val_ssim_3d"] >= 0.90
        and metrics["val_mae_hu_body"] <= 80.0
        and 0.80 <= metrics["val_gradient_energy_ratio"] <= 1.20
    )
    model.train()
    return metrics


def main() -> None:
    args = parse_args()
    if args.min_steps < 0 or args.min_steps > args.max_steps:
        raise ValueError("Require 0 <= min_steps <= max_steps")
    context = initialize_distributed()
    seed_everything(args.seed, context.rank)
    torch.backends.cudnn.benchmark = True
    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)

    channels = parse_int_tuple(args.channels)
    config = AutoencoderConfig(
        channels=channels,
        num_res_blocks=(2,) * len(channels),
        attention_levels=(False,) * len(channels),
        latent_channels=args.latent_channels,
        norm_num_groups=32,
        with_encoder_nonlocal_attention=False,
        with_decoder_nonlocal_attention=False,
        use_flash_attention=False,
        use_checkpointing=False,
    )
    train_dataset = UPathLIDC128Dataset(
        args.dataset_root,
        "train",
        ct_only_augmentation=True,
        load_projections=False,
    )
    full_val_dataset = UPathLIDC128Dataset(
        args.dataset_root,
        "val",
        load_projections=False,
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

    model: nn.Module = HighFidelityAutoencoderKL(config).to(context.device)
    if context.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            broadcast_buffers=True,
            static_graph=True,
        )
    weights = replace(
        AutoencoderLossWeights(),
        gradient_energy=0.10,
        lesion=0.0,
        lesion_high_frequency=0.0,
    )
    objective = AutoencoderObjective(weights).to(context.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
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
    best_psnr = -math.inf
    best_metrics: dict[str, float] = {}
    best_ceiling_psnr = -math.inf
    best_ceiling_metrics: dict[str, float] = {}
    validations_without_improvement = 0
    if args.resume:
        step, best_psnr, checkpoint = restore_training_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            scaler,
        )
        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        best_metrics = checkpoint.get("best_metrics", {})
        best_ceiling_psnr = float(
            checkpoint.get("best_ceiling_psnr", -math.inf)
        )
        best_ceiling_metrics = checkpoint.get("best_ceiling_metrics", {})
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
                    "autoencoder_config": asdict(config),
                    "architecture_note": (
                        "High-resolution 32^3 latent with convolutional residual "
                        "blocks. Quadratic global attention is disabled in the "
                        "autoencoder; global reasoning is performed by the "
                        "geometry conditioner and latent diffusion stages."
                    ),
                    "loss_weights": asdict(weights),
                    "world_size": context.world_size,
                    "ceiling_gate": {
                        "psnr_3d_min": 32.0,
                        "ssim_3d_min": 0.90,
                        "body_mae_hu_max": 80.0,
                        "gradient_energy_ratio": [0.80, 1.20],
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
                "loss_l1",
                "loss_window",
                "loss_gradient",
                "loss_gradient_energy",
                "loss_high_frequency",
                "loss_ssim",
                "loss_kl",
                "loss_body",
                "loss_lung",
                "lr",
                "sec_per_step",
                "memory_gb",
            ],
        )
        print(
            f"AE train={len(train_dataset)} val={len(full_val_dataset)} "
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
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(context.device, args.precision):
                    outputs = model(batch["ct"])
                losses = objective(
                    outputs["reconstruction"].float(),
                    batch["ct"].float(),
                    outputs["mean"].float(),
                    outputs["sigma"].float(),
                    batch["body_mask"],
                    batch["lung_mask"],
                    batch["lesion_mask"],
                    batch["has_lesion_label"],
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
                            "sec_per_step": elapsed,
                            "memory_gb": peak_memory_gb(context.device),
                        }
                        logger.write(row)
                        print(
                            f"step={step} loss={row['loss']:.5f} "
                            f"l1={row['loss_l1']:.5f} grad={row['loss_gradient']:.5f} "
                            f"sec={elapsed:.3f} mem={row['memory_gb']:.1f}G",
                            flush=True,
                        )

                if step % args.validate_every == 0:
                    with ema.apply_to(unwrap_model(model)):
                        metrics = validate(
                            model,
                            objective,
                            val_loader,
                            context,
                            args.precision,
                        )
                    improved = metrics["val_psnr_3d"] > best_psnr + 0.01
                    if improved:
                        best_psnr = metrics["val_psnr_3d"]
                        best_metrics = metrics
                        validations_without_improvement = 0
                    else:
                        validations_without_improvement += 1
                    ceiling_improved = bool(
                        metrics["ceiling_gate"]
                        and metrics["val_psnr_3d"] > best_ceiling_psnr
                    )
                    if ceiling_improved:
                        best_ceiling_psnr = metrics["val_psnr_3d"]
                        best_ceiling_metrics = metrics
                    if context.is_main:
                        payload = training_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            step,
                            best_psnr,
                            {
                                "autoencoder_config": asdict(config),
                                "loss_weights": asdict(weights),
                                "validation": metrics,
                                "best_metrics": best_metrics,
                                "best_ceiling_psnr": best_ceiling_psnr,
                                "best_ceiling_metrics": best_ceiling_metrics,
                                "validations_without_improvement": validations_without_improvement,
                                "ema": ema.state_dict(),
                            },
                        )
                        print(
                            f"validation step={step} psnr3d={metrics['val_psnr_3d']:.3f} "
                            f"ssim3d={metrics['val_ssim_3d']:.4f} "
                            f"body_mae={metrics['val_mae_hu_body']:.1f}HU "
                            f"grad_ratio={metrics['val_gradient_energy_ratio']:.3f} "
                            f"gate={metrics['ceiling_gate']}",
                            flush=True,
                        )
                        atomic_torch_save(payload, args.output_dir / "latest.pt")
                        validation_report = {
                            "step": step,
                            "current": metrics,
                            "best": best_metrics,
                            "best_ceiling": best_ceiling_metrics,
                        }
                        (args.output_dir / "validation.json").write_text(
                            json.dumps(validation_report, indent=2),
                            encoding="utf-8",
                        )
                        if improved:
                            atomic_torch_save(payload, args.output_dir / "best.pt")
                        if ceiling_improved:
                            atomic_torch_save(
                                payload,
                                args.output_dir / "best_ceiling.pt",
                            )
                    stop_tensor = torch.tensor(
                        int(
                            step >= args.min_steps
                            and bool(best_ceiling_metrics)
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
                        best_psnr,
                        {
                            "autoencoder_config": asdict(config),
                            "loss_weights": asdict(weights),
                            "best_metrics": best_metrics,
                            "best_ceiling_psnr": best_ceiling_psnr,
                            "best_ceiling_metrics": best_ceiling_metrics,
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
                best_psnr,
                {
                    "autoencoder_config": asdict(config),
                    "loss_weights": asdict(weights),
                    "best_metrics": best_metrics,
                    "best_ceiling_psnr": best_ceiling_psnr,
                    "best_ceiling_metrics": best_ceiling_metrics,
                    "validations_without_improvement": validations_without_improvement,
                    "ema": ema.state_dict(),
                },
            )
            atomic_torch_save(payload, args.output_dir / "latest.pt")
            if completed_normally:
                (args.output_dir / "training_complete.json").write_text(
                    json.dumps(
                        {
                            "status": (
                                "pass" if bool(best_ceiling_metrics) else "fail"
                            ),
                            "step": step,
                            "reason": "early_stop" if stop else "max_steps",
                            "best_ceiling": best_ceiling_metrics,
                            "checkpoint": "best_ceiling.pt",
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            logger.close()
        shutdown_distributed()


if __name__ == "__main__":
    main()
