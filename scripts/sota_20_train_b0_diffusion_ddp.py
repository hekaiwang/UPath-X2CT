"""Train the B0 residual-latent diffusion model with DDP."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from upathx2ct.sota.cone_projection import ConeProjectorConfig, DifferentiableConeProjector
from upathx2ct.sota.data import UPathLIDC128Dataset
from upathx2ct.sota.geometry import ConeBeamGeometry
from upathx2ct.sota.losses import B0DiffusionLossWeights, B0DiffusionObjective
from upathx2ct.sota.model import B0Config
from upathx2ct.sota.projective_lifting import PhysicalGrid
from upathx2ct.sota.protocol import sha256_file
from upathx2ct.b0.autoencoder import load_autoencoder_checkpoint
from upathx2ct.b0.diffusion import DiffusionConfig, ResidualLatentDiffusion
from upathx2ct.b0.ema import ExponentialMovingAverage, resolve_checkpoint_model_state
from upathx2ct.b0.training import CSVMetricLogger, atomic_torch_save, autocast_context, cosine_warmup_lambda, initialize_distributed, make_grad_scaler, move_batch, peak_memory_gb, prune_checkpoints, reduce_mean, restore_training_checkpoint, seed_everything, seed_worker, shutdown_distributed, training_checkpoint, unwrap_model

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--latent-root', type=Path, required=True)
    parser.add_argument('--coarse-cache-root', type=Path, required=True)
    parser.add_argument('--lesion-sidecar-root', type=Path, help='Audited XML contour sidecar, required for pathology preset.')
    parser.add_argument('--autoencoder-checkpoint', type=Path, required=True)
    parser.add_argument('--projector-calibration', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=2, help='Per GPU')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-train-cases', type=int, default=0)
    parser.add_argument('--max-val-cases', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=200000)
    parser.add_argument('--schedule-steps', type=int, default=200000, help='Original cosine schedule horizon; independent of stopping budget')
    parser.add_argument('--min-steps', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--warmup-steps', type=int, default=3000)
    parser.add_argument('--ema-decay', type=float, default=0.9999)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--validate-every', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--early-stop-validations', type=int, default=15)
    parser.add_argument('--precision', choices=('bf16', 'fp16', 'fp32'), default='bf16')
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument('--initialize-from', type=Path, help='Initialize network weights from a prior EMA checkpoint while resetting optimizer, scheduler, step and residual scale.')
    initialization.add_argument('--resume', type=Path, help='Resume the same run including optimizer and residual scale.')
    parser.add_argument('--seed', type=int, default=20260724)
    return parser.parse_args()

def make_loader(dataset, batch_size, workers, sampler, shuffle) -> DataLoader:
    kwargs = {'dataset': dataset, 'batch_size': batch_size, 'sampler': sampler, 'shuffle': shuffle if sampler is None else False, 'num_workers': workers, 'pin_memory': torch.cuda.is_available(), 'persistent_workers': workers > 0, 'worker_init_fn': seed_worker, 'drop_last': shuffle}
    if workers > 0:
        kwargs['prefetch_factor'] = 2
    return DataLoader(**kwargs)

@torch.inference_mode()
def validate(diffusion: nn.Module, objective: B0DiffusionObjective, autoencoder: nn.Module, loader: DataLoader, context, precision: str, seed: int) -> dict[str, float]:
    diffusion.eval()
    keys = ('loss', 'loss_noise', 'loss_clean', 'loss_ct', 'loss_window', 'loss_gradient', 'loss_high_frequency', 'loss_ssim', 'loss_projection', 'loss_lesion_ct', 'loss_lesion_gradient', 'loss_lesion_contrast')
    sums = torch.zeros(len(keys) + 2, device=context.device, dtype=torch.float64)
    devices = [context.device.index] if context.device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed + context.rank)
        for batch in loader:
            batch = move_batch(batch, context.device)
            with autocast_context(context.device, precision):
                result = unwrap_model(diffusion)(batch['target_latent'], batch['coarse_latent'], batch['condition'])
            losses = objective(result, autoencoder, batch['ct'], batch['xray_ap_raw'], batch['xray_lat_raw'], batch.get('lesion_probability'), batch.get('lesion_consensus_mask'), batch.get('has_xml_contour_target'))
            batch_size = batch['ct'].shape[0]
            sums[:len(keys)] += torch.stack([losses[key].double() for key in keys]) * batch_size
            sums[-2] += losses['xml_positive_cases'].double()
            sums[-1] += batch_size
    if context.world_size > 1:
        torch.distributed.all_reduce(sums)
    count = sums[-1].clamp_min(1.0)
    metrics = {f"val_{key.removeprefix('loss_')}": float(sums[index] / count) for (index, key) in enumerate(keys)}
    metrics['val_xml_positive_cases'] = int(sums[-2].item())
    metrics['val_cases'] = int(sums[-1].item())
    diffusion.train()
    return metrics

def main() -> None:
    args = parse_args()
    if args.min_steps < 0 or args.min_steps > args.max_steps:
        raise ValueError('Require 0 <= min_steps <= max_steps')
    if args.max_train_cases < 0 or args.max_val_cases < 0:
        raise ValueError('Case limits must be nonnegative')
    context = initialize_distributed()
    seed_everything(args.seed, context.rank)
    torch.backends.cudnn.benchmark = True
    if context.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(context.device)
    cache_summary = json.loads((args.coarse_cache_root / 'summary.json').read_text(encoding='utf-8'))
    if cache_summary.get('status') != 'pass':
        raise RuntimeError('B0 coarse cache has not passed')
    if cache_summary.get('model_variant') != 'B0':
        raise RuntimeError('B0 diffusion requires a B0 condition cache')
    residual_stats = json.loads((args.coarse_cache_root / 'residual_stats.json').read_text(encoding='utf-8'))
    if residual_stats.get('definition_split') != 'train':
        raise RuntimeError('Residual scale was not defined from train only')
    autoencoder = load_autoencoder_checkpoint(str(args.autoencoder_checkpoint), device='cpu').to(context.device)
    autoencoder.eval().requires_grad_(False)
    b0_config = B0Config(latent_channels=autoencoder.config.latent_channels)
    config = DiffusionConfig(latent_channels=autoencoder.config.latent_channels, condition_channels=b0_config.condition_channels, residual_scale=1.0)
    diffusion: nn.Module = ResidualLatentDiffusion(config).to(context.device)
    new_residual_scale = torch.tensor(residual_stats['std'], dtype=torch.float32)
    diffusion.set_residual_scale(new_residual_scale)
    initialization_hash = None
    if args.initialize_from is not None:
        initialization_checkpoint = torch.load(args.initialize_from, map_location='cpu')
        diffusion.load_state_dict(resolve_checkpoint_model_state(initialization_checkpoint, prefer_ema=True), strict=True)
        diffusion.set_residual_scale(new_residual_scale)
        initialization_hash = sha256_file(args.initialize_from)
    if context.world_size > 1:
        diffusion = DistributedDataParallel(diffusion, device_ids=[context.local_rank], broadcast_buffers=False, static_graph=True)
    train_dataset = UPathLIDC128Dataset(args.dataset_root, 'train', include_raw_drr=True, latent_root=args.latent_root, coarse_cache_root=args.coarse_cache_root, lesion_sidecar_root=args.lesion_sidecar_root, max_samples=args.max_train_cases)
    full_val_dataset = UPathLIDC128Dataset(args.dataset_root, 'val', include_raw_drr=True, latent_root=args.latent_root, coarse_cache_root=args.coarse_cache_root, lesion_sidecar_root=args.lesion_sidecar_root, max_samples=args.max_val_cases)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if context.world_size > 1 else None
    val_dataset = Subset(full_val_dataset, range(context.rank, len(full_val_dataset), context.world_size)) if context.world_size > 1 else full_val_dataset
    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, train_sampler, True)
    val_loader = make_loader(val_dataset, 1, max(1, args.num_workers // 2), None, False)
    calibration = json.loads(args.projector_calibration.read_text(encoding='utf-8'))
    projector = DifferentiableConeProjector(PhysicalGrid(b0_config.global_shape_zyx, b0_config.global_origin_centered_lps_mm, b0_config.global_spacing_xyz_mm), ConeBeamGeometry(b0_config.source_detector_distance_mm, b0_config.source_object_distance_mm, b0_config.detector_shape, b0_config.detector_pixel_size_uv_mm, b0_config.principal_point_uv_px), ConeProjectorConfig(detector_shape=tuple(calibration['detector_shape']), ray_samples=int(calibration['ray_samples']))).to(context.device)
    objective_weights = B0DiffusionLossWeights()
    objective = B0DiffusionObjective(projector, calibration, config.train_timesteps, objective_weights).to(context.device)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: cosine_warmup_lambda(step, args.warmup_steps, args.schedule_steps))
    scaler = make_grad_scaler(args.precision)
    ema = ExponentialMovingAverage(unwrap_model(diffusion), args.ema_decay)
    step = 0
    best_loss = math.inf
    best_admissible_score = math.inf
    best_metrics: dict[str, float] = {}
    best_admissible_metrics: dict[str, float] = {}
    baseline_metrics: dict[str, float] = {}
    validations_without_improvement = 0
    if args.resume:
        (step, _, checkpoint) = restore_training_checkpoint(args.resume, diffusion, optimizer, scheduler, scaler)
        if 'ema' in checkpoint:
            ema.load_state_dict(checkpoint['ema'])
        best_loss = float(checkpoint.get('best_loss', math.inf))
        best_admissible_score = float(checkpoint.get('best_admissible_score', math.inf))
        best_metrics = checkpoint.get('best_metrics', {})
        best_admissible_metrics = checkpoint.get('best_admissible_metrics', {})
        baseline_metrics = checkpoint.get('baseline_metrics', {})
        validations_without_improvement = int(checkpoint.get('validations_without_improvement', 0))
    selection_contract = {'selection': 'minimum validation loss'}
    if context.is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'training_complete.json').unlink(missing_ok=True)
        (args.output_dir / 'config.json').write_text(json.dumps({**vars(args), 'diffusion_config': asdict(config), 'loss_weights': asdict(objective.weights), 'residual_stats': residual_stats, 'baseline_validation': baseline_metrics, 'selection_contract': selection_contract, 'world_size': context.world_size, 'provenance': {'coarse_cache_summary_sha256': sha256_file(args.coarse_cache_root / 'summary.json'), 'autoencoder_checkpoint_sha256': sha256_file(args.autoencoder_checkpoint), 'projector_calibration_sha256': sha256_file(args.projector_calibration), 'initialization_checkpoint_sha256': initialization_hash, 'lesion_sidecar_summary_sha256': sha256_file(args.lesion_sidecar_root / 'summary.json') if args.lesion_sidecar_root is not None else None}}, indent=2, default=str), encoding='utf-8')
        logger = CSVMetricLogger(args.output_dir / 'train.csv', ['step', 'loss', 'loss_noise', 'loss_clean', 'loss_ct', 'loss_window', 'loss_gradient', 'loss_high_frequency', 'loss_ssim', 'loss_projection', 'loss_lesion_ct', 'loss_lesion_gradient', 'loss_lesion_contrast', 'xml_positive_cases', 'lr', 'sec_per_step', 'memory_gb'])
        print(f"{cache_summary.get('model_variant', 'B0')} diffusion train={len(train_dataset)} val={len(full_val_dataset)} world={context.world_size} global_batch={args.batch_size * context.world_size} start={step}", flush=True)
    else:
        logger = None
    epoch = 0
    stop = False
    completed_normally = False
    previous_time = time.time()
    diffusion.train()
    try:
        while step < args.max_steps and (not stop):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch in train_loader:
                if step >= args.max_steps or stop:
                    break
                step += 1
                batch = move_batch(batch, context.device)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(context.device, args.precision):
                    result = diffusion(batch['target_latent'], batch['coarse_latent'], batch['condition'])
                losses = objective(result, autoencoder, batch['ct'], batch['xray_ap_raw'], batch['xray_lat_raw'], batch.get('lesion_probability'), batch.get('lesion_consensus_mask'), batch.get('has_xml_contour_target'))
                scaler.scale(losses['loss']).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                ema.update(unwrap_model(diffusion))
                now = time.time()
                elapsed = now - previous_time
                previous_time = now
                if step % args.log_every == 0:
                    reduced = {key: float(reduce_mean(value, context)) for (key, value) in losses.items()}
                    if context.is_main:
                        row = {'step': step, **reduced, 'lr': optimizer.param_groups[0]['lr'], 'sec_per_step': elapsed, 'memory_gb': peak_memory_gb(context.device)}
                        logger.write(row)
                        print(f"step={step} loss={row['loss']:.5f} noise={row['loss_noise']:.5f} clean={row['loss_clean']:.5f} ct={row['loss_ct']:.5f} lesion={row['loss_lesion_ct']:.5f} sec={elapsed:.3f} mem={row['memory_gb']:.1f}G", flush=True)
                if step % args.validate_every == 0:
                    with ema.apply_to(unwrap_model(diffusion)):
                        metrics = validate(diffusion, objective, autoencoder, val_loader, context, args.precision, args.seed)
                    loss_improved = metrics['val_loss'] < best_loss - 0.0001
                    if loss_improved:
                        best_loss = metrics['val_loss']
                        best_metrics = metrics
                    admissible = True
                    admissible_score = metrics['val_loss']
                    selected_improved = loss_improved
                    if selected_improved:
                        validations_without_improvement = 0
                    else:
                        validations_without_improvement += 1
                    if context.is_main:
                        payload = training_checkpoint(diffusion, optimizer, scheduler, scaler, step, -best_loss, {'diffusion_config': asdict(config), 'loss_weights': asdict(objective.weights), 'residual_stats': residual_stats, 'validation': metrics, 'baseline_metrics': baseline_metrics, 'best_metrics': best_metrics, 'best_admissible_metrics': best_admissible_metrics, 'best_loss': best_loss, 'best_admissible_score': best_admissible_score, 'validations_without_improvement': validations_without_improvement, 'ema': ema.state_dict()})
                        print(f"validation step={step} loss={metrics['val_loss']:.5f} noise={metrics['val_noise']:.5f} clean={metrics['val_clean']:.5f} proj={metrics['val_projection']:.5f} lesion={metrics['val_lesion_ct']:.5f} admissible={admissible}", flush=True)
                        atomic_torch_save(payload, args.output_dir / 'latest.pt')
                        (args.output_dir / 'validation.json').write_text(json.dumps({'step': step, 'current': metrics, 'admissible': admissible, 'admissible_score': admissible_score, 'baseline': baseline_metrics, 'best_by_loss': best_metrics, 'best_admissible': best_admissible_metrics}, indent=2), encoding='utf-8')
                        if selected_improved:
                            atomic_torch_save(payload, args.output_dir / 'best.pt')
                    stop_tensor = torch.tensor(int(step >= args.min_steps and validations_without_improvement >= args.early_stop_validations), device=context.device)
                    if context.world_size > 1:
                        torch.distributed.broadcast(stop_tensor, src=0)
                    stop = bool(stop_tensor.item())
                if step % args.save_every == 0 and context.is_main:
                    payload = training_checkpoint(diffusion, optimizer, scheduler, scaler, step, -best_loss, {'diffusion_config': asdict(config), 'loss_weights': asdict(objective.weights), 'residual_stats': residual_stats, 'baseline_metrics': baseline_metrics, 'best_metrics': best_metrics, 'best_admissible_metrics': best_admissible_metrics, 'best_loss': best_loss, 'best_admissible_score': best_admissible_score, 'validations_without_improvement': validations_without_improvement, 'ema': ema.state_dict()})
                    atomic_torch_save(payload, args.output_dir / f'step-{step:07d}.pt')
                    prune_checkpoints(args.output_dir, 'step-*.pt', keep=3)
            epoch += 1
        completed_normally = True
    finally:
        if context.is_main:
            payload = training_checkpoint(diffusion, optimizer, scheduler, scaler, step, -best_loss, {'diffusion_config': asdict(config), 'loss_weights': asdict(objective.weights), 'residual_stats': residual_stats, 'baseline_metrics': baseline_metrics, 'best_metrics': best_metrics, 'best_admissible_metrics': best_admissible_metrics, 'best_loss': best_loss, 'best_admissible_score': best_admissible_score, 'validations_without_improvement': validations_without_improvement, 'ema': ema.state_dict()})
            atomic_torch_save(payload, args.output_dir / 'latest.pt')
            if completed_normally:
                (args.output_dir / 'training_complete.json').write_text(json.dumps({'status': 'pass' if bool(best_metrics) else 'fail', 'step': step, 'reason': 'early_stop' if stop else 'max_steps', 'objective_preset': 'global', 'baseline': baseline_metrics, 'best_by_loss': best_metrics, 'best_admissible': best_admissible_metrics, 'checkpoint': 'best.pt'}, indent=2), encoding='utf-8')
            logger.close()
        shutdown_distributed()
if __name__ == '__main__':
    main()
