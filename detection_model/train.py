"""Run python train.py in this directory; tensorboard --logdir output/tensorboard.

To resume, set RESUME to OUTPUT_DIR / "last.pt" and continue from the next epoch.
Checkpoint reference: https://docs.pytorch.org/tutorials/beginner/saving_loading_models
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from data import DetectionDataset, collate_fn, move_batch
from model import DetectionLoss, GRADIENT_CHECKPOINTING, build_model
from test import evaluate, visualize

RUN_TAG = "aug-res-1"

DATASET_ROOT = "~/erc-work-data/data-of-cuneiform-ocr-data/processed-data-exp-1/"
VARIANT = "base-coco"
DEBUG_SUBSET = False  # True applies the subset rule defined in data.py.
DEVICE = "cuda"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
RESUME =  None # "output/last.pt" # For example, OUTPUT_DIR / "last.pt" or OUTPUT_DIR / "epoch_0005.pt".
TRAIN_CONFIG = {
    "epochs": 1000,
    "batch_size": 2,
    "eval_batch_size": 1,
    "grad_accum_steps": 4,
    "lr": 2e-4,
    "lr_backbone": 2e-5,
    "weight_decay": 1e-4,
    "lr_drop_epoch": 40,
    "max_grad_norm": 0.1,
    "eval_interval": 5,
    "num_workers": 14,
    "amp_dtype": torch.float16,  # One shared configuration; V100 supports FP16 but not BF16.
    "seed": 42,
    "save_interval": 5,
    "max_detections": 300,
}


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, global_step,
                    best_ap, generator, dataset, val_dataset, train_config=None, *, run_tag=None):
    """Save full training state at an epoch boundary, replacing the target after the write completes."""
    numpy_rng = np.random.get_state()
    config = TRAIN_CONFIG if train_config is None else train_config
    state = {
        "model": model.state_dict(), "model_config": model.config.to_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "epoch": epoch, "global_step": global_step,
        "best_ap": best_ap, "run_tag": RUN_TAG if run_tag is None else run_tag,
        "train_config": {**config, "amp_dtype": str(config["amp_dtype"])},
        "class_names": dataset.class_names, "category_to_label": dataset.category_to_label,
        "train_ids": dataset.ids, "val_ids": val_dataset.ids,
        "python_rng": random.getstate(),
        "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_rng": generator.get_state(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def checkpoint_path(output_dir, filename, tag=None):
    """Optionally add a CuneiMAE run tag to filenames; keep baseline filenames unchanged."""
    path = Path(output_dir) / filename
    return path if tag is None else path.with_name(f"{path.stem}_{tag}{path.suffix}")


def restore_training(checkpoint, optimizer, scheduler, scaler, generator):
    # The model is already restored on CUDA; optimizer momentum is restored to its parameters' device.
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    random.setstate(checkpoint["python_rng"])
    name, keys, pos, has_gauss, cached = checkpoint["numpy_rng"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(checkpoint["torch_rng"])
    if checkpoint["cuda_rng"]:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    generator.set_state(checkpoint["loader_rng"])
    return checkpoint["epoch"] + 1, checkpoint["global_step"], checkpoint["best_ap"]


def main(*, model_builder=None, train_config=None, debug_subset=None,
         output_dir=None, resume=RESUME, run_tag=None, tag_checkpoints=False):
    """Shared training loop; alternate entrypoints supply a model builder and experiment settings."""
    model_builder = build_model if model_builder is None else model_builder
    config = TRAIN_CONFIG if train_config is None else train_config
    debug_subset = DEBUG_SUBSET if debug_subset is None else debug_subset
    output_dir = Path(OUTPUT_DIR if output_dir is None else output_dir)
    run_tag = RUN_TAG if run_tag is None else run_tag
    if not torch.cuda.is_available():
        raise RuntimeError("本脚本固定使用 CUDA，请在 GPU 环境运行")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])

    root = Path(DATASET_ROOT).expanduser() / VARIANT
    train_dataset = DetectionDataset(root, "train", debug_subset)
    val_dataset = DetectionDataset(root, "val", debug_subset)
    if (train_dataset.category_to_label != val_dataset.category_to_label
            or train_dataset.class_names != val_dataset.class_names):
        raise ValueError("train/val 的类别 ID 和名称必须一致")
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, generator=generator,
        num_workers=config["num_workers"], collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["eval_batch_size"], shuffle=False,
        num_workers=config["num_workers"], collate_fn=collate_fn, pin_memory=True,
    )
    print(f"train={len(train_dataset)}, val={len(val_dataset)}, classes={len(train_dataset.class_names)}")

    checkpoint = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if checkpoint is not None:
        if (checkpoint["class_names"] != train_dataset.class_names
                or checkpoint["category_to_label"] != train_dataset.category_to_label
                or checkpoint["train_ids"] != train_dataset.ids or checkpoint["val_ids"] != val_dataset.ids):
            raise ValueError("续训时请保持原来的类别映射和数据 subset")
    model = model_builder(train_dataset.class_names, checkpoint).to(DEVICE)
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    criterion = DetectionLoss(model.config).to(DEVICE)

    # Keep the original parameter groups: lower LR for backbone, reference_points, and sampling_offsets.
    low_lr, other = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = low_lr if any(part in name for part in ("backbone", "reference_points", "sampling_offsets")) else other
            group.append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": other, "lr": config["lr"]},
         {"params": low_lr, "lr": config["lr_backbone"]}], weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [config["lr_drop_epoch"]], gamma=0.1)
    scaler = torch.amp.GradScaler("cuda")
    start_epoch, global_step, best_ap = 1, 0, float("-inf")
    if checkpoint is not None:
        start_epoch, global_step, best_ap = restore_training(checkpoint, optimizer, scheduler, scaler, generator)
        del checkpoint

    # Use batch global_step for all events; hide events after the checkpoint step when resuming.
    # https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
    with SummaryWriter(str(output_dir / "tensorboard" / "train" / run_tag),
                       purge_step=global_step + 1 if resume else None) as writer:
        for epoch in range(start_epoch, config["epochs"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            sums = {"total": 0.0, "decoder": 0.0, "encoder": 0.0}
            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
            for step, batch in enumerate(progress):
                batch = move_batch(batch, DEVICE)
                accumulation = config["grad_accum_steps"]
                # Average the final accumulation group by its actual batch count, including small subsets.
                group_size = min(accumulation, len(train_loader) - (step // accumulation) * accumulation)
                with torch.autocast("cuda", dtype=config["amp_dtype"]):
                    outputs = model(**batch)
                    loss, parts = criterion(outputs, batch["labels"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {step}")
                scaler.scale(loss / group_size).backward()
                if epoch == start_epoch and step == 0:
                    # The two-stage proposal heads must also receive gradients.
                    assert all(p.grad is not None for head in (model.class_embed, model.bbox_embed)
                               for p in head.parameters()), "Some detection heads received no gradient"
                if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                global_step += 1
                for key, value in {"total": loss.detach(), **parts}.items():
                    sums[key] += float(value)
                    writer.add_scalar(f"loss/{key}", float(value), global_step)
                writer.add_scalar("lr/detector", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("lr/backbone", optimizer.param_groups[1]["lr"], global_step)
                progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
            scheduler.step()
            for key, value in sums.items():
                writer.add_scalar(f"epoch_loss/{key}", value / len(train_loader), global_step)

            improved = False
            if epoch % config["eval_interval"] == 0 or epoch == config["epochs"]:
                metrics, predictions = evaluate(model, val_loader, val_dataset, config["max_detections"])
                for name, value in metrics.items():
                    writer.add_scalar(f"val/{name}", value, global_step)
                visualize(val_dataset, predictions, writer=writer, step=global_step, max_images=10)
                improved = metrics["AP"] > best_ap
                best_ap = max(best_ap, metrics["AP"])
                print(f"epoch={epoch}, val_AP={metrics['AP']:.4f}, best_AP={best_ap:.4f}")

            tag = run_tag if tag_checkpoints else None
            destinations = [checkpoint_path(output_dir, "best.pt", tag)] if improved else []
            if epoch % config["save_interval"] == 0 or epoch == config["epochs"]:
                destinations += [checkpoint_path(output_dir, f"epoch_{epoch:04d}.pt", tag),
                                 checkpoint_path(output_dir, "last.pt", tag)]
            for path in destinations:
                save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, global_step,
                                best_ap, generator, train_dataset, val_dataset,
                                train_config=config, run_tag=run_tag)
            writer.flush()


if __name__ == "__main__":
    main()
