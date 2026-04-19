import re
import json
import math
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, Counter
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Global config
# ============================================================
SEED = 42

DATA_DIR = Path("./Dataset/Dataset_Feulgen_40times")
OUT_DIR = Path("./Output/Benchmark_Feulgen_40times")
MODELS_DIR = OUT_DIR / "saved_models"
FIG_DIR = OUT_DIR / "figures"
EXCEL_PATH = OUT_DIR / "summary.xlsx"

LABELS = ["I", "P", "M", "A", "T"]
LABEL_TO_IDX = {x: i for i, x in enumerate(LABELS)}
IDX_TO_LABEL = {i: x for x, i in LABEL_TO_IDX.items()}

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 0
N_SPLITS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DEVICE = "cuda" if DEVICE.type == "cuda" else "cpu"
USE_AMP = DEVICE.type == "cuda"

sns.set_style("white")


# ============================================================
# Imbalance strategy
# ============================================================
UNDERSAMPLE_TARGETS = {
    "I": 0.65,   # keep 65%
    "P": 0.80,   # keep 80%
}

AUGMENT_REPEATS = {
    "M": 5,
    "A": 5,
    "T": 6,
}

ROT_DEG = 25


# ============================================================
# Utilities
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def natural_key(s):
    parts = re.split(r"(\d+)", str(s))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def normalize_rgb(arr):
    arr = arr.astype(np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(3):
        ch = arr[:, :, c]
        lo = np.percentile(ch, 1.0)
        hi = np.percentile(ch, 99.5)
        if hi <= lo:
            hi = lo + 1.0
        ch = (ch - lo) / (hi - lo)
        ch = np.clip(ch, 0.0, 1.0)
        out[:, :, c] = ch
    return out


def load_rgb_image_as_pil(path: Path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr = normalize_rgb(arr)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ============================================================
# Sample parsing
# ============================================================
def parse_samples(data_dir: Path):
    img_files = sorted(data_dir.glob("*_DNA.png"), key=lambda p: natural_key(p.name))
    pattern = re.compile(r"^(?P<id>\d+)_(?P<label>[IPMAT])_DNA\.png$", re.IGNORECASE)

    samples = []
    for img_path in img_files:
        m = pattern.match(img_path.name)
        if not m:
            continue

        sample_id = m.group("id")
        label = m.group("label").upper()

        if label not in LABEL_TO_IDX:
            continue

        samples.append(
            {
                "sample_id": sample_id,
                "group_id": sample_id,
                "label": label,
                "label_idx": LABEL_TO_IDX[label],
                "img_path": str(img_path),
            }
        )

    samples = sorted(samples, key=lambda x: int(x["sample_id"]))
    return samples


# ============================================================
# Dataset balancing for train fold only
# ============================================================
def balance_train_samples(train_samples, seed=42):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for s in train_samples:
        grouped[s["label"]].append(s)

    balanced = []

    for phase in LABELS:
        items = grouped.get(phase, [])
        if len(items) == 0:
            continue

        if phase in UNDERSAMPLE_TARGETS:
            keep_ratio = UNDERSAMPLE_TARGETS[phase]
            keep_n = max(1, int(round(len(items) * keep_ratio)))
            items = items.copy()
            rng.shuffle(items)
            items = items[:keep_n]

        for s in items:
            s_base = dict(s)
            s_base["aug_id"] = 0
            balanced.append(s_base)

            n_aug = AUGMENT_REPEATS.get(phase, 0)
            for k in range(n_aug):
                s_aug = dict(s)
                s_aug["aug_id"] = k + 1
                balanced.append(s_aug)

    rng.shuffle(balanced)
    return balanced


# ============================================================
# Dataset
# ============================================================
def random_rgb_transform(img):
    angle = random.uniform(-ROT_DEG, ROT_DEG)

    img = TF.affine(
        img,
        angle=angle,
        translate=[0, 0],
        scale=1.0,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=255,
    )

    if random.random() < 0.5:
        img = TF.hflip(img)

    if random.random() < 0.5:
        img = TF.vflip(img)

    return img


class FeulgenDataset(Dataset):
    def __init__(self, samples, pretrained=False, train=False):
        self.samples = samples
        self.pretrained = pretrained
        self.train = train

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img = load_rgb_image_as_pil(Path(s["img_path"]))

        if self.train and s.get("aug_id", 0) > 0:
            img = random_rgb_transform(img)

        img = TF.resize(img, [IMG_SIZE, IMG_SIZE], interpolation=InterpolationMode.BILINEAR)
        x = TF.to_tensor(img)

        if self.pretrained:
            x = TF.normalize(x, self.mean, self.std)

        y = s["label_idx"]
        meta = {"sample_id": s["sample_id"], "label": s["label"]}
        return x, y, meta


# ============================================================
# Models
# ============================================================
@dataclass
class ModelConfig:
    name: str
    depth: int
    pretrained: bool = True
    lr: float = 3e-4
    weight_decay: float = 1e-4


MODEL_CONFIGS = [
    ModelConfig(name="ResNet18_pretrained", depth=18, pretrained=True, lr=5e-4),
    ModelConfig(name="ResNet34_pretrained", depth=34, pretrained=True, lr=3e-4),
]


def build_model(cfg: ModelConfig):
    if cfg.depth == 18:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
        model = models.resnet18(weights=weights)
    elif cfg.depth == 34:
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if cfg.pretrained else None
        model = models.resnet34(weights=weights)
    else:
        raise ValueError(f"Unsupported depth: {cfg.depth}")

    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    return model.to(DEVICE)


# ============================================================
# Metrics
# ============================================================
def compute_macro_roc(y_true, probs, n_classes):
    y_true = np.array(y_true)
    probs = np.array(probs)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr_grid = np.linspace(0.0, 1.0, 400)

    tprs = []
    aucs = []

    for c in range(n_classes):
        if np.sum(y_bin[:, c]) == 0 or np.sum(y_bin[:, c]) == len(y_bin[:, c]):
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        aucs.append(auc(fpr, tpr))

    if len(tprs) == 0:
        return fpr_grid, np.zeros_like(fpr_grid), 0.0

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = float(np.mean(aucs))
    return fpr_grid, mean_tpr, mean_auc


def compute_metrics(y_true, y_pred, probs):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        average="macro",
        zero_division=0,
    )
    fpr_grid, mean_tpr, mean_auc = compute_macro_roc(y_true, probs, len(LABELS))
    return {
        "accuracy": float(acc),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "roc_fpr": fpr_grid,
        "roc_tpr": mean_tpr,
        "roc_auc": float(mean_auc),
    }


# ============================================================
# Training / evaluation
# ============================================================
def make_loaders(train_samples, val_samples, cfg: ModelConfig):
    train_ds = FeulgenDataset(train_samples, pretrained=cfg.pretrained, train=True)
    val_ds = FeulgenDataset(val_samples, pretrained=cfg.pretrained, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
    )
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer=None, scaler=None, scheduler=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    losses = []
    ys_true, ys_pred = [], []
    probs_all = []

    for x, y, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
                logits = model(x)
                loss = criterion(logits, y)

            probs = torch.softmax(logits, dim=1)

            if train_mode:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

        losses.append(loss.item())
        pred = torch.argmax(logits, dim=1)

        ys_true.extend(y.detach().cpu().numpy().tolist())
        ys_pred.extend(pred.detach().cpu().numpy().tolist())
        probs_all.extend(probs.detach().cpu().numpy().tolist())

    metrics = compute_metrics(ys_true, ys_pred, probs_all)
    metrics["loss"] = float(np.mean(losses))
    metrics["y_true"] = ys_true
    metrics["y_pred"] = ys_pred
    metrics["probs"] = probs_all
    return metrics


def fold_dir_of(cfg: ModelConfig, fold_id: int):
    return MODELS_DIR / cfg.name / f"fold_{fold_id}"


def save_history_csv(save_path: Path, history: dict):
    df = pd.DataFrame(history)
    df.insert(0, "epoch", np.arange(1, len(df) + 1))
    df.to_csv(save_path, index=False)


def train_one_fold(cfg: ModelConfig, fold_id, train_samples_raw, val_samples):
    fold_dir = fold_dir_of(cfg, fold_id)
    ensure_dir(fold_dir)

    train_samples = balance_train_samples(train_samples_raw, seed=SEED + fold_id)

    print(f"[{cfg.name}][Fold {fold_id}] raw train size = {len(train_samples_raw)}, balanced train size = {len(train_samples)}")
    print(pd.Series([s['label'] for s in train_samples]).value_counts().reindex(LABELS, fill_value=0))

    train_loader, val_loader = make_loaders(train_samples, val_samples, cfg)

    model = build_model(cfg)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        epochs=NUM_EPOCHS,
        steps_per_epoch=max(1, len(train_loader)),
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    scaler = torch.amp.GradScaler(AMP_DEVICE, enabled=USE_AMP)

    history = defaultdict(list)
    best_metric = -1.0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer=optimizer, scaler=scaler, scheduler=scheduler)
        val_metrics = run_epoch(model, val_loader, criterion)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_metric:
            best_metric = val_metrics["accuracy"]
            best_state = {
                "model_state": copy.deepcopy(model.state_dict()),
                "epoch": epoch + 1,
                "best_val_acc": val_metrics["accuracy"],
            }

        print(
            f"[{cfg.name}][Fold {fold_id}] Epoch {epoch+1:03d}/{NUM_EPOCHS} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

    save_history_csv(fold_dir / "training_history.csv", history)
    torch.save(best_state, fold_dir / "best_model.pt")

    model.load_state_dict(best_state["model_state"])
    best_train_metrics = run_epoch(model, train_loader, criterion)
    best_val_metrics = run_epoch(model, val_loader, criterion)

    conf_val = confusion_matrix(
        best_val_metrics["y_true"],
        best_val_metrics["y_pred"],
        labels=list(range(len(LABELS))),
    )

    result = {
        "fold": fold_id,
        "best_epoch": best_state["epoch"],
        "best_val_acc": best_state["best_val_acc"],
        "history": history,
        "train_metrics": best_train_metrics,
        "val_metrics": best_val_metrics,
        "confusion_val": conf_val,
        "n_params": sum(p.numel() for p in model.parameters()),
    }

    with open(fold_dir / "fold_result.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "fold": int(result["fold"]),
                "best_epoch": int(result["best_epoch"]),
                "best_val_acc": float(result["best_val_acc"]),
                "history": {k: [float(x) for x in v] for k, v in result["history"].items()},
                "train_metrics": {
                    "accuracy": float(result["train_metrics"]["accuracy"]),
                    "macro_precision": float(result["train_metrics"]["macro_precision"]),
                    "macro_recall": float(result["train_metrics"]["macro_recall"]),
                    "macro_f1": float(result["train_metrics"]["macro_f1"]),
                    "roc_auc": float(result["train_metrics"]["roc_auc"]),
                    "roc_fpr": [float(x) for x in result["train_metrics"]["roc_fpr"]],
                    "roc_tpr": [float(x) for x in result["train_metrics"]["roc_tpr"]],
                    "loss": float(result["train_metrics"]["loss"]),
                },
                "val_metrics": {
                    "accuracy": float(result["val_metrics"]["accuracy"]),
                    "macro_precision": float(result["val_metrics"]["macro_precision"]),
                    "macro_recall": float(result["val_metrics"]["macro_recall"]),
                    "macro_f1": float(result["val_metrics"]["macro_f1"]),
                    "roc_auc": float(result["val_metrics"]["roc_auc"]),
                    "roc_fpr": [float(x) for x in result["val_metrics"]["roc_fpr"]],
                    "roc_tpr": [float(x) for x in result["val_metrics"]["roc_tpr"]],
                    "loss": float(result["val_metrics"]["loss"]),
                },
                "confusion_val": result["confusion_val"].tolist(),
                "n_params": int(result["n_params"]),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return result


# ============================================================
# Plotting
# ============================================================
def summarize_curves_asymmetric(curves):
    arr = np.array(curves, dtype=np.float32)
    center = arr.mean(axis=0)
    lower = np.quantile(arr, 0.16, axis=0)
    upper = np.quantile(arr, 0.84, axis=0)
    return center, lower, upper


def plot_training_curves(all_results, model_names, colors):
    epochs = np.arange(1, NUM_EPOCHS + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharex=True)

    metric_map = [
        ("train_acc", "Training Accuracy", axes[0, 0], "Accuracy"),
        ("val_acc", "Validation Accuracy", axes[0, 1], "Accuracy"),
        ("train_loss", "Training Loss", axes[1, 0], "Loss"),
        ("val_loss", "Validation Loss", axes[1, 1], "Loss"),
    ]

    legend_handles = []

    for model_name, color in zip(model_names, colors):
        fold_results = all_results[model_name]["fold_results"]

        for metric_key, title, ax, ylabel in metric_map:
            curves = []
            for fr in fold_results:
                if metric_key in fr["history"] and len(fr["history"][metric_key]) == NUM_EPOCHS:
                    curves.append(fr["history"][metric_key])

            if len(curves) == 0:
                continue

            center, lower, upper = summarize_curves_asymmetric(curves)
            line, = ax.plot(epochs, center, color=color, linewidth=2.2, label=model_name)
            ax.fill_between(epochs, lower, upper, color=color, alpha=0.18)
            ax.set_title(title, fontsize=14, pad=10)
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(False)

            if metric_key == "val_acc":
                legend_handles.append(line)

    if len(legend_handles) > 0:
        axes[0, 1].legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig.subplots_adjust(left=0.08, right=0.83, bottom=0.08, top=0.94, wspace=0.25, hspace=0.28)
    fig.savefig(FIG_DIR / "training_curves_2x2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_panels(all_results, model_names, colors):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    split_map = [("train_metrics", "Training Set ROC", axes[0]), ("val_metrics", "Validation Set ROC", axes[1])]
    legend_handles = []

    for model_name, color in zip(model_names, colors):
        fold_results = all_results[model_name]["fold_results"]

        for split_key, title, ax in split_map:
            tprs = [fr[split_key]["roc_tpr"] for fr in fold_results]
            aucs = [fr[split_key]["roc_auc"] for fr in fold_results]
            fpr_grid = fold_results[0][split_key]["roc_fpr"]

            center, lower, upper = summarize_curves_asymmetric(tprs)
            mean_auc = float(np.mean(aucs))
            std_auc = float(np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0)

            line, = ax.plot(
                fpr_grid, center, color=color, linewidth=2.2,
                label=f"{model_name} (AUC={mean_auc:.3f}±{std_auc:.3f})"
            )
            ax.fill_between(
                fpr_grid,
                np.maximum(0, lower),
                np.minimum(1, upper),
                color=color,
                alpha=0.18,
            )
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
            ax.set_title(title, fontsize=14, pad=10)
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.grid(False)

            if split_key == "val_metrics":
                legend_handles.append(line)

    axes[1].legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.subplots_adjust(left=0.08, right=0.80, bottom=0.12, top=0.90, wspace=0.28)
    fig.savefig(FIG_DIR / "roc_panels.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(all_results, model_names):
    n_models = len(model_names)
    ncols = 2
    nrows = math.ceil(n_models / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.8 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, model_name in zip(axes, model_names):
        fold_results = all_results[model_name]["fold_results"]
        cm_raw = np.sum([fr["confusion_val"] for fr in fold_results], axis=0).astype(np.int64)
        row_sums = cm_raw.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_raw, row_sums, out=np.zeros_like(cm_raw, dtype=float), where=row_sums > 0)

        ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
        ax.set_title(model_name, fontsize=14, pad=10)
        ax.set_xticks(np.arange(len(LABELS)))
        ax.set_yticks(np.arange(len(LABELS)))
        ax.set_xticklabels(LABELS, fontsize=11)
        ax.set_yticklabels(LABELS, fontsize=11)
        ax.set_xlabel("Predicted phase", fontsize=12)
        ax.set_ylabel("True phase", fontsize=12)

        ax.set_xticks(np.arange(-0.5, len(LABELS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(LABELS), 1), minor=True)
        ax.grid(which="minor", color=(1, 1, 1, 0.25), linestyle="-", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)

        for spine in ax.spines.values():
            spine.set_visible(False)

        for i in range(cm_raw.shape[0]):
            for j in range(cm_raw.shape[1]):
                ax.text(
                    j, i, str(cm_raw[i, j]),
                    ha="center", va="center",
                    color="white", fontsize=11, fontweight="medium"
                )

    for ax in axes[n_models:]:
        ax.axis("off")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.92, wspace=0.25, hspace=0.28)
    fig.savefig(FIG_DIR / "confusion_matrices_combined.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    set_seed(SEED)
    ensure_dir(OUT_DIR)
    ensure_dir(MODELS_DIR)
    ensure_dir(FIG_DIR)

    samples = parse_samples(DATA_DIR)
    if len(samples) == 0:
        raise RuntimeError("No valid samples found in Dataset_Feulgen_40times.")

    df_all = pd.DataFrame(samples)
    print("Total samples:", len(df_all))
    print(df_all["label"].value_counts().reindex(LABELS, fill_value=0))

    y_all = np.array([s["label_idx"] for s in samples], dtype=np.int64)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_results = {}
    summary_rows = []
    fold_rows = []

    for cfg in MODEL_CONFIGS:
        print(f"\n========== {cfg.name} ==========")
        fold_results = []

        for fold_id, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(samples)), y_all), start=1):
            train_samples_raw = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]

            fold_result = train_one_fold(
                cfg,
                fold_id,
                train_samples_raw,
                val_samples,
            )

            fold_results.append(fold_result)

            fold_rows.append(
                {
                    "model": cfg.name,
                    "fold": fold_id,
                    "best_epoch": fold_result["best_epoch"],
                    "best_val_acc": fold_result["best_val_acc"],
                    "train_accuracy": fold_result["train_metrics"]["accuracy"],
                    "train_macro_f1": fold_result["train_metrics"]["macro_f1"],
                    "train_auc": fold_result["train_metrics"]["roc_auc"],
                    "train_loss": fold_result["train_metrics"]["loss"],
                    "val_accuracy": fold_result["val_metrics"]["accuracy"],
                    "val_macro_f1": fold_result["val_metrics"]["macro_f1"],
                    "val_auc": fold_result["val_metrics"]["roc_auc"],
                    "val_loss": fold_result["val_metrics"]["loss"],
                    "n_params": fold_result["n_params"],
                }
            )

        summary = {
            "model": cfg.name,
            "depth": cfg.depth,
            "pretrained": cfg.pretrained,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "epochs": NUM_EPOCHS,
            "cv_folds": N_SPLITS,
            "n_params": int(fold_results[0]["n_params"]),
            "best_epoch_mean": float(np.mean([fr["best_epoch"] for fr in fold_results])),
            "best_epoch_std": float(np.std([fr["best_epoch"] for fr in fold_results], ddof=1) if len(fold_results) > 1 else 0.0),
            "best_val_acc_mean": float(np.mean([fr["best_val_acc"] for fr in fold_results])),
            "best_val_acc_std": float(np.std([fr["best_val_acc"] for fr in fold_results], ddof=1) if len(fold_results) > 1 else 0.0),
            "train_acc_mean": float(np.mean([fr["train_metrics"]["accuracy"] for fr in fold_results])),
            "train_acc_std": float(np.std([fr["train_metrics"]["accuracy"] for fr in fold_results], ddof=1) if len(fold_results) > 1 else 0.0),
            "val_acc_mean": float(np.mean([fr["val_metrics"]["accuracy"] for fr in fold_results])),
            "val_acc_std": float(np.std([fr["val_metrics"]["accuracy"] for fr in fold_results], ddof=1) if len(fold_results) > 1 else 0.0),
            "train_f1_mean": float(np.mean([fr["train_metrics"]["macro_f1"] for fr in fold_results])),
            "train_f1_std": float(np.std([fr["train_metrics"]["macro_f1"] for fr in fold_results], ddof=1) if len(fold_results) > 1 else 0.0),
            "val_f1_mean": float(np.mean([fr["val_metrics"]["macro_f1"] for fr in fold_results])),
            "val_f1_std": float(np.std([fr["val_metrics"]["macro_f1"] for fr in fold_results], ddof=1) if len(fold_results) > 1 else 0.0),
            "train_auc_mean": float(np.mean([fr["train_metrics"]["roc_auc"] for fr in fold_results])),
            "train_auc_std": float(np.std([fr["train_metrics"]["roc_auc"] for fr in fold_results], ddof=1) if len(fold_results) > 1 else 0.0),
            "val_auc_mean": float(np.mean([fr["val_metrics"]["roc_auc"] for fr in fold_results])),
            "val_auc_std": float(np.std([fr["val_metrics"]["roc_auc"] for fr in fold_results], ddof=1) if len(fold_results) > 1 else 0.0),
            "train_loss_mean": float(np.mean([fr["train_metrics"]["loss"] for fr in fold_results])),
            "train_loss_std": float(np.std([fr["train_metrics"]["loss"] for fr in fold_results], ddof=1) if len(fold_results) > 1 else 0.0),
            "val_loss_mean": float(np.mean([fr["val_metrics"]["loss"] for fr in fold_results])),
            "val_loss_std": float(np.std([fr["val_metrics"]["loss"] for fr in fold_results], ddof=1) if len(fold_results) > 1 else 0.0),
        }
        summary_rows.append(summary)

        all_results[cfg.name] = {
            "config": cfg,
            "fold_results": fold_results,
        }

    summary_df = pd.DataFrame(summary_rows)
    folds_df = pd.DataFrame(fold_rows)

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        folds_df.to_excel(writer, sheet_name="fold_metrics", index=False)
        pd.DataFrame(
            [
                {
                    "SEED": SEED,
                    "IMG_SIZE": IMG_SIZE,
                    "BATCH_SIZE": BATCH_SIZE,
                    "NUM_EPOCHS": NUM_EPOCHS,
                    "NUM_WORKERS": NUM_WORKERS,
                    "N_SPLITS": N_SPLITS,
                    "ROT_DEG": ROT_DEG,
                    "UNDERSAMPLE_TARGETS": str(UNDERSAMPLE_TARGETS),
                    "AUGMENT_REPEATS": str(AUGMENT_REPEATS),
                    "DATA_DIR": str(DATA_DIR),
                    "OUT_DIR": str(OUT_DIR),
                    "DEVICE": str(DEVICE),
                    "USE_AMP": USE_AMP,
                    "AMP_DEVICE": AMP_DEVICE,
                }
            ]
        ).to_excel(writer, sheet_name="global_config", index=False)

    model_names = [cfg.name for cfg in MODEL_CONFIGS]
    colors = sns.color_palette("viridis", n_colors=len(model_names))

    plot_training_curves(all_results, model_names, colors)
    plot_roc_panels(all_results, model_names, colors)
    plot_confusion_matrices(all_results, model_names)

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Excel: {EXCEL_PATH.resolve()}")
    print(f"Figures: {FIG_DIR.resolve()}")
    print(f"Models: {MODELS_DIR.resolve()}")


if __name__ == "__main__":
    main()