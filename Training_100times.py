import re
import json
import math
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
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


SEED = 42
DATA_DIR = Path("./Dataset/Dataset_100times")
OUT_DIR = Path("./Output/Benchmark_100times")
MODELS_DIR = OUT_DIR / "saved_models"
FIG_DIR = OUT_DIR / "figures"
EXCEL_PATH = OUT_DIR / "benchmark_summary.xlsx"

LABELS = ["I", "P", "M", "A", "T"]
LABEL_TO_IDX = {x: i for i, x in enumerate(LABELS)}
IDX_TO_LABEL = {i: x for x, i in LABEL_TO_IDX.items()}

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 200
NUM_WORKERS = 0
N_SPLITS = 5
HOLDOUT_RATIO = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DEVICE = "cuda" if DEVICE.type == "cuda" else "cpu"
USE_AMP = DEVICE.type == "cuda"

sns.set_style("white")


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


def confirm_yes_no(prompt_text: str) -> bool:
    while True:
        s = input(prompt_text).strip().upper()
        if s in {"Y", "N"}:
            return s == "Y"
        print("Please input Y or N.")


def normalize_gray(arr, low_pct=1.0, high_pct=99.5):
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, low_pct)
    hi = np.percentile(arr, high_pct)
    if hi <= lo:
        hi = lo + 1.0
    arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def load_gray_image(path: Path):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)
    return normalize_gray(arr)


def parse_samples(data_dir: Path):
    dna_files = sorted(data_dir.glob("*_DNA.png"), key=lambda p: natural_key(p.name))
    pattern = re.compile(r"^(?P<id>\d+)_(?P<label>[IPMAT])_DNA\.png$", re.IGNORECASE)

    samples = []
    missing_pairs = []

    for dna_path in dna_files:
        m = pattern.match(dna_path.name)
        if not m:
            continue

        sample_id = m.group("id")
        label = m.group("label").upper()
        tub_path = data_dir / f"{sample_id}_{label}_Tubulin.png"

        if label not in LABEL_TO_IDX:
            continue

        if not tub_path.exists():
            missing_pairs.append(dna_path.name)
            continue

        samples.append(
            {
                "sample_id": sample_id,
                "group_id": sample_id,
                "label": label,
                "label_idx": LABEL_TO_IDX[label],
                "dna_path": str(dna_path),
                "tub_path": str(tub_path),
            }
        )

    if missing_pairs:
        print(f"[Warning] Missing Tubulin pairs for {len(missing_pairs)} DNA files.")

    samples = sorted(samples, key=lambda x: int(x["sample_id"]))
    return samples


def balance_train_indices(indices, samples):
    buckets = defaultdict(list)
    for idx in indices:
        label = samples[idx]["label"]
        buckets[label].append(idx)

    other_labels = ["P", "M", "A", "T"]
    other_counts = [len(buckets.get(lbl, [])) for lbl in other_labels]
    target_i = int(round(np.mean(other_counts))) if len(other_counts) > 0 else len(buckets.get("I", []))

    i_indices = buckets.get("I", [])
    if len(i_indices) > target_i and target_i > 0:
        random.shuffle(i_indices)
        buckets["I"] = i_indices[:target_i]

    balanced = []
    for k in LABELS:
        balanced.extend(buckets.get(k, []))
    random.shuffle(balanced)
    return balanced


class CellDataset(Dataset):
    def __init__(self, samples, pretrained=False, train=False):
        self.samples = samples
        self.pretrained = pretrained

        aug = []
        if train:
            aug = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
            ]

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                *aug,
                transforms.ToTensor(),
            ]
        )

        self.pretrained_norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        dna = load_gray_image(Path(s["dna_path"]))
        tub = load_gray_image(Path(s["tub_path"]))
        merge = 0.5 * dna + 0.5 * tub

        rgb = np.stack([dna, tub, merge], axis=-1)
        rgb = (rgb * 255).astype(np.uint8)

        x = self.transform(rgb)
        if self.pretrained:
            x = self.pretrained_norm(x)

        y = s["label_idx"]
        meta = {"sample_id": s["sample_id"], "label": s["label"]}
        return x, y, meta


class SmallCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class DeepCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


@dataclass
class ModelConfig:
    name: str
    kind: str
    depth: int | None = None
    pretrained: bool = False
    lr: float = 1e-3
    weight_decay: float = 1e-4


MODEL_CONFIGS = [
    ModelConfig(name="CNN_Small", kind="cnn_small", pretrained=False, lr=3e-3),
    ModelConfig(name="CNN_Deep", kind="cnn_deep", pretrained=False, lr=3e-3),
    ModelConfig(name="ResNet18_scratch", kind="resnet", depth=18, pretrained=False, lr=3e-3),
    ModelConfig(name="ResNet18_pretrained", kind="resnet", depth=18, pretrained=True, lr=5e-4),
    ModelConfig(name="ResNet34_scratch", kind="resnet", depth=34, pretrained=False, lr=2e-3),
    ModelConfig(name="ResNet34_pretrained", kind="resnet", depth=34, pretrained=True, lr=3e-4),
]


def build_model(cfg: ModelConfig):
    if cfg.kind == "cnn_small":
        return SmallCNN(num_classes=len(LABELS))
    if cfg.kind == "cnn_deep":
        return DeepCNN(num_classes=len(LABELS))
    if cfg.kind == "resnet":
        if cfg.depth == 18:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
            model = models.resnet18(weights=weights)
        elif cfg.depth == 34:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if cfg.pretrained else None
            model = models.resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet depth: {cfg.depth}")
        model.fc = nn.Linear(model.fc.in_features, len(LABELS))
        return model
    raise ValueError(f"Unknown model kind: {cfg.kind}")


def compute_class_weights(train_samples):
    counts = Counter(s["label_idx"] for s in train_samples)
    total = sum(counts.values())
    weights = []
    for i in range(len(LABELS)):
        c = counts.get(i, 1)
        weights.append(total / (len(LABELS) * c))
    return torch.tensor(weights, dtype=torch.float32)


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
        y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0
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


def make_loaders(samples, train_indices, val_indices, cfg: ModelConfig):
    train_balanced = balance_train_indices(train_indices, samples)

    train_samples = [samples[i] for i in train_balanced]
    val_samples = [samples[i] for i in val_indices]

    train_ds = CellDataset(train_samples, pretrained=cfg.pretrained, train=True)
    val_ds = CellDataset(val_samples, pretrained=cfg.pretrained, train=False)

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
    return train_loader, val_loader, train_samples, val_samples


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, scheduler=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    losses = []
    ys_true, ys_pred = [], []
    probs_all = []
    metas_all = []

    for x, y, metas in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

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

        if isinstance(metas, dict):
            bs = len(metas["sample_id"])
            for i in range(bs):
                metas_all.append(
                    {
                        "sample_id": metas["sample_id"][i],
                        "label": metas["label"][i],
                    }
                )

    metrics = compute_metrics(ys_true, ys_pred, probs_all)
    metrics["loss"] = float(np.mean(losses))
    metrics["y_true"] = ys_true
    metrics["y_pred"] = ys_pred
    metrics["probs"] = probs_all
    metrics["metas"] = metas_all
    return metrics


def fold_dir_of(cfg: ModelConfig, fold_id: int):
    return MODELS_DIR / cfg.name / f"fold_{fold_id}"


def final_dir_of(cfg: ModelConfig):
    return MODELS_DIR / cfg.name / "final_trainpool"


def best_fold_ckpt_path(cfg: ModelConfig, fold_id: int):
    return fold_dir_of(cfg, fold_id) / "best_model.pt"


def best_final_ckpt_path(cfg: ModelConfig):
    return final_dir_of(cfg) / "best_final_model.pt"


def save_history_csv(save_path: Path, history: dict):
    if len(history) == 0:
        return
    df = pd.DataFrame(history)
    df.insert(0, "epoch", np.arange(1, len(df) + 1))
    df.to_csv(save_path, index=False)


def save_fold_result_json(fold_dir: Path, fold_result: dict):
    serializable = {
        "fold": int(fold_result["fold"]),
        "best_epoch": int(fold_result["best_epoch"]),
        "best_val_acc": float(fold_result["best_val_acc"]),
        "history": {k: [float(x) for x in v] for k, v in fold_result.get("history", {}).items()},
        "train_metrics": {
            "accuracy": float(fold_result["train_metrics"]["accuracy"]),
            "macro_precision": float(fold_result["train_metrics"]["macro_precision"]),
            "macro_recall": float(fold_result["train_metrics"]["macro_recall"]),
            "macro_f1": float(fold_result["train_metrics"]["macro_f1"]),
            "roc_auc": float(fold_result["train_metrics"]["roc_auc"]),
            "roc_fpr": [float(x) for x in fold_result["train_metrics"]["roc_fpr"]],
            "roc_tpr": [float(x) for x in fold_result["train_metrics"]["roc_tpr"]],
            "loss": float(fold_result["train_metrics"]["loss"]),
        },
        "val_metrics": {
            "accuracy": float(fold_result["val_metrics"]["accuracy"]),
            "macro_precision": float(fold_result["val_metrics"]["macro_precision"]),
            "macro_recall": float(fold_result["val_metrics"]["macro_recall"]),
            "macro_f1": float(fold_result["val_metrics"]["macro_f1"]),
            "roc_auc": float(fold_result["val_metrics"]["roc_auc"]),
            "roc_fpr": [float(x) for x in fold_result["val_metrics"]["roc_fpr"]],
            "roc_tpr": [float(x) for x in fold_result["val_metrics"]["roc_tpr"]],
            "loss": float(fold_result["val_metrics"]["loss"]),
        },
        "confusion_val": fold_result["confusion_val"].tolist(),
        "n_params": int(fold_result["n_params"]),
    }
    with open(fold_dir / "fold_result.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def load_fold_result_json(cfg: ModelConfig, fold_id: int):
    fold_dir = fold_dir_of(cfg, fold_id)
    json_path = fold_dir / "fold_result.json"
    if not json_path.exists():
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["confusion_val"] = np.array(data["confusion_val"], dtype=np.int64)
    for split in ["train_metrics", "val_metrics"]:
        data[split]["roc_fpr"] = np.array(data[split]["roc_fpr"], dtype=np.float32)
        data[split]["roc_tpr"] = np.array(data[split]["roc_tpr"], dtype=np.float32)
    return data


def save_final_result_json(final_dir: Path, result: dict):
    serializable = {
        "best_epoch": int(result["best_epoch"]),
        "best_holdout_acc": float(result["best_holdout_acc"]),
        "n_params": int(result["n_params"]),
        "history": {k: [float(x) for x in v] for k, v in result.get("history", {}).items()},
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
        "holdout_metrics": {
            "accuracy": float(result["holdout_metrics"]["accuracy"]),
            "macro_precision": float(result["holdout_metrics"]["macro_precision"]),
            "macro_recall": float(result["holdout_metrics"]["macro_recall"]),
            "macro_f1": float(result["holdout_metrics"]["macro_f1"]),
            "roc_auc": float(result["holdout_metrics"]["roc_auc"]),
            "roc_fpr": [float(x) for x in result["holdout_metrics"]["roc_fpr"]],
            "roc_tpr": [float(x) for x in result["holdout_metrics"]["roc_tpr"]],
            "loss": float(result["holdout_metrics"]["loss"]),
        },
    }
    with open(final_dir / "final_result.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def load_final_result_json(cfg: ModelConfig):
    final_dir = final_dir_of(cfg)
    json_path = final_dir / "final_result.json"
    if not json_path.exists():
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for split in ["train_metrics", "holdout_metrics"]:
        data[split]["roc_fpr"] = np.array(data[split]["roc_fpr"], dtype=np.float32)
        data[split]["roc_tpr"] = np.array(data[split]["roc_tpr"], dtype=np.float32)
    return data


def load_best_fold_model(cfg: ModelConfig, fold_id: int):
    ckpt_path = best_fold_ckpt_path(cfg, fold_id)
    if not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = build_model(cfg).to(DEVICE)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    return model, ckpt


def load_best_final_model(cfg: ModelConfig):
    ckpt_path = best_final_ckpt_path(cfg)
    if not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = build_model(cfg).to(DEVICE)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    return model, ckpt


def evaluate_existing_fold_checkpoint(cfg, fold_id, samples, train_idx, val_idx):
    loaded = load_best_fold_model(cfg, fold_id)
    if loaded is None:
        return None

    model, ckpt = loaded
    fold_dir = fold_dir_of(cfg, fold_id)
    train_loader, val_loader, train_samples, _ = make_loaders(samples, train_idx, val_idx, cfg)
    class_weights = compute_class_weights(train_samples).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_metrics = run_epoch(model, train_loader, criterion, DEVICE, optimizer=None)
    val_metrics = run_epoch(model, val_loader, criterion, DEVICE, optimizer=None)
    conf_val = confusion_matrix(val_metrics["y_true"], val_metrics["y_pred"], labels=list(range(len(LABELS))))

    fold_result = {
        "fold": fold_id,
        "best_epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1,
        "best_val_acc": float(ckpt.get("best_val_acc", val_metrics["accuracy"])) if isinstance(ckpt, dict) else float(val_metrics["accuracy"]),
        "history": {},
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "confusion_val": conf_val,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    save_fold_result_json(fold_dir, fold_result)
    return fold_result


def evaluate_existing_final_checkpoint(cfg, train_pool_samples, holdout_samples):
    loaded = load_best_final_model(cfg)
    if loaded is None:
        return None

    model, ckpt = loaded
    final_dir = final_dir_of(cfg)

    train_pool_indices = list(range(len(train_pool_samples)))
    balanced_idx = balance_train_indices(train_pool_indices, train_pool_samples)
    balanced_train_samples = [train_pool_samples[i] for i in balanced_idx]

    train_ds = CellDataset(balanced_train_samples, pretrained=cfg.pretrained, train=True)
    holdout_ds = CellDataset(holdout_samples, pretrained=cfg.pretrained, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=USE_AMP)
    holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=USE_AMP)

    class_weights = compute_class_weights(balanced_train_samples).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_metrics = run_epoch(model, train_loader, criterion, DEVICE, optimizer=None)
    holdout_metrics = run_epoch(model, holdout_loader, criterion, DEVICE, optimizer=None)

    result = {
        "best_epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1,
        "best_holdout_acc": float(ckpt.get("best_holdout_acc", holdout_metrics["accuracy"])) if isinstance(ckpt, dict) else float(holdout_metrics["accuracy"]),
        "history": {},
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    save_final_result_json(final_dir, result)
    return result


def scan_model_resume_status(cfg: ModelConfig):
    missing_fold_json_with_ckpt = []
    for fold_id in range(1, N_SPLITS + 1):
        ckpt_exists = best_fold_ckpt_path(cfg, fold_id).exists()
        json_exists = (fold_dir_of(cfg, fold_id) / "fold_result.json").exists()
        if ckpt_exists and not json_exists:
            missing_fold_json_with_ckpt.append(fold_id)

    final_ckpt_exists = best_final_ckpt_path(cfg).exists()
    final_json_exists = (final_dir_of(cfg) / "final_result.json").exists()
    missing_final_json_with_ckpt = final_ckpt_exists and (not final_json_exists)

    return {
        "missing_fold_json_with_ckpt": missing_fold_json_with_ckpt,
        "missing_final_json_with_ckpt": missing_final_json_with_ckpt,
    }


def train_one_fold(cfg: ModelConfig, fold_id, samples, train_idx, val_idx, resume=True, force_retrain=False):
    fold_dir = fold_dir_of(cfg, fold_id)
    ensure_dir(fold_dir)

    if resume and not force_retrain:
        loaded_json = load_fold_result_json(cfg, fold_id)
        if loaded_json is not None:
            print(f"[Resume] Loaded cached json for {cfg.name} fold {fold_id}")
            return loaded_json

        ckpt_exists = best_fold_ckpt_path(cfg, fold_id).exists()
        if ckpt_exists:
            print(f"[Resume] Evaluating existing best_model.pt for {cfg.name} fold {fold_id}")
            evaluated = evaluate_existing_fold_checkpoint(cfg, fold_id, samples, train_idx, val_idx)
            if evaluated is not None:
                return evaluated

    train_loader, val_loader, train_samples, _ = make_loaders(samples, train_idx, val_idx, cfg)

    model = build_model(cfg).to(DEVICE)
    class_weights = compute_class_weights(train_samples).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        train_metrics = run_epoch(model, train_loader, criterion, DEVICE, optimizer=optimizer, scaler=scaler, scheduler=scheduler)
        val_metrics = run_epoch(model, val_loader, criterion, DEVICE, optimizer=None, scaler=None, scheduler=None)

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
    torch.save(best_state, best_fold_ckpt_path(cfg, fold_id))

    model.load_state_dict(best_state["model_state"])
    best_train_metrics = run_epoch(model, train_loader, criterion, DEVICE, optimizer=None, scaler=None, scheduler=None)
    best_val_metrics = run_epoch(model, val_loader, criterion, DEVICE, optimizer=None, scaler=None, scheduler=None)

    conf_val = confusion_matrix(
        best_val_metrics["y_true"], best_val_metrics["y_pred"], labels=list(range(len(LABELS)))
    )

    fold_result = {
        "fold": fold_id,
        "best_epoch": best_state["epoch"],
        "best_val_acc": best_state["best_val_acc"],
        "history": history,
        "train_metrics": best_train_metrics,
        "val_metrics": best_val_metrics,
        "confusion_val": conf_val,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    save_fold_result_json(fold_dir, fold_result)
    return fold_result


def train_final_holdout_model(cfg: ModelConfig, train_pool_samples, holdout_samples, resume=True, force_retrain=False):
    final_dir = final_dir_of(cfg)
    ensure_dir(final_dir)

    if resume and not force_retrain:
        loaded_json = load_final_result_json(cfg)
        if loaded_json is not None:
            print(f"[Resume] Loaded cached final json for {cfg.name}")
            return loaded_json

        ckpt_exists = best_final_ckpt_path(cfg).exists()
        if ckpt_exists:
            print(f"[Resume] Evaluating existing best_final_model.pt for {cfg.name}")
            evaluated = evaluate_existing_final_checkpoint(cfg, train_pool_samples, holdout_samples)
            if evaluated is not None:
                return evaluated

    train_pool_indices = list(range(len(train_pool_samples)))
    balanced_idx = balance_train_indices(train_pool_indices, train_pool_samples)
    balanced_train_samples = [train_pool_samples[i] for i in balanced_idx]

    train_ds = CellDataset(balanced_train_samples, pretrained=cfg.pretrained, train=True)
    holdout_ds = CellDataset(holdout_samples, pretrained=cfg.pretrained, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=USE_AMP)
    holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=USE_AMP)

    model = build_model(cfg).to(DEVICE)
    class_weights = compute_class_weights(balanced_train_samples).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        train_metrics = run_epoch(model, train_loader, criterion, DEVICE, optimizer=optimizer, scaler=scaler, scheduler=scheduler)
        holdout_metrics = run_epoch(model, holdout_loader, criterion, DEVICE, optimizer=None, scaler=None, scheduler=None)

        history["train_loss"].append(train_metrics["loss"])
        history["holdout_loss"].append(holdout_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["holdout_acc"].append(holdout_metrics["accuracy"])

        if holdout_metrics["accuracy"] > best_metric:
            best_metric = holdout_metrics["accuracy"]
            best_state = {
                "model_state": copy.deepcopy(model.state_dict()),
                "epoch": epoch + 1,
                "best_holdout_acc": holdout_metrics["accuracy"],
            }

        print(
            f"[{cfg.name}][Final] Epoch {epoch+1:03d}/{NUM_EPOCHS} "
            f"train_loss={train_metrics['loss']:.4f} holdout_loss={holdout_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} holdout_acc={holdout_metrics['accuracy']:.4f}"
        )

    save_history_csv(final_dir / "final_training_history.csv", history)
    torch.save(best_state, best_final_ckpt_path(cfg))

    model.load_state_dict(best_state["model_state"])
    train_metrics = run_epoch(model, train_loader, criterion, DEVICE, optimizer=None, scaler=None, scheduler=None)
    holdout_metrics = run_epoch(model, holdout_loader, criterion, DEVICE, optimizer=None, scaler=None, scheduler=None)

    result = {
        "best_epoch": best_state["epoch"],
        "best_holdout_acc": best_state["best_holdout_acc"],
        "history": history,
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    save_final_result_json(final_dir, result)
    return result


def summarize_curves_asymmetric(curves):
    arr = np.array(curves, dtype=np.float32)
    center = arr.mean(axis=0)
    lower = np.quantile(arr, 0.16, axis=0)
    upper = np.quantile(arr, 0.84, axis=0)
    return center, lower, upper


def aggregate_model_results(cfg, fold_results, holdout_result):
    summary = {
        "model": cfg.name,
        "pretrained": cfg.pretrained,
        "kind": cfg.kind,
        "depth": cfg.depth,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "cv_folds": N_SPLITS,
        "n_params": int(fold_results[0]["n_params"]),
        "best_epoch_mean": float(np.mean([fr["best_epoch"] for fr in fold_results])),
        "best_epoch_std": float(np.std([fr["best_epoch"] for fr in fold_results], ddof=1) if len(fold_results) > 1 else 0.0),
        "best_val_acc_mean": float(np.mean([fr["best_val_acc"] for fr in fold_results])),
        "best_val_acc_std": float(np.std([fr["best_val_acc"] for fr in fold_results], ddof=1) if len(fold_results) > 1 else 0.0),
    }

    for split in ["train_metrics", "val_metrics"]:
        prefix = "train" if split == "train_metrics" else "val"
        for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc", "loss"]:
            vals = [fr[split][key] for fr in fold_results]
            summary[f"{prefix}_{key}_mean"] = float(np.mean(vals))
            summary[f"{prefix}_{key}_std"] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    summary["holdout_best_epoch"] = int(holdout_result["best_epoch"])
    summary["holdout_best_acc"] = float(holdout_result["best_holdout_acc"])
    summary["holdout_accuracy"] = float(holdout_result["holdout_metrics"]["accuracy"])
    summary["holdout_macro_f1"] = float(holdout_result["holdout_metrics"]["macro_f1"])
    summary["holdout_auc"] = float(holdout_result["holdout_metrics"]["roc_auc"])
    summary["holdout_loss"] = float(holdout_result["holdout_metrics"]["loss"])
    return summary


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
                if metric_key in fr.get("history", {}) and len(fr["history"][metric_key]) == NUM_EPOCHS:
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
    ncols = 3
    nrows = math.ceil(n_models / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.5 * nrows))
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
        ax.grid(False)

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

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.93, wspace=0.22, hspace=0.28)
    fig.savefig(FIG_DIR / "confusion_matrices_combined.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(SEED)
    ensure_dir(OUT_DIR)
    ensure_dir(MODELS_DIR)
    ensure_dir(FIG_DIR)

    samples = parse_samples(DATA_DIR)
    if len(samples) == 0:
        raise RuntimeError("No valid samples found in Dataset_100times.")

    df_all = pd.DataFrame(samples)
    print("Total samples:", len(df_all))
    print(df_all["label"].value_counts().reindex(LABELS, fill_value=0))

    y_all = np.array([s["label_idx"] for s in samples])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=HOLDOUT_RATIO, random_state=SEED)
    train_pool_idx, holdout_idx = next(splitter.split(np.arange(len(samples)), y_all))

    train_pool_samples = [samples[i] for i in train_pool_idx]
    holdout_samples = [samples[i] for i in holdout_idx]

    print("\nTrain pool counts:")
    print(pd.Series([s["label"] for s in train_pool_samples]).value_counts().reindex(LABELS, fill_value=0))
    print("\nHoldout counts:")
    print(pd.Series([s["label"] for s in holdout_samples]).value_counts().reindex(LABELS, fill_value=0))

    retrain_all_models = confirm_yes_no("Retrain ALL models from scratch? [Y/N]: ")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    y_train_pool = np.array([s["label_idx"] for s in train_pool_samples])

    all_results = {}
    summary_rows = []
    fold_rows = []

    for cfg in MODEL_CONFIGS:
        print(f"\n========== {cfg.name} ==========")
        fold_results = []

        if retrain_all_models:
            force_retrain_for_model = True
        else:
            force_retrain_for_model = False
            resume_status = scan_model_resume_status(cfg)

            if (
                len(resume_status["missing_fold_json_with_ckpt"]) > 0
                or resume_status["missing_final_json_with_ckpt"]
            ):
                msg_parts = []
                if len(resume_status["missing_fold_json_with_ckpt"]) > 0:
                    msg_parts.append(f"fold checkpoints without json: {resume_status['missing_fold_json_with_ckpt']}")
                if resume_status["missing_final_json_with_ckpt"]:
                    msg_parts.append("final checkpoint without json")
                msg = "; ".join(msg_parts)

                force_retrain_for_model = confirm_yes_no(
                    f"[{cfg.name}] Found checkpoint(s) but missing json ({msg}). Retrain this model? [Y/N]: "
                )

        for fold_id, (tr_rel, va_rel) in enumerate(skf.split(np.arange(len(train_pool_samples)), y_train_pool), start=1):
            fold_result = train_one_fold(
                cfg,
                fold_id,
                train_pool_samples,
                tr_rel,
                va_rel,
                resume=(not retrain_all_models),
                force_retrain=force_retrain_for_model,
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

        holdout_result = train_final_holdout_model(
            cfg,
            train_pool_samples,
            holdout_samples,
            resume=(not retrain_all_models),
            force_retrain=force_retrain_for_model,
        )

        all_results[cfg.name] = {
            "config": cfg,
            "fold_results": fold_results,
            "holdout_result": holdout_result,
        }

        summary_rows.append(aggregate_model_results(cfg, fold_results, holdout_result))

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
                    "HOLDOUT_RATIO": HOLDOUT_RATIO,
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