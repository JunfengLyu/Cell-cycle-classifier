# ============================================================
# Manifold_100times_learning.py
# Scheme A only:
#   1) image / layer3 / avgpool centroid combo on diffusion map
#   2) avgpool PCA / PHATE / diffusion combo
#
# Added logic:
#   If intermediate data already exist, ask user whether to relearn (Y/N)
# ============================================================

import re
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.linalg import eigh

import matplotlib.pyplot as plt
import seaborn as sns

import phate


# ============================================================
# Global config
# ============================================================

SEED = 42

DATA_DIR = Path("./Dataset/Dataset_100times")
TRAIN_OUT_DIR = Path("./Output/Benchmark_100times")
CKPT_PATH = TRAIN_OUT_DIR / "saved_models" / "ResNet34_pretrained" / "final_trainpool" / "best_final_model.pt"

OUT_DIR = TRAIN_OUT_DIR / "Manifold_learning"
FIG_DIR = OUT_DIR / "figures"
FEAT_DIR = OUT_DIR / "features"
TABLE_DIR = OUT_DIR / "tables"

LABELS = ["I", "P", "M", "A", "T"]
LABEL_TO_IDX = {x: i for i, x in enumerate(LABELS)}

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_PCA_DIM = 50
DIFFUSION_N_COMPONENTS = 5
DIFFUSION_SIGMA_MODE = "median"
DIFFUSION_ALPHA = 0.5
PHATE_DIM = 2

sns.set_style("white")

PHASE_COLORS = {
    "I": "#440154",
    "P": "#3b528b",
    "M": "#21918c",
    "A": "#5ec962",
    "T": "#fde725",
}


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


# ============================================================
# Dataset
# ============================================================

class CellDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        dna = load_gray_image(Path(s["dna_path"]))
        tub = load_gray_image(Path(s["tub_path"]))
        merge = 0.5 * dna + 0.5 * tub

        rgb = np.stack([dna, tub, merge], axis=-1)
        rgb_uint8 = (rgb * 255).astype(np.uint8)

        x = self.transform(rgb_uint8)

        meta = {
            "sample_id": s["sample_id"],
            "label": s["label"],
            "label_idx": s["label_idx"],
            "rgb_float": torch.from_numpy(rgb.astype(np.float32)),
        }
        return x, meta


# ============================================================
# Model
# ============================================================

def build_resnet34_head(num_classes=5):
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_trained_model(ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = build_resnet34_head(num_classes=len(LABELS)).to(DEVICE)

    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ============================================================
# Feature extraction
# ============================================================

@torch.no_grad()
def extract_representations(model, loader):
    layer3_feats = []
    avgpool_feats = []
    image_flat = []

    sample_ids = []
    labels = []
    label_idxs = []

    cache = {}

    def hook_layer3(module, inp, out):
        cache["layer3"] = out.mean(dim=(2, 3)).detach().cpu().numpy()

    def hook_avgpool(module, inp, out):
        cache["avgpool"] = out.view(out.size(0), -1).detach().cpu().numpy()

    h1 = model.layer3.register_forward_hook(hook_layer3)
    h2 = model.avgpool.register_forward_hook(hook_avgpool)

    for x, metas in loader:
        x = x.to(DEVICE, non_blocking=True)
        _ = model(x)

        batch_layer3 = cache["layer3"]
        batch_avgpool = cache["avgpool"]

        bs = len(metas["sample_id"])
        for i in range(bs):
            rgb = metas["rgb_float"][i].numpy()
            image_flat.append(rgb.reshape(-1))

        layer3_feats.append(batch_layer3)
        avgpool_feats.append(batch_avgpool)

        sample_ids.extend(list(metas["sample_id"]))
        labels.extend(list(metas["label"]))
        label_idxs.extend([int(v) for v in metas["label_idx"]])

    h1.remove()
    h2.remove()

    return {
        "sample_id": np.array(sample_ids),
        "label": np.array(labels),
        "label_idx": np.array(label_idxs, dtype=np.int64),
        "image_flat": np.stack(image_flat, axis=0).astype(np.float32),
        "layer3": np.concatenate(layer3_feats, axis=0).astype(np.float32),
        "avgpool": np.concatenate(avgpool_feats, axis=0).astype(np.float32),
    }


# ============================================================
# Feature prep
# ============================================================

def zscore_fit_transform(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (X - mean) / std, mean, std


def prepare_feature_spaces(raw_dict):
    spaces = {}

    X_img = raw_dict["image_flat"]
    X_img_z, _, _ = zscore_fit_transform(X_img)
    pca_img = PCA(
        n_components=min(IMAGE_PCA_DIM, X_img_z.shape[0] - 1, X_img_z.shape[1]),
        random_state=SEED,
    )
    X_img_pca = pca_img.fit_transform(X_img_z).astype(np.float32)
    spaces["image_pca50"] = {
        "X": X_img_pca,
        "source": "image_flat",
        "prep": "zscore + PCA50",
        "explained_var_sum": float(np.sum(pca_img.explained_variance_ratio_)),
    }

    X_l3 = raw_dict["layer3"]
    X_l3_z, _, _ = zscore_fit_transform(X_l3)
    spaces["layer3"] = {
        "X": X_l3_z.astype(np.float32),
        "source": "layer3_gap",
        "prep": "zscore",
        "explained_var_sum": np.nan,
    }

    X_ap = raw_dict["avgpool"]
    X_ap_z, _, _ = zscore_fit_transform(X_ap)
    spaces["avgpool"] = {
        "X": X_ap_z.astype(np.float32),
        "source": "avgpool_prefc",
        "prep": "zscore",
        "explained_var_sum": np.nan,
    }

    return spaces


# ============================================================
# Manifold methods
# ============================================================

def build_rbf_affinity(X, sigma="median"):
    D = pairwise_distances(X, metric="euclidean")
    if sigma == "median":
        tri = D[np.triu_indices_from(D, k=1)]
        tri = tri[tri > 0]
        s = np.median(tri) if len(tri) > 0 else 1.0
        if not np.isfinite(s) or s <= 0:
            s = 1.0
    else:
        s = float(sigma)

    K = np.exp(-(D ** 2) / (2.0 * (s ** 2)))
    return K, D, s


def diffusion_map_embedding(X, n_components=5, alpha=0.5, sigma="median"):
    K, _, used_sigma = build_rbf_affinity(X, sigma=sigma)

    q = np.sum(K, axis=1)
    q_alpha = np.power(q, -alpha, where=(q > 0), out=np.zeros_like(q))
    K_tilde = (q_alpha[:, None] * K) * q_alpha[None, :]

    d = np.sum(K_tilde, axis=1)
    d_inv = np.divide(1.0, d, out=np.zeros_like(d), where=(d > 0))

    D_half = np.diag(np.sqrt(d_inv))
    S = D_half @ K_tilde @ D_half

    evals, evecs = eigh(S)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    evals_use = evals[1:n_components + 1]
    evecs_use = evecs[:, 1:n_components + 1]

    psi = D_half @ evecs_use
    coords = psi * evals_use[None, :]

    return coords.astype(np.float32), evals_use.astype(np.float32), {
        "sigma": float(used_sigma),
        "all_evals": evals.astype(np.float32),
    }


def compute_pca2(X):
    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(X).astype(np.float32)
    evr = pca.explained_variance_ratio_.astype(np.float32)
    return Z, evr


def compute_phate2(X):
    op = phate.PHATE(
        n_components=PHATE_DIM,
        knn=10,
        random_state=SEED,
        verbose=False,
    )
    Z = op.fit_transform(X).astype(np.float32)
    return Z


def analyze_space(X, space_name):
    print(f"\n[Analyze] {space_name}")

    Z_pca2, evr = compute_pca2(X)
    Z_phate2 = compute_phate2(X)
    Z_diff, evals, extra = diffusion_map_embedding(
        X,
        n_components=DIFFUSION_N_COMPONENTS,
        alpha=DIFFUSION_ALPHA,
        sigma=DIFFUSION_SIGMA_MODE,
    )
    Z_diff2 = Z_diff[:, :2]

    return {
        "pca2": Z_pca2,
        "phate2": Z_phate2,
        "diffusion_full": Z_diff,
        "diffusion2": Z_diff2,
        "metrics": {
            "space": space_name,
            "feature_dim": int(X.shape[1]),
            "pca2_explained_var_1": float(evr[0]),
            "pca2_explained_var_2": float(evr[1]),
            "diffusion_sigma": float(extra["sigma"]),
            "diffusion_eval1": float(evals[0]) if len(evals) > 0 else np.nan,
            "diffusion_eval2": float(evals[1]) if len(evals) > 1 else np.nan,
        },
    }


# ============================================================
# Cache I/O
# ============================================================

def cache_paths():
    return {
        "meta_csv": TABLE_DIR / "sample_metadata.csv",
        "summary_csv": TABLE_DIR / "manifold_metrics_summary.csv",
        "summary_json": OUT_DIR / "analysis_summary.json",
        "coords_csv": TABLE_DIR / "embedding_coordinates.csv",
        "image_feat": FEAT_DIR / "image_pca50_features.npy",
        "layer3_feat": FEAT_DIR / "layer3_features.npy",
        "avgpool_feat": FEAT_DIR / "avgpool_features.npy",
        "image_pca2": FEAT_DIR / "image_pca50_pca2.npy",
        "image_phate2": FEAT_DIR / "image_pca50_phate2.npy",
        "image_diff2": FEAT_DIR / "image_pca50_diffusion2.npy",
        "layer3_pca2": FEAT_DIR / "layer3_pca2.npy",
        "layer3_phate2": FEAT_DIR / "layer3_phate2.npy",
        "layer3_diff2": FEAT_DIR / "layer3_diffusion2.npy",
        "avgpool_pca2": FEAT_DIR / "avgpool_pca2.npy",
        "avgpool_phate2": FEAT_DIR / "avgpool_phate2.npy",
        "avgpool_diff2": FEAT_DIR / "avgpool_diffusion2.npy",
    }


def intermediate_cache_exists():
    paths = cache_paths()
    need = [
        paths["meta_csv"],
        paths["summary_csv"],
        paths["image_feat"],
        paths["layer3_feat"],
        paths["avgpool_feat"],
        paths["image_pca2"],
        paths["image_phate2"],
        paths["image_diff2"],
        paths["layer3_pca2"],
        paths["layer3_phate2"],
        paths["layer3_diff2"],
        paths["avgpool_pca2"],
        paths["avgpool_phate2"],
        paths["avgpool_diff2"],
    ]
    return all(p.exists() for p in need)


def save_outputs(raw_dict, spaces, results):
    paths = cache_paths()

    meta_df = pd.DataFrame({
        "sample_id": raw_dict["sample_id"],
        "label": raw_dict["label"],
        "label_idx": raw_dict["label_idx"],
    })
    meta_df.to_csv(paths["meta_csv"], index=False)

    np.save(paths["image_feat"], spaces["image_pca50"]["X"])
    np.save(paths["layer3_feat"], spaces["layer3"]["X"])
    np.save(paths["avgpool_feat"], spaces["avgpool"]["X"])

    np.save(paths["image_pca2"], results["image_pca50"]["pca2"])
    np.save(paths["image_phate2"], results["image_pca50"]["phate2"])
    np.save(paths["image_diff2"], results["image_pca50"]["diffusion2"])

    np.save(paths["layer3_pca2"], results["layer3"]["pca2"])
    np.save(paths["layer3_phate2"], results["layer3"]["phate2"])
    np.save(paths["layer3_diff2"], results["layer3"]["diffusion2"])

    np.save(paths["avgpool_pca2"], results["avgpool"]["pca2"])
    np.save(paths["avgpool_phate2"], results["avgpool"]["phate2"])
    np.save(paths["avgpool_diff2"], results["avgpool"]["diffusion2"])

    rows = []
    for key in ["image_pca50", "layer3", "avgpool"]:
        row = results[key]["metrics"].copy()
        row["prep"] = spaces[key]["prep"]
        row["source"] = spaces[key]["source"]
        row["explained_var_sum_pre"] = spaces[key]["explained_var_sum"]
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(paths["summary_csv"], index=False)

    coord_rows = []
    for key in ["image_pca50", "layer3", "avgpool"]:
        for i in range(len(raw_dict["sample_id"])):
            coord_rows.append({
                "space": key,
                "sample_id": raw_dict["sample_id"][i],
                "label": raw_dict["label"][i],
                "pca1": results[key]["pca2"][i, 0],
                "pca2": results[key]["pca2"][i, 1],
                "phate1": results[key]["phate2"][i, 0],
                "phate2": results[key]["phate2"][i, 1],
                "diff1": results[key]["diffusion2"][i, 0],
                "diff2": results[key]["diffusion2"][i, 1],
            })
    pd.DataFrame(coord_rows).to_csv(paths["coords_csv"], index=False)

    with open(paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def load_cached_outputs():
    paths = cache_paths()

    meta_df = pd.read_csv(paths["meta_csv"])
    raw_dict = {
        "sample_id": meta_df["sample_id"].astype(str).to_numpy(),
        "label": meta_df["label"].astype(str).to_numpy(),
        "label_idx": meta_df["label_idx"].astype(int).to_numpy(),
    }

    spaces = {
        "image_pca50": {
            "X": np.load(paths["image_feat"]),
            "source": "image_flat",
            "prep": "zscore + PCA50",
            "explained_var_sum": np.nan,
        },
        "layer3": {
            "X": np.load(paths["layer3_feat"]),
            "source": "layer3_gap",
            "prep": "zscore",
            "explained_var_sum": np.nan,
        },
        "avgpool": {
            "X": np.load(paths["avgpool_feat"]),
            "source": "avgpool_prefc",
            "prep": "zscore",
            "explained_var_sum": np.nan,
        },
    }

    results = {
        "image_pca50": {
            "pca2": np.load(paths["image_pca2"]),
            "phate2": np.load(paths["image_phate2"]),
            "diffusion2": np.load(paths["image_diff2"]),
            "metrics": {},
        },
        "layer3": {
            "pca2": np.load(paths["layer3_pca2"]),
            "phate2": np.load(paths["layer3_phate2"]),
            "diffusion2": np.load(paths["layer3_diff2"]),
            "metrics": {},
        },
        "avgpool": {
            "pca2": np.load(paths["avgpool_pca2"]),
            "phate2": np.load(paths["avgpool_phate2"]),
            "diffusion2": np.load(paths["avgpool_diff2"]),
            "metrics": {},
        },
    }

    summary_csv = paths["summary_csv"]
    if summary_csv.exists():
        summary_df = pd.read_csv(summary_csv)
        for key in ["image_pca50", "layer3", "avgpool"]:
            row = summary_df[summary_df["space"] == key]
            if len(row) > 0:
                results[key]["metrics"] = row.iloc[0].to_dict()

    return raw_dict, spaces, results


# ============================================================
# Plotting
# ============================================================

def plot_phase_centroid_combo(manifold_results, labels):
    keys = ["image_pca50", "layer3", "avgpool"]
    titles = {
        "image_pca50": "Image space (PCA50)",
        "layer3": "Layer3 GAP",
        "avgpool": "AvgPool pre-fc",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9))

    for ax, key in zip(axes, keys):
        Z = manifold_results[key]["diffusion2"]

        for lab in LABELS:
            idx = np.where(labels == lab)[0]
            ax.scatter(
                Z[idx, 0],
                Z[idx, 1],
                s=34,
                alpha=0.6,
                color=PHASE_COLORS[lab],
                edgecolors="none",
            )


        centers = []
        for lab in LABELS:
            idx = np.where(labels == lab)[0]
            centers.append(Z[idx].mean(axis=0))
        centers = np.stack(centers, axis=0)

        ax.plot(
            centers[:, 0],
            centers[:, 1],
            linestyle="--",
            color="gray",
            linewidth=2,
            alpha=0.8,
        )

        for i, lab in enumerate(LABELS):
            ax.scatter(
                centers[i, 0],
                centers[i, 1],
                s=120,
                marker="s",
                color=PHASE_COLORS[lab],
                alpha=0.9,
                edgecolors="black",
                linewidths=1.2,
                zorder=5,
            )

        ax.set_title(titles[key], fontsize=14)
        ax.set_xlabel("Diffusion 1", fontsize=11)
        ax.set_ylabel("Diffusion 2", fontsize=11)
        ax.grid(False)

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.14, top=0.90, wspace=0.22)
    fig.savefig(FIG_DIR / "schemeA_phase_centroid_combo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_avgpool_schemeA(manifold_results, labels):
    key = "avgpool"
    panels = [
        ("pca2", "AvgPool pre-fc | PCA", "Dim 1", "Dim 2"),
        ("phate2", "AvgPool pre-fc | PHATE", "Dim 1", "Dim 2"),
        ("diffusion2", "AvgPool pre-fc | Diffusion map", "Diffusion 1", "Diffusion 2"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    for ax, (field, title, xlabel, ylabel) in zip(axes, panels):
        Z = manifold_results[key][field]

        for lab in LABELS:
            idx = np.where(labels == lab)[0]
            ax.scatter(
                Z[idx, 0],
                Z[idx, 1],
                s=34,
                alpha=0.8,
                color=PHASE_COLORS[lab],
                edgecolors="none",
                label=lab,
            )

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(False)

    axes[-1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.14, top=0.90, wspace=0.24)
    fig.savefig(FIG_DIR / "schemeA_avgpool_PCA_PHATE_Diffusion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)
    ensure_dir(OUT_DIR)
    ensure_dir(FIG_DIR)
    ensure_dir(FEAT_DIR)
    ensure_dir(TABLE_DIR)

    samples = parse_samples(DATA_DIR)
    if len(samples) == 0:
        raise RuntimeError(f"No valid samples found in {DATA_DIR}")

    print("Total samples:", len(samples))
    print(pd.Series([s["label"] for s in samples]).value_counts().reindex(LABELS, fill_value=0))

    use_cache = False
    if intermediate_cache_exists():
        relearn = confirm_yes_no("Cached intermediate data found. Relearn manifold representations? [Y/N]: ")
        use_cache = not relearn

    if use_cache:
        print("[Resume] Loading cached intermediate data.")
        raw_dict, spaces, results = load_cached_outputs()
    else:
        model = load_trained_model(CKPT_PATH)
        print(f"Loaded checkpoint: {CKPT_PATH}")

        ds = CellDataset(samples)
        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE.type == "cuda"),
        )

        raw_dict = extract_representations(model, loader)
        spaces = prepare_feature_spaces(raw_dict)

        results = {}
        for key in ["image_pca50", "layer3", "avgpool"]:
            results[key] = analyze_space(
                spaces[key]["X"],
                space_name=key,
            )

        save_outputs(raw_dict, spaces, results)

    plot_phase_centroid_combo(results, raw_dict["label"])
    plot_avgpool_schemeA(results, raw_dict["label"])

    print("\nDone.")
    print(f"Output dir: {OUT_DIR.resolve()}")
    print(f"Figures:    {FIG_DIR.resolve()}")
    print(f"Features:   {FEAT_DIR.resolve()}")
    print(f"Tables:     {TABLE_DIR.resolve()}")


if __name__ == "__main__":
    main()