import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


DATASET_NAME = "Feulgen_40times"
DATASET_DIR = Path("./Dataset/Dataset_Feulgen_40times")
OUTPUT_DIR = Path("./Output/Data_profile/Feulgen")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHASES = ["I", "P", "M", "A", "T"]
PHASE_NAMES = {
    "I": "Interphase",
    "P": "Prophase",
    "M": "Metaphase",
    "A": "Anaphase",
    "T": "Telophase",
}

PHASE_COLORS = {
    "I": plt.cm.plasma(0.08),
    "P": plt.cm.plasma(0.28),
    "M": plt.cm.plasma(0.50),
    "A": plt.cm.plasma(0.72),
    "T": plt.cm.plasma(0.92),
}


def natural_key(s):
    parts = re.split(r"(\d+)", str(s))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def parse_feulgen_dataset(dataset_dir: Path):
    files = sorted(dataset_dir.glob("*_DNA.png"), key=lambda p: natural_key(p.name))

    samples = []
    for f in files:
        m = re.match(r"^(\d+)_([IPMAT])_DNA\.png$", f.name, flags=re.IGNORECASE)
        if not m:
            continue
        idx, phase = m.group(1), m.group(2).upper()
        samples.append({
            "idx": idx,
            "phase": phase,
            "img_path": f,
        })
    return samples


def read_rgb_uint8(path: Path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def flatten_rgb_feature(img_rgb, downsample_to=64):
    small = np.array(
        Image.fromarray(img_rgb).resize((downsample_to, downsample_to), Image.Resampling.BILINEAR)
    )
    feat = small.astype(np.float32).reshape(-1) / 255.0
    return feat


def build_rgb_features(samples, downsample_to=64):
    X = []
    y = []
    imgs = []

    for s in samples:
        img = read_rgb_uint8(s["img_path"])
        X.append(flatten_rgb_feature(img, downsample_to=downsample_to))
        y.append(s["phase"])
        imgs.append(img)

    if len(X) == 0:
        return None, None, None

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    return X, y, imgs


def compute_pca_2d(X):
    n_samples = X.shape[0]
    n_features = X.shape[1]

    if n_samples == 1:
        return np.array([[0.0, 0.0]], dtype=float), np.array([1.0, 0.0])

    n_components = min(2, n_samples, n_features)
    pca = PCA(n_components=n_components, random_state=42)
    emb = pca.fit_transform(X)

    if emb.shape[1] == 1:
        emb = np.column_stack([emb[:, 0], np.zeros(len(emb), dtype=float)])
        var_ratio = np.array([pca.explained_variance_ratio_[0], 0.0])
    else:
        var_ratio = pca.explained_variance_ratio_

    return emb, var_ratio


def compute_tsne_2d(X):
    n_samples = X.shape[0]
    if n_samples == 1:
        return np.array([[0.0, 0.0]], dtype=float)

    pca_dim = min(50, X.shape[1], max(2, n_samples - 1))
    X_pca = PCA(n_components=pca_dim, random_state=42).fit_transform(X)

    perplexity = min(30, max(5, (n_samples - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    emb = tsne.fit_transform(X_pca)
    return emb


def plot_phase_histogram(ax, samples, dataset_name):
    phase_counts = Counter(s["phase"] for s in samples)
    counts = np.array([phase_counts.get(p, 0) for p in PHASES], dtype=int)
    colors = [PHASE_COLORS[p] for p in PHASES]

    bars = ax.bar(range(len(PHASES)), counts, color=colors, width=0.72)

    ax.set_xticks(range(len(PHASES)))
    ax.set_xticklabels([PHASE_NAMES[p] for p in PHASES], rotation=20)
    ax.set_ylabel("Cell count")
    ax.set_title(f"{dataset_name}: phase histogram")

    ymax = max(counts.max(), 1)
    ax.set_ylim(0, ymax * 1.24 + 1)

    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.04 + 0.05,
            str(c),
            ha="center",
            va="bottom",
            fontsize=10,
            clip_on=False,
        )


def plot_embedding_panel(ax, embedding, labels, title, xlabel, ylabel, show_legend=False):
    for phase in PHASES:
        mask = labels == phase
        if np.any(mask):
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=34,
                alpha=0.85,
                color=PHASE_COLORS[phase],
                label=PHASE_NAMES[phase] if show_legend else None,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if show_legend:
        ax.legend(frameon=False, fontsize=9, loc="best")


def save_single_histogram(samples, dataset_name, save_path):
    fig, ax = plt.subplots(figsize=(9, 5.8))
    plot_phase_histogram(ax, samples, dataset_name)
    fig.tight_layout(pad=1.4)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_rgb_pca(embedding, labels, var_ratio, dataset_name, save_path):
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    plot_embedding_panel(
        ax, embedding, labels,
        f"{dataset_name}: RGB PCA",
        f"PC1 ({var_ratio[0] * 100:.1f}%)",
        f"PC2 ({var_ratio[1] * 100:.1f}%)",
        show_legend=True,
    )
    fig.tight_layout(pad=1.4)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_rgb_tsne(embedding, labels, dataset_name, save_path):
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    plot_embedding_panel(
        ax, embedding, labels,
        f"{dataset_name}: RGB t-SNE",
        "t-SNE 1",
        "t-SNE 2",
        show_legend=True,
    )
    fig.tight_layout(pad=1.4)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_example_contact_sheet(imgs, labels, dataset_name, save_path, thumb_size=96, max_per_phase=6):
    grouped = defaultdict(list)
    for img, phase in zip(imgs, labels):
        grouped[phase].append(img)

    rows = len(PHASES)
    cols = max_per_phase
    canvas_h = rows * thumb_size
    canvas_w = cols * thumb_size

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    for r, phase in enumerate(PHASES):
        phase_imgs = grouped.get(phase, [])[:max_per_phase]
        for c, img in enumerate(phase_imgs):
            thumb = np.array(
                Image.fromarray(img).resize((thumb_size, thumb_size), Image.Resampling.BILINEAR)
            )
            x0 = c * thumb_size
            y0 = r * thumb_size
            canvas[y0:y0 + thumb_size, x0:x0 + thumb_size] = thumb

    fig, ax = plt.subplots(figsize=(max(10, cols * 1.8), 7.2))
    ax.imshow(canvas)
    ax.set_xticks([])
    ax.set_yticks([(i + 0.5) * thumb_size for i in range(rows)])
    ax.set_yticklabels([PHASE_NAMES[p] for p in PHASES])
    ax.set_title(f"{dataset_name}: example cells")
    fig.tight_layout(pad=1.5)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_full_combo(samples, emb_pca, var_ratio, emb_tsne, labels, imgs, dataset_name, save_path):
    fig = plt.figure(figsize=(23, 5.6))

    ax1 = fig.add_subplot(1, 4, 1)
    plot_phase_histogram(ax1, samples, dataset_name)

    ax2 = fig.add_subplot(1, 4, 2)
    plot_embedding_panel(
        ax2, emb_pca, labels,
        f"{dataset_name}: RGB PCA",
        f"PC1 ({var_ratio[0] * 100:.1f}%)",
        f"PC2 ({var_ratio[1] * 100:.1f}%)",
        show_legend=True,
    )

    ax3 = fig.add_subplot(1, 4, 3)
    plot_embedding_panel(
        ax3, emb_tsne, labels,
        f"{dataset_name}: RGB t-SNE",
        "t-SNE 1",
        "t-SNE 2",
        show_legend=False,
    )

    ax4 = fig.add_subplot(1, 4, 4)

    grouped = defaultdict(list)
    for img, phase in zip(imgs, labels):
        grouped[phase].append(img)

    thumb_size = 64
    max_per_phase = 4
    rows = len(PHASES)
    cols = max_per_phase
    canvas_h = rows * thumb_size
    canvas_w = cols * thumb_size
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    for r, phase in enumerate(PHASES):
        phase_imgs = grouped.get(phase, [])[:max_per_phase]
        for c, img in enumerate(phase_imgs):
            thumb = np.array(
                Image.fromarray(img).resize((thumb_size, thumb_size), Image.Resampling.BILINEAR)
            )
            x0 = c * thumb_size
            y0 = r * thumb_size
            canvas[y0:y0 + thumb_size, x0:x0 + thumb_size] = thumb

    ax4.imshow(canvas)
    ax4.set_xticks([])
    ax4.set_yticks([(i + 0.5) * thumb_size for i in range(rows)])
    ax4.set_yticklabels([PHASE_NAMES[p] for p in PHASES])
    ax4.set_title(f"{dataset_name}: examples")

    fig.tight_layout(pad=1.5)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    samples = parse_feulgen_dataset(DATASET_DIR)

    if len(samples) == 0:
        print(f"[Skip] No valid samples found in {DATASET_DIR}")
        return

    print(f"[{DATASET_NAME}] valid cells: {len(samples)}")

    X, y, imgs = build_rgb_features(samples, downsample_to=64)

    emb_pca, var_ratio = compute_pca_2d(X)
    emb_tsne = compute_tsne_2d(X)

    hist_path = OUTPUT_DIR / f"{DATASET_NAME}_phase_histogram.png"
    pca_path = OUTPUT_DIR / f"{DATASET_NAME}_RGB_PCA.png"
    tsne_path = OUTPUT_DIR / f"{DATASET_NAME}_RGB_tSNE.png"
    example_path = OUTPUT_DIR / f"{DATASET_NAME}_examples.png"
    full_combo_path = OUTPUT_DIR / f"{DATASET_NAME}_tSNE_examples_combo.png"

    save_single_histogram(samples, DATASET_NAME, hist_path)
    save_rgb_pca(emb_pca, y, var_ratio, DATASET_NAME, pca_path)
    save_rgb_tsne(emb_tsne, y, DATASET_NAME, tsne_path)
    make_example_contact_sheet(imgs, y, DATASET_NAME, example_path, thumb_size=96, max_per_phase=6)
    save_full_combo(samples, emb_pca, var_ratio, emb_tsne, y, imgs, DATASET_NAME, full_combo_path)

    print("Saved:")
    print(f"  {hist_path}")
    print(f"  {pca_path}")
    print(f"  {tsne_path}")
    print(f"  {example_path}")
    print(f"  {full_combo_path}")


if __name__ == "__main__":
    main()