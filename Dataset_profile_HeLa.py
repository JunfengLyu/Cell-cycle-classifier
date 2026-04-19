import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


DATASETS = [
    ("20times", Path("./Dataset/Dataset_20times")),
    ("100times", Path("./Dataset/Dataset_100times")),
]

OUTPUT_DIR = Path("./Output/Data_profile")
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


def parse_dataset_pairs(dataset_dir: Path):
    dna_files = sorted(dataset_dir.glob("*_DNA.png"), key=lambda p: natural_key(p.name))
    tub_files = sorted(dataset_dir.glob("*_Tubulin.png"), key=lambda p: natural_key(p.name))

    dna_map = {}
    tub_map = {}

    for f in dna_files:
        m = re.match(r"^(\d+)_([IPMAT])_DNA\.png$", f.name, flags=re.IGNORECASE)
        if m:
            idx, phase = m.group(1), m.group(2).upper()
            dna_map[(idx, phase)] = f

    for f in tub_files:
        m = re.match(r"^(\d+)_([IPMAT])_Tubulin\.png$", f.name, flags=re.IGNORECASE)
        if m:
            idx, phase = m.group(1), m.group(2).upper()
            tub_map[(idx, phase)] = f

    keys = sorted(set(dna_map) & set(tub_map), key=lambda x: (int(x[0]), x[1]))

    samples = []
    for idx, phase in keys:
        samples.append({
            "idx": idx,
            "phase": phase,
            "dna_path": dna_map[(idx, phase)],
            "tub_path": tub_map[(idx, phase)],
        })
    return samples


def read_gray_uint8(path: Path):
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def flatten_single_channel_feature(img8, downsample_to=64):
    small = np.array(
        Image.fromarray(img8).resize((downsample_to, downsample_to), Image.Resampling.BILINEAR)
    )
    feat = small.astype(np.float32).ravel() / 255.0
    return feat


def build_channel_features(samples, downsample_to=64):
    X_dna = []
    X_tub = []
    y = []
    dna_imgs = []
    tub_imgs = []

    for s in samples:
        dna8 = read_gray_uint8(s["dna_path"])
        tub8 = read_gray_uint8(s["tub_path"])

        X_dna.append(flatten_single_channel_feature(dna8, downsample_to=downsample_to))
        X_tub.append(flatten_single_channel_feature(tub8, downsample_to=downsample_to))
        y.append(s["phase"])
        dna_imgs.append(dna8)
        tub_imgs.append(tub8)

    if len(X_dna) == 0:
        return None, None, None, None, None

    X_dna = np.asarray(X_dna, dtype=np.float32)
    X_tub = np.asarray(X_tub, dtype=np.float32)
    y = np.asarray(y)

    return X_dna, X_tub, y, dna_imgs, tub_imgs


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


def make_blue_channel_image(gray_img):
    h, w = gray_img.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 2] = gray_img
    return rgb


def make_green_channel_image(gray_img):
    h, w = gray_img.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 1] = gray_img
    return rgb


def make_merge_image(dna8, tub8):
    h, w = dna8.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 2] = dna8
    rgb[:, :, 1] = tub8
    return rgb


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


def plot_pca_panel(ax, embedding, labels, title, var_ratio, show_legend=False):
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

    ax.set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}%)")
    ax.set_title(title)

    if show_legend:
        ax.legend(frameon=False, fontsize=9, loc="best")


def save_single_histogram(samples, dataset_name, save_path):
    fig, ax = plt.subplots(figsize=(9, 5.8))
    plot_phase_histogram(ax, samples, dataset_name)
    fig.tight_layout(pad=1.4)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_channel_pca_combo(emb_dna, emb_tub, labels, var_dna, var_tub, dataset_name, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))

    plot_pca_panel(
        axes[0], emb_dna, labels,
        f"{dataset_name}: DNA PCA",
        var_dna,
        show_legend=True,
    )
    plot_pca_panel(
        axes[1], emb_tub, labels,
        f"{dataset_name}: Tubulin PCA",
        var_tub,
        show_legend=False,
    )

    fig.tight_layout(pad=1.5)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_example_contact_sheet(dna_imgs, tub_imgs, labels, dataset_name, save_path, thumb_size=64, max_per_phase=6):
    grouped = defaultdict(list)
    for dna8, tub8, phase in zip(dna_imgs, tub_imgs, labels):
        grouped[phase].append((dna8, tub8))

    rows = len(PHASES)
    cols_per_phase = max_per_phase
    block_w = thumb_size * 3 + 20
    canvas_h = rows * thumb_size
    canvas_w = cols_per_phase * block_w

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for r, phase in enumerate(PHASES):
        pairs = grouped.get(phase, [])[:max_per_phase]
        for c, (dna8, tub8) in enumerate(pairs):
            dna_thumb = np.array(
                Image.fromarray(make_blue_channel_image(dna8)).resize(
                    (thumb_size, thumb_size), Image.Resampling.BILINEAR
                )
            )
            tub_thumb = np.array(
                Image.fromarray(make_green_channel_image(tub8)).resize(
                    (thumb_size, thumb_size), Image.Resampling.BILINEAR
                )
            )
            merge_thumb = np.array(
                Image.fromarray(make_merge_image(dna8, tub8)).resize(
                    (thumb_size, thumb_size), Image.Resampling.BILINEAR
                )
            )

            x0 = c * block_w
            y0 = r * thumb_size

            canvas[y0:y0 + thumb_size, x0:x0 + thumb_size] = dna_thumb
            canvas[y0:y0 + thumb_size, x0 + thumb_size:x0 + 2 * thumb_size] = tub_thumb
            canvas[y0:y0 + thumb_size, x0 + 2 * thumb_size:x0 + 3 * thumb_size] = merge_thumb

    fig, ax = plt.subplots(figsize=(max(10, cols_per_phase * 2.2), 7.0))
    ax.imshow(canvas)
    ax.set_xticks([])
    ax.set_yticks([(i + 0.5) * thumb_size for i in range(rows)])
    ax.set_yticklabels([PHASE_NAMES[p] for p in PHASES])

    ax.set_title(f"{dataset_name}: DNA (blue), Tubulin (green), Merge examples")

    ax.text(0.5 * thumb_size, -10, "DNA", ha="center", va="bottom", fontsize=11, color="blue")
    ax.text(1.5 * thumb_size, -10, "Tubulin", ha="center", va="bottom", fontsize=11, color="green")
    ax.text(2.5 * thumb_size, -10, "Merge", ha="center", va="bottom", fontsize=11, color="black")

    fig.tight_layout(pad=1.5)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_full_combo(samples, emb_dna, emb_tub, labels, var_dna, var_tub, dataset_name, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4))

    plot_phase_histogram(axes[0], samples, dataset_name)
    plot_pca_panel(
        axes[1], emb_dna, labels,
        f"{dataset_name}: DNA PCA",
        var_dna,
        show_legend=True,
    )
    plot_pca_panel(
        axes[2], emb_tub, labels,
        f"{dataset_name}: Tubulin PCA",
        var_tub,
        show_legend=False,
    )

    fig.tight_layout(pad=1.6)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_one_dataset(dataset_name, dataset_dir):
    samples = parse_dataset_pairs(dataset_dir)

    if len(samples) == 0:
        print(f"[Skip] No valid samples found in {dataset_dir}")
        return

    print(f"[{dataset_name}] valid cells: {len(samples)}")

    X_dna, X_tub, y, dna_imgs, tub_imgs = build_channel_features(samples, downsample_to=64)

    emb_dna, var_dna = compute_pca_2d(X_dna)
    emb_tub, var_tub = compute_pca_2d(X_tub)

    hist_path = OUTPUT_DIR / f"{dataset_name}_phase_histogram.png"
    pca_combo_path = OUTPUT_DIR / f"{dataset_name}_DNA_Tubulin_PCA_combo.png"
    example_path = OUTPUT_DIR / f"{dataset_name}_channel_examples.png"
    full_combo_path = OUTPUT_DIR / f"{dataset_name}_profile_combo.png"

    save_single_histogram(samples, dataset_name, hist_path)
    save_channel_pca_combo(emb_dna, emb_tub, y, var_dna, var_tub, dataset_name, pca_combo_path)
    make_example_contact_sheet(dna_imgs, tub_imgs, y, dataset_name, example_path, thumb_size=64, max_per_phase=6)
    save_full_combo(samples, emb_dna, emb_tub, y, var_dna, var_tub, dataset_name, full_combo_path)

    print("Saved:")
    print(f"  {hist_path}")
    print(f"  {pca_combo_path}")
    print(f"  {example_path}")
    print(f"  {full_combo_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_name, dataset_dir in DATASETS:
        process_one_dataset(dataset_name, dataset_dir)


if __name__ == "__main__":
    main()