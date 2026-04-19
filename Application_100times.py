import re
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from cellpose import core
from cellpose import models as cp_models


# ============================================================
# Config
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "Dataset_raw" / "100times"
CELLPOSE_PROJECT_DIR = BASE_DIR / "Cellpose" / "HeLa_100times"
BENCHMARK_DIR = BASE_DIR / "Output" / "Benchmark_100times"
CLASSIFIER_MODELS_DIR = BENCHMARK_DIR / "saved_models"

OUT_DIR = BASE_DIR / "Application" / "100times_results"
VIS_DIR = OUT_DIR / "visualizations"
TABLE_DIR = OUT_DIR / "tables"
MASK_DIR = OUT_DIR / "masks"
META_DIR = OUT_DIR / "metadata"

BASE_CELLPOSE_MODEL = "cpsam"

BOX_SIZE = 800
IMG_SIZE = 224

DIAMETER = 600
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
MIN_SIZE = 2000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = ["I", "P", "M", "A", "T"]
IDX_TO_LABEL = {i: x for i, x in enumerate(LABELS)}
PHASE_TO_NUM = {"I": 1, "P": 2, "M": 3, "A": 4, "T": 5}

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

MODEL_CONFIGS = {
    "CNN_Small": {"kind": "cnn_small", "depth": None, "pretrained": False},
    "CNN_Deep": {"kind": "cnn_deep", "depth": None, "pretrained": False},
    "ResNet18_scratch": {"kind": "resnet", "depth": 18, "pretrained": False},
    "ResNet18_pretrained": {"kind": "resnet", "depth": 18, "pretrained": True},
    "ResNet34_scratch": {"kind": "resnet", "depth": 34, "pretrained": False},
    "ResNet34_pretrained": {"kind": "resnet", "depth": 34, "pretrained": True},
}

# Guidance 1: Tubulin bridge / telophase-like merge, only for pretrained cellpose fallback
GUIDANCE_DILATE_RADIUS = 7
GUIDANCE_GLOBAL_TUBULIN_PCT = 98.8
GUIDANCE_LOCAL_Z = 2.0
GUIDANCE_MIN_BRIGHT_PIXELS = 12
GUIDANCE_SIDE_RING_RADIUS = 14
GUIDANCE_SIDE_INNER_RADIUS = 3
GUIDANCE_MIN_SIDE_PIXELS = 18
GUIDANCE_COS_THR = 0.12
GUIDANCE_WEIGHTED_COS_THR = 0.18
GUIDANCE_MIN_GRAD_MAG = 6.0

# Guidance 2: adjacent Anaphase pair + single nuclear region
ANAPHASE_BAND_RADIUS = 10
ANAPHASE_DNA_PERCENTILE = 80
ANAPHASE_MIN_AREA = 40


# ============================================================
# Utilities
# ============================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def natural_key(s):
    s = str(s.stem if isinstance(s, Path) else s)
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def normalize_image(img, gamma_val=1.0, low_pct=1.0, high_pct=99.5):
    img = img.astype(np.float32)
    lo = np.percentile(img, low_pct)
    hi = np.percentile(img, high_pct)
    if hi <= lo:
        hi = lo + 1.0
    arr = (img - lo) / (hi - lo)
    arr = np.clip(arr, 0.0, 1.0)
    arr = arr ** gamma_val
    return (arr * 255).astype(np.uint8)


def normalize_gray(arr, low_pct=1.0, high_pct=99.5):
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, low_pct)
    hi = np.percentile(arr, high_pct)
    if hi <= lo:
        hi = lo + 1.0
    arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def normalize_dapi_100x(img):
    return normalize_image(img, gamma_val=1.05, low_pct=1.0, high_pct=98.8)


def read_gray(path: Path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def read_rgb(path: Path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_rgb(path: Path, arr: np.ndarray):
    Image.fromarray(arr).save(path)


def save_mask(path: Path, mask: np.ndarray):
    Image.fromarray(mask.astype(np.uint16)).save(path)


def relabel_sequential(mask: np.ndarray):
    out = np.zeros_like(mask, dtype=np.int32)
    labs = np.unique(mask)
    labs = labs[labs > 0]
    for new_id, old_id in enumerate(labs, start=1):
        out[mask == old_id] = new_id
    return out


# ============================================================
# Image pairing
# ============================================================
def is_v_id(name: str):
    return re.fullmatch(r"v\d+", name, flags=re.IGNORECASE) is not None


def iter_files(input_dir: Path):
    return sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS],
        key=natural_key
    )


def find_v_image_pairs(input_dir: Path):
    dna_map = {}
    tub_map = {}

    for f in iter_files(input_dir):
        stem = f.stem
        m_dna = re.match(r"^(.*)_DNA$", stem, flags=re.IGNORECASE)
        m_tub = re.match(r"^(.*)_Tubulin$", stem, flags=re.IGNORECASE)

        if m_dna:
            rid = m_dna.group(1)
            if is_v_id(rid):
                dna_map[rid] = f
        elif m_tub:
            rid = m_tub.group(1)
            if is_v_id(rid):
                tub_map[rid] = f

    common = sorted(set(dna_map) & set(tub_map), key=natural_key)
    return [(rid, dna_map[rid], tub_map[rid]) for rid in common]


def build_merge_image_100x(dna_path: Path, tub_path: Path):
    dna = read_gray(dna_path)
    tub = read_gray(tub_path)

    dna8 = normalize_dapi_100x(dna)
    tub8 = normalize_image(tub, gamma_val=1.0, low_pct=1.0, high_pct=99.5)

    img3 = np.zeros((dna8.shape[0], dna8.shape[1], 3), dtype=np.uint8)
    img3[:, :, 1] = tub8
    img3[:, :, 2] = dna8
    return img3, dna8, tub8, dna, tub


# ============================================================
# Cellpose model selection
# ============================================================
def find_latest_finetuned_cellpose_model(project_dir: Path):
    if not project_dir.exists():
        return None

    run_dirs = sorted(project_dir.glob("Run_*"), key=natural_key, reverse=True)
    for run_dir in run_dirs:
        models_dir = run_dir / "models"
        if not models_dir.exists():
            continue
        candidates = sorted(
            [p for p in models_dir.iterdir() if p.is_file() or p.is_dir()],
            key=natural_key
        )
        if len(candidates) > 0:
            return candidates[-1]
    return None


def make_cellpose_model(model_path_or_name):
    try:
        use_gpu = core.use_gpu()
    except Exception:
        use_gpu = False

    if model_path_or_name == BASE_CELLPOSE_MODEL:
        print(f"[Cellpose] Using pretrained {BASE_CELLPOSE_MODEL}")
        return cp_models.CellposeModel(gpu=use_gpu, model_type=BASE_CELLPOSE_MODEL)

    print(f"[Cellpose] Using fine-tuned model: {model_path_or_name}")
    return cp_models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path_or_name))


# ============================================================
# DAPI border-truncation filter
# ============================================================
def remove_border_truncated_nuclei_by_dapi(
    dapi_img,
    mask,
    margin=2,
    dapi_percentile=75,
    min_dapi_pixels=40,
):
    if dapi_img.shape != mask.shape:
        raise ValueError("dapi_img and mask shape mismatch")

    h, w = mask.shape
    out = mask.copy()
    labs = np.unique(mask)
    labs = labs[labs > 0]
    margin = max(int(margin), 1)

    for lid in labs:
        region = (mask == lid)
        vals = dapi_img[region]
        if vals.size == 0:
            continue

        thr = np.percentile(vals, dapi_percentile)
        nuc_region = region & (dapi_img >= thr)

        if nuc_region.sum() < min_dapi_pixels:
            continue

        touches = (
            nuc_region[:margin, :].any()
            or nuc_region[h - margin:, :].any()
            or nuc_region[:, :margin].any()
            or nuc_region[:, w - margin:].any()
        )

        if touches:
            out[out == lid] = 0

    return relabel_sequential(out)


# ============================================================
# Connectivity / adjacency helpers
# ============================================================
class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self):
        out = {}
        for x in self.parent:
            r = self.find(x)
            out.setdefault(r, []).append(x)
        return out


def get_adjacent_pairs(mask):
    pairs = set()

    right_a = mask[:, :-1]
    right_b = mask[:, 1:]
    valid = (right_a > 0) & (right_b > 0) & (right_a != right_b)
    ys, xs = np.where(valid)
    for y, x in zip(ys, xs):
        a = int(right_a[y, x])
        b = int(right_b[y, x])
        if a > b:
            a, b = b, a
        pairs.add((a, b))

    down_a = mask[:-1, :]
    down_b = mask[1:, :]
    valid = (down_a > 0) & (down_b > 0) & (down_a != down_b)
    ys, xs = np.where(valid)
    for y, x in zip(ys, xs):
        a = int(down_a[y, x])
        b = int(down_b[y, x])
        if a > b:
            a, b = b, a
        pairs.add((a, b))

    return sorted(pairs)


def make_pair_contact_band(mask, a, b, dilate_radius=6):
    h, w = mask.shape
    contact = np.zeros((h, w), dtype=np.uint8)

    left = mask[:, :-1]
    right = mask[:, 1:]
    valid = ((left == a) & (right == b)) | ((left == b) & (right == a))
    ys, xs = np.where(valid)
    contact[ys, xs] = 1
    contact[ys, np.minimum(xs + 1, w - 1)] = 1

    up = mask[:-1, :]
    down = mask[1:, :]
    valid = ((up == a) & (down == b)) | ((up == b) & (down == a))
    ys, xs = np.where(valid)
    contact[ys, xs] = 1
    contact[np.minimum(ys + 1, h - 1), xs] = 1

    if contact.sum() == 0:
        return contact.astype(bool)

    k = 2 * dilate_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    band = cv2.dilate(contact, kernel, iterations=1) > 0
    return band


def component_centroid(mask, lid):
    ys, xs = np.where(mask == lid)
    if len(xs) == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)


# ============================================================
# Guidance 1: Tubulin bridge / telophase-like merge
# ============================================================
def compute_tubulin_gradients(tub8):
    sm = cv2.GaussianBlur(tub8.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx * gx + gy * gy)
    return gx, gy, gm


def detect_tubulin_hotspot_between_pair(tub8, gx, gy, gm, mask, a, b):
    band = make_pair_contact_band(mask, a, b, dilate_radius=GUIDANCE_DILATE_RADIUS)
    if band.sum() == 0:
        return False, None, {}

    vals = tub8[band].astype(np.float32)
    if vals.size == 0:
        return False, None, {}

    global_thr = np.percentile(tub8, GUIDANCE_GLOBAL_TUBULIN_PCT)
    local_thr = vals.mean() + GUIDANCE_LOCAL_Z * vals.std()
    thr = max(global_thr, local_thr)

    bright = band & (tub8 >= thr)
    if int(bright.sum()) < GUIDANCE_MIN_BRIGHT_PIXELS:
        return False, None, {"reason": "not_enough_bright_pixels"}

    nlab, lab, stats, cents = cv2.connectedComponentsWithStats(bright.astype(np.uint8), connectivity=8)

    best_label = None
    best_score = None
    for lid in range(1, nlab):
        area = stats[lid, cv2.CC_STAT_AREA]
        ys, xs = np.where(lab == lid)
        if len(xs) == 0:
            continue
        score = float(area) * 1000.0 + float(tub8[ys, xs].max())
        if best_score is None or score > best_score:
            best_score = score
            best_label = lid

    if best_label is None:
        return False, None, {"reason": "no_bright_component"}

    hotspot_region = (lab == best_label)
    ys, xs = np.where(hotspot_region)
    intens = tub8[ys, xs]
    imax = int(np.argmax(intens))
    hotspot = np.array([float(xs[imax]), float(ys[imax])], dtype=np.float32)

    ca = component_centroid(mask, a)
    cb = component_centroid(mask, b)
    if ca is None or cb is None:
        return False, None, {"reason": "missing_centroid"}

    u_ab = cb - ca
    norm = float(np.linalg.norm(u_ab))
    if norm < 1e-6:
        return False, None, {"reason": "centroids_too_close"}
    u_ab = u_ab / norm
    u_ba = -u_ab

    # side regions near the boundary, but excluding immediate contact center
    band_outer = make_pair_contact_band(mask, a, b, dilate_radius=GUIDANCE_SIDE_RING_RADIUS)
    band_inner = make_pair_contact_band(mask, a, b, dilate_radius=GUIDANCE_SIDE_INNER_RADIUS)
    ring = band_outer & (~band_inner)

    side_a = ring & (mask == a)
    side_b = ring & (mask == b)

    if int(side_a.sum()) < GUIDANCE_MIN_SIDE_PIXELS or int(side_b.sum()) < GUIDANCE_MIN_SIDE_PIXELS:
        return False, None, {"reason": "not_enough_side_pixels"}

    def side_gradient_score(side_mask, unit_vec):
        ys, xs = np.where(side_mask)
        if len(xs) == 0:
            return -1.0, -1.0

        gxs = gx[ys, xs]
        gys = gy[ys, xs]
        mags = gm[ys, xs] + 1e-6

        projs = (gxs * unit_vec[0] + gys * unit_vec[1]) / mags
        mean_cos = float(np.mean(projs))
        weighted_cos = float(np.sum(projs * mags) / np.sum(mags))
        mean_mag = float(np.mean(mags))
        return mean_cos, weighted_cos, mean_mag

    mean_cos_a, weighted_cos_a, mean_mag_a = side_gradient_score(side_a, u_ab)
    mean_cos_b, weighted_cos_b, mean_mag_b = side_gradient_score(side_b, u_ba)

    grad_ok = (
        mean_mag_a >= GUIDANCE_MIN_GRAD_MAG
        and mean_mag_b >= GUIDANCE_MIN_GRAD_MAG
        and mean_cos_a >= GUIDANCE_COS_THR
        and mean_cos_b >= GUIDANCE_COS_THR
        and weighted_cos_a >= GUIDANCE_WEIGHTED_COS_THR
        and weighted_cos_b >= GUIDANCE_WEIGHTED_COS_THR
    )

    bright_ok = int(bright.sum()) >= GUIDANCE_MIN_BRIGHT_PIXELS

    should_merge = bright_ok and grad_ok

    debug = {
        "bright_pixels": int(bright.sum()),
        "mean_cos_a": mean_cos_a,
        "mean_cos_b": mean_cos_b,
        "weighted_cos_a": weighted_cos_a,
        "weighted_cos_b": weighted_cos_b,
        "mean_mag_a": mean_mag_a,
        "mean_mag_b": mean_mag_b,
    }

    return should_merge, (float(hotspot[0]), float(hotspot[1])), debug


def merge_adjacent_cells_with_tubulin_guidance(mask, tub8):
    labels = [int(x) for x in np.unique(mask) if x > 0]
    if len(labels) == 0:
        return relabel_sequential(mask), {}, {}

    pairs = get_adjacent_pairs(mask)
    if len(pairs) == 0:
        return relabel_sequential(mask), {}, {}

    uf = UnionFind(labels)
    pair_hotspots = []
    pair_debug = {}

    gx, gy, gm = compute_tubulin_gradients(tub8)

    for a, b in pairs:
        should_merge, hotspot, debug = detect_tubulin_hotspot_between_pair(tub8, gx, gy, gm, mask, a, b)
        pair_debug[f"{a}-{b}"] = debug
        if should_merge:
            uf.union(a, b)
            pair_hotspots.append((a, b, hotspot))

    groups = uf.groups()
    guidance_centers = {}
    out = np.zeros_like(mask, dtype=np.int32)
    group_members = {}

    new_id = 1
    for members in groups.values():
        members = sorted(members)
        comp_mask = np.isin(mask, members)
        out[comp_mask] = new_id
        group_members[new_id] = members

        comp_hotspots = []
        member_set = set(members)
        for a, b, hotspot in pair_hotspots:
            if a in member_set and b in member_set:
                comp_hotspots.append(hotspot)

        if len(comp_hotspots) > 0:
            best = None
            best_val = None
            for hx, hy in comp_hotspots:
                x = int(round(hx))
                y = int(round(hy))
                x = min(max(x, 0), tub8.shape[1] - 1)
                y = min(max(y, 0), tub8.shape[0] - 1)
                val = float(tub8[y, x])
                if best_val is None or val > best_val:
                    best_val = val
                    best = (float(x), float(y))
            guidance_centers[new_id] = best

        new_id += 1

    return relabel_sequential(out), guidance_centers, group_members


# ============================================================
# Guidance 2: adjacent Anaphase + single nuclear region
# ============================================================
def has_single_continuous_nuclear_region_between_pair(
    dna8,
    mask,
    a,
    b,
    band_radius=10,
    dna_percentile=80,
    min_area=40,
):
    band = make_pair_contact_band(mask, a, b, dilate_radius=band_radius)
    if band.sum() == 0:
        return False

    pair_region = ((mask == a) | (mask == b))
    roi = band & pair_region
    if roi.sum() == 0:
        return False

    vals = dna8[roi]
    if vals.size == 0:
        return False

    thr = np.percentile(vals, dna_percentile)
    nuc = roi & (dna8 >= thr)

    if nuc.sum() < min_area:
        return False

    nuc_u8 = nuc.astype(np.uint8)
    nlab, lab, stats, cents = cv2.connectedComponentsWithStats(nuc_u8, connectivity=8)

    kept = 0
    for lid in range(1, nlab):
        area = stats[lid, cv2.CC_STAT_AREA]
        if area >= min_area:
            kept += 1

    return kept == 1


def merge_adjacent_anaphase_pairs_with_single_nuclear_region(
    mask,
    dna8,
    pred_label_map,
):
    labels = [int(x) for x in np.unique(mask) if x > 0]
    if len(labels) == 0:
        return relabel_sequential(mask), {}

    pairs = get_adjacent_pairs(mask)
    if len(pairs) == 0:
        return relabel_sequential(mask), {}

    uf = UnionFind(labels)

    for a, b in pairs:
        la = pred_label_map.get(int(a), None)
        lb = pred_label_map.get(int(b), None)

        if la != "A" or lb != "A":
            continue

        single_region = has_single_continuous_nuclear_region_between_pair(
            dna8=dna8,
            mask=mask,
            a=a,
            b=b,
            band_radius=ANAPHASE_BAND_RADIUS,
            dna_percentile=ANAPHASE_DNA_PERCENTILE,
            min_area=ANAPHASE_MIN_AREA,
        )

        if single_region:
            uf.union(a, b)

    groups = uf.groups()
    out = np.zeros_like(mask, dtype=np.int32)
    new_group_members = {}

    new_id = 1
    for members in groups.values():
        members = sorted(members)
        comp_mask = np.isin(mask, members)
        out[comp_mask] = new_id
        new_group_members[new_id] = members
        new_id += 1

    return relabel_sequential(out), new_group_members


# ============================================================
# Classification models
# ============================================================
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
class LoadedClassifier:
    name: str
    pretrained_norm: bool
    model: nn.Module


def build_classifier(model_name: str):
    cfg = MODEL_CONFIGS[model_name]
    kind = cfg["kind"]
    depth = cfg["depth"]

    if kind == "cnn_small":
        return SmallCNN(num_classes=len(LABELS))
    if kind == "cnn_deep":
        return DeepCNN(num_classes=len(LABELS))
    if kind == "resnet":
        if depth == 18:
            model = models.resnet18(weights=None)
        elif depth == 34:
            model = models.resnet34(weights=None)
        else:
            raise ValueError(f"Unsupported depth: {depth}")
        model.fc = nn.Linear(model.fc.in_features, len(LABELS))
        return model

    raise ValueError(f"Unknown model kind: {kind}")


def load_available_classifiers():
    loaded = []

    if not CLASSIFIER_MODELS_DIR.exists():
        raise FileNotFoundError(f"Classifier model dir not found: {CLASSIFIER_MODELS_DIR}")

    for model_name in MODEL_CONFIGS.keys():
        ckpt_path = CLASSIFIER_MODELS_DIR / model_name / "final_trainpool" / "best_final_model.pt"
        if not ckpt_path.exists():
            continue

        model = build_classifier(model_name).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        loaded.append(
            LoadedClassifier(
                name=model_name,
                pretrained_norm=MODEL_CONFIGS[model_name]["pretrained"],
                model=model,
            )
        )

    if len(loaded) == 0:
        raise RuntimeError(
            f"No final classifier checkpoint found in {CLASSIFIER_MODELS_DIR}. "
            "Please train 100times classifiers first."
        )

    print("[Classifier] Loaded models:")
    for item in loaded:
        print(f"  - {item.name}")
    return loaded


# ============================================================
# Crop / preprocess / inference
# ============================================================
def crop_center_box(arr, cx, cy, box_size):
    half = box_size // 2
    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = x1 + box_size
    y2 = y1 + box_size

    h, w = arr.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return None, (x1, y1, x2, y2), False

    return arr[y1:y2, x1:x2].copy(), (x1, y1, x2, y2), True


def build_classifier_input_from_crop(dna_crop, tub_crop):
    dna = normalize_gray(dna_crop)
    tub = normalize_gray(tub_crop)
    merge = 0.5 * dna + 0.5 * tub

    rgb = np.stack([dna, tub, merge], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    return rgb


def tensorize_rgb(rgb_uint8, pretrained_norm=False):
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    x = tfm(rgb_uint8)
    if pretrained_norm:
        norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        x = norm(x)
    return x.unsqueeze(0).to(DEVICE)


@torch.no_grad()
def ensemble_predict(rgb_uint8, loaded_models):
    probs_all = []
    per_model_probs = {}

    for item in loaded_models:
        x = tensorize_rgb(rgb_uint8, pretrained_norm=item.pretrained_norm)
        logits = item.model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        probs_all.append(probs)
        per_model_probs[item.name] = probs.tolist()

    mean_probs = np.mean(np.stack(probs_all, axis=0), axis=0)
    pred_idx = int(np.argmax(mean_probs))
    pred_label = IDX_TO_LABEL[pred_idx]
    confidence = float(mean_probs[pred_idx])

    return {
        "pred_idx": pred_idx,
        "pred_label": pred_label,
        "pred_num": PHASE_TO_NUM[pred_label],
        "confidence": confidence,
        "mean_probs": mean_probs.tolist(),
        "per_model_probs": per_model_probs,
    }


# ============================================================
# Geometry / visualization
# ============================================================
def mask_centroid(mask, label_id):
    ys, xs = np.where(mask == label_id)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def draw_box(img, x1, y1, x2, y2, color=(255, 255, 0), thickness=2):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def overlay_mask_contours(img_rgb, mask, color=(0, 255, 255), thickness=1):
    overlay = img_rgb.copy()
    labs = [x for x in np.unique(mask) if x > 0]
    for lid in labs:
        binary = (mask == lid).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness)
    return overlay


def put_label(img, x, y, text, color=(255, 255, 255), bg=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thick = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    x = int(x)
    y = int(y)

    x1 = max(0, x)
    y1 = max(0, y - th - baseline - 4)
    x2 = min(img.shape[1] - 1, x + tw + 6)
    y2 = min(img.shape[0] - 1, y + 2)

    cv2.rectangle(img, (x1, y1), (x2, y2), bg, -1)
    cv2.putText(img, text, (x + 3, y - 3), font, scale, color, thick, cv2.LINE_AA)


# ============================================================
# Main pipeline
# ============================================================
def run_one_image(rid, dna_path, tub_path, cellpose_model, loaded_classifiers, using_finetuned):
    merge_img, dna8, tub8, dna_raw, tub_raw = build_merge_image_100x(dna_path, tub_path)

    print(f"\n[Image] {rid}")

    masks, flows, styles = cellpose_model.eval(
        merge_img,
        channels=[2, 3],
        diameter=DIAMETER,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        min_size=MIN_SIZE,
    )

    masks = remove_border_truncated_nuclei_by_dapi(
        dapi_img=dna8,
        mask=masks,
        margin=2,
        dapi_percentile=75,
        min_dapi_pixels=40,
    )
    masks = relabel_sequential(masks)

    # --------------------------------------------------------
    # guidance 1: tubulin bridge guidance for pretrained fallback
    # --------------------------------------------------------
    guidance_centers = {}
    guidance_members = {}
    if not using_finetuned:
        masks, guidance_centers, guidance_members = merge_adjacent_cells_with_tubulin_guidance(masks, tub8)
        print(f"[Guidance-bridge] merged components with bridge hotspots: {len(guidance_centers)}")

    # --------------------------------------------------------
    # first-pass classification
    # --------------------------------------------------------
    first_pass = {}
    valid_labels = [x for x in np.unique(masks) if x > 0]

    for lid in valid_labels:
        ctr = guidance_centers.get(int(lid), None)
        if ctr is None:
            ctr = mask_centroid(masks, lid)
        if ctr is None:
            continue

        cx, cy = ctr
        dna_crop, _, ok1 = crop_center_box(dna_raw, cx, cy, BOX_SIZE)
        tub_crop, _, ok2 = crop_center_box(tub_raw, cx, cy, BOX_SIZE)
        if not (ok1 and ok2):
            continue

        rgb_crop = build_classifier_input_from_crop(dna_crop, tub_crop)
        pred = ensemble_predict(rgb_crop, loaded_classifiers)

        first_pass[int(lid)] = {
            "center": (float(cx), float(cy)),
            "pred_label": pred["pred_label"],
            "pred_idx": int(pred["pred_idx"]),
        }

    # --------------------------------------------------------
    # guidance 2: adjacent Anaphase pair + single nuclear region
    # --------------------------------------------------------
    pred_label_map = {lid: info["pred_label"] for lid, info in first_pass.items()}
    masks_before_second = masks.copy()
    masks, second_merge_members = merge_adjacent_anaphase_pairs_with_single_nuclear_region(
        mask=masks,
        dna8=dna8,
        pred_label_map=pred_label_map,
    )
    masks = relabel_sequential(masks)

    # propagate guidance centers only for unchanged single components
    final_guidance_centers = {}
    for new_id, members in second_merge_members.items():
        if len(members) == 1 and members[0] in guidance_centers:
            final_guidance_centers[new_id] = guidance_centers[members[0]]

    vis = overlay_mask_contours(merge_img, masks, color=(0, 255, 255), thickness=1)

    rows = []
    detections = []
    valid_labels = [x for x in np.unique(masks) if x > 0]

    for lid in valid_labels:
        # final crop center:
        # if this final object is unchanged from bridge guidance, keep bridge hotspot
        # otherwise use current centroid after merges
        ctr = final_guidance_centers.get(int(lid), None)
        guidance_source = "bridge_hotspot" if ctr is not None else "centroid"

        if ctr is None:
            ctr = mask_centroid(masks, lid)
        if ctr is None:
            continue

        cx, cy = ctr

        dna_crop, (x1, y1, x2, y2), ok1 = crop_center_box(dna_raw, cx, cy, BOX_SIZE)
        tub_crop, _, ok2 = crop_center_box(tub_raw, cx, cy, BOX_SIZE)

        if not (ok1 and ok2):
            rows.append({
                "image_id": rid,
                "cell_id": int(lid),
                "cx": float(cx),
                "cy": float(cy),
                "box_x1": int(x1),
                "box_y1": int(y1),
                "box_x2": int(x2),
                "box_y2": int(y2),
                "status": "skip_out_of_bounds",
                "pred_idx": None,
                "pred_label": None,
                "pred_num": None,
                "confidence": None,
                "prob_I": None,
                "prob_P": None,
                "prob_M": None,
                "prob_A": None,
                "prob_T": None,
                "used_guidance_center": bool(guidance_source != "centroid"),
                "guidance_source": guidance_source,
                "per_model_probs_json": None,
            })
            continue

        rgb_crop = build_classifier_input_from_crop(dna_crop, tub_crop)
        pred = ensemble_predict(rgb_crop, loaded_classifiers)

        draw_box(vis, x1, y1, x2, y2, color=(255, 255, 0), thickness=2)
        cv2.drawMarker(
            vis,
            (int(round(cx)), int(round(cy))),
            (255, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
        )

        label_text = f"{pred['pred_num']}:{pred['pred_label']}"
        put_label(vis, x1 + 6, y1 + 28, label_text, color=(255, 255, 255), bg=(0, 0, 0))

        row = {
            "image_id": rid,
            "cell_id": int(lid),
            "cx": float(cx),
            "cy": float(cy),
            "box_x1": int(x1),
            "box_y1": int(y1),
            "box_x2": int(x2),
            "box_y2": int(y2),
            "status": "ok",
            "pred_idx": int(pred["pred_idx"]),
            "pred_label": pred["pred_label"],
            "pred_num": int(pred["pred_num"]),
            "confidence": float(pred["confidence"]),
            "prob_I": float(pred["mean_probs"][0]),
            "prob_P": float(pred["mean_probs"][1]),
            "prob_M": float(pred["mean_probs"][2]),
            "prob_A": float(pred["mean_probs"][3]),
            "prob_T": float(pred["mean_probs"][4]),
            "used_guidance_center": bool(guidance_source != "centroid"),
            "guidance_source": guidance_source,
            "per_model_probs_json": json.dumps(pred["per_model_probs"], ensure_ascii=False),
        }
        rows.append(row)

        detections.append({
            "cell_id": int(lid),
            "center": [float(cx), float(cy)],
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "label": pred["pred_label"],
            "label_idx": int(pred["pred_idx"]),
            "pred_num": int(pred["pred_num"]),
            "confidence": float(pred["confidence"]),
            "mean_probs": [float(v) for v in pred["mean_probs"]],
            "used_guidance_center": bool(guidance_source != "centroid"),
            "guidance_source": guidance_source,
        })

    save_mask(MASK_DIR / f"{rid}_masks.tif", masks)
    save_rgb(VIS_DIR / f"{rid}_prediction.png", vis)

    meta = {
        "image_id": rid,
        "device": str(DEVICE),
        "cellpose_model": str(find_latest_finetuned_cellpose_model(CELLPOSE_PROJECT_DIR)) if using_finetuned else BASE_CELLPOSE_MODEL,
        "classifier": "ensemble_final_trainpool",
        "box_size": BOX_SIZE,
        "using_finetuned_cellpose": bool(using_finetuned),
        "detections": detections,
    }
    with open(META_DIR / f"{rid}_prediction.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return rows


def main():
    ensure_dir(OUT_DIR)
    ensure_dir(VIS_DIR)
    ensure_dir(TABLE_DIR)
    ensure_dir(MASK_DIR)
    ensure_dir(META_DIR)

    pairs = find_v_image_pairs(RAW_DIR)
    if len(pairs) == 0:
        raise RuntimeError(f"No v* DNA/Tubulin image pairs found in {RAW_DIR}")

    finetuned_model = find_latest_finetuned_cellpose_model(CELLPOSE_PROJECT_DIR)
    using_finetuned = finetuned_model is not None

    if using_finetuned:
        cellpose_model = make_cellpose_model(finetuned_model)
    else:
        cellpose_model = make_cellpose_model(BASE_CELLPOSE_MODEL)

    loaded_classifiers = load_available_classifiers()

    all_rows = []
    for rid, dna_path, tub_path in pairs:
        rows = run_one_image(
            rid,
            dna_path,
            tub_path,
            cellpose_model,
            loaded_classifiers,
            using_finetuned=using_finetuned,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    summary_cols = [
        "image_id", "cell_id", "cx", "cy",
        "box_x1", "box_y1", "box_x2", "box_y2",
        "status",
        "pred_idx", "pred_label", "pred_num", "confidence",
        "prob_I", "prob_P", "prob_M", "prob_A", "prob_T",
        "used_guidance_center", "guidance_source",
        "per_model_probs_json",
    ]
    df = df.reindex(columns=summary_cols)
    df.to_csv(TABLE_DIR / "prediction_summary.csv", index=False, encoding="utf-8-sig")

    ok_df = df[df["status"] == "ok"].copy() if "status" in df.columns else pd.DataFrame()
    if len(ok_df) > 0:
        count_df = (
            ok_df.groupby(["image_id", "pred_label"])
            .size()
            .reset_index(name="count")
        )
        count_df.to_csv(TABLE_DIR / "prediction_counts_by_image.csv", index=False, encoding="utf-8-sig")

        total_count_df = (
            ok_df.groupby("pred_label")
            .size()
            .reindex(LABELS, fill_value=0)
            .reset_index()
        )
        total_count_df.columns = ["pred_label", "count"]
        total_count_df["pred_num"] = total_count_df["pred_label"].map(PHASE_TO_NUM)
        total_count_df.to_csv(TABLE_DIR / "prediction_counts_total.csv", index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Visualizations saved to: {VIS_DIR}")
    print(f"Tables saved to: {TABLE_DIR}")
    print(f"Masks saved to: {MASK_DIR}")
    print(f"Metadata saved to: {META_DIR}")


if __name__ == "__main__":
    main()