import re
from pathlib import Path
from collections import Counter
import json

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from cellpose import models as cp_models, core


BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "Dataset_raw" / "Feulgen_40times"
OUT_DIR = BASE_DIR / "Application" / "Feulgen_40times_results"

CELLPOSE_PROJECT_DIR = BASE_DIR / "Cellpose" / "Feulgen_40times"
CLASSIFIER_DIR = BASE_DIR / "Output" / "Benchmark_Feulgen_40times" / "saved_models" / "ResNet34_pretrained"

LABELS = ["I", "P", "M", "A", "T"]
LABEL_TO_IDX = {x: i for i, x in enumerate(LABELS)}
IDX_TO_LABEL = {i: x for x, i in LABEL_TO_IDX.items()}

IMG_SIZE = 224
BOX_SIZE = 100
N_FOLDS = 5

CELLPOSE_BASE_MODEL = "cpsam"
CELLPOSE_DIAMETER = 80
CELLPOSE_FLOW_THRESHOLD = 0.4
CELLPOSE_CELLPROB_THRESHOLD = 0.0
CELLPOSE_MIN_SIZE = 60

BORDER_MARGIN = 2
PURPLE_PERCENTILE = 75
MIN_PURPLE_PIXELS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# visualization style
BOX_COLOR = (20, 70, 170)      # deep blue in RGB array
CENTER_COLOR = (20, 70, 170)
TEXT_COLOR = (20, 70, 170)
TEXT_BG_COLOR = (255, 255, 255)
BOX_THICKNESS = 2
CONTOUR_COLOR = (0, 255, 0)
CONTOUR_THICKNESS = 1
MARKER_SIZE = 10
MARKER_THICKNESS = 2
FONT_SCALE = 0.7
FONT_THICKNESS = 2


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def natural_key(path_or_str):
    s = str(path_or_str.stem if isinstance(path_or_str, Path) else path_or_str)
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def read_rgb(path: Path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


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


def find_v_images(raw_dir: Path):
    files = [p for p in raw_dir.iterdir() if p.is_file()]
    imgs = []
    for f in files:
        if re.fullmatch(r"v\d+", f.stem, flags=re.IGNORECASE):
            imgs.append(f)
    return sorted(imgs, key=natural_key)


def relabel_sequential(mask):
    out = np.zeros_like(mask, dtype=np.int32)
    labs = np.unique(mask)
    labs = labs[labs > 0]
    for i, lid in enumerate(labs, start=1):
        out[mask == lid] = i
    return out


def remove_border_truncated_nuclei_by_purple(img_rgb, mask):
    h, w = mask.shape
    out = mask.copy()

    rgb = img_rgb.astype(np.float32)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    purple_score = 0.5 * (r + b) - g

    labs = np.unique(mask)
    labs = labs[labs > 0]

    for lid in labs:
        region = (mask == lid)
        vals = purple_score[region]
        if vals.size == 0:
            continue

        thr = np.percentile(vals, PURPLE_PERCENTILE)
        purple_region = region & (purple_score >= thr)

        if purple_region.sum() < MIN_PURPLE_PIXELS:
            continue

        touches = (
            purple_region[:BORDER_MARGIN, :].any()
            or purple_region[h - BORDER_MARGIN:, :].any()
            or purple_region[:, :BORDER_MARGIN].any()
            or purple_region[:, w - BORDER_MARGIN:].any()
        )
        if touches:
            out[out == lid] = 0

    return relabel_sequential(out)


def latest_cellpose_model(project_dir: Path):
    if not project_dir.exists():
        return None

    run_dirs = sorted(
        [p for p in project_dir.glob("Run_*") if p.is_dir()],
        key=natural_key
    )
    if not run_dirs:
        return None

    for run_dir in reversed(run_dirs):
        models_dir = run_dir / "models"
        if models_dir.exists():
            cands = sorted(
                [p for p in models_dir.iterdir() if p.is_file()],
                key=lambda p: p.stat().st_mtime
            )
            if cands:
                return str(cands[-1])

    return None


def build_cellpose_model():
    model_path = latest_cellpose_model(CELLPOSE_PROJECT_DIR)

    try:
        use_gpu = core.use_gpu()
    except Exception:
        use_gpu = False

    if model_path is None:
        print("[Cellpose] Using pretrained cpsam")
        return cp_models.CellposeModel(gpu=use_gpu, model_type=CELLPOSE_BASE_MODEL)
    else:
        print(f"[Cellpose] Using fine-tuned model: {model_path}")
        return cp_models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)


def mask_centroid(mask, lid):
    ys, xs = np.where(mask == lid)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


# ============================================================
# crop logic: strictly matched to Dataset_cellcrop_Feulgen.py
# ============================================================
def crop_with_padding_rgb(img, cx, cy, box_size):
    h, w = img.shape[:2]
    half = box_size // 2

    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = x1 + box_size
    y2 = y1 + box_size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if any([pad_left, pad_top, pad_right, pad_bottom]):
        img = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=255,
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    crop = img[y1:y2, x1:x2, :]
    return crop


def classifier_box_coords(cx, cy, box_size):
    half = box_size // 2
    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = x1 + box_size
    y2 = y1 + box_size
    return x1, y1, x2, y2


# ============================================================
# Classifier: matched to Training_Feulgen_40times.py
# ============================================================
def build_resnet34(num_classes=5):
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def strip_prefix_if_needed(state_dict, prefix="module."):
    keys = list(state_dict.keys())
    if len(keys) > 0 and all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def make_classifier_input(rgb_crop):
    arr = normalize_rgb(rgb_crop)
    arr = (arr * 255).astype(np.uint8)

    img = Image.fromarray(arr, mode="RGB")
    img = TF.resize(img, [IMG_SIZE, IMG_SIZE], interpolation=InterpolationMode.BILINEAR)

    x = TF.to_tensor(img)
    x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return x.unsqueeze(0)


def load_fold_models():
    fold_models = []

    for fold_id in range(1, N_FOLDS + 1):
        ckpt_path = CLASSIFIER_DIR / f"fold_{fold_id}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"[Warning] Missing fold model: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location=DEVICE)

        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        else:
            state_dict = ckpt

        state_dict = strip_prefix_if_needed(state_dict, prefix="module.")

        model = build_resnet34(num_classes=len(LABELS)).to(DEVICE)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        fold_models.append(model)

    if not fold_models:
        raise FileNotFoundError(f"No fold models found in: {CLASSIFIER_DIR}")

    print(f"[Classifier] Loaded {len(fold_models)} fold models.")
    return fold_models


@torch.no_grad()
def vote_predict(models_list, x):
    votes = []
    probs_all = []

    x = x.to(DEVICE)

    for model in models_list:
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        pred = int(np.argmax(probs))
        votes.append(pred)
        probs_all.append(probs)

    vote_counter = Counter(votes)
    top_count = max(vote_counter.values())
    tied = [k for k, v in vote_counter.items() if v == top_count]

    probs_mean = np.mean(np.stack(probs_all, axis=0), axis=0)

    if len(tied) == 1:
        final_pred = tied[0]
    else:
        final_pred = max(tied, key=lambda k: probs_mean[k])

    return final_pred, probs_mean, votes


def mask_to_contours(mask_binary):
    contours, _ = cv2.findContours(
        mask_binary.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def draw_label_with_bg(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, FONT_SCALE, FONT_THICKNESS)

    x1 = max(0, int(x))
    y1 = max(0, int(y - th - baseline - 6))
    x2 = min(img.shape[1] - 1, int(x + tw + 8))
    y2 = min(img.shape[0] - 1, int(y + 2))

    cv2.rectangle(img, (x1, y1), (x2, y2), TEXT_BG_COLOR, -1)
    cv2.putText(
        img, text, (x1 + 4, y2 - 4),
        font, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA
    )


def draw_results(img_rgb, masks, detections, out_path: Path):
    vis = img_rgb.copy()

    for lid in sorted([x for x in np.unique(masks) if x > 0]):
        region = (masks == lid).astype(np.uint8)
        contours = mask_to_contours(region)
        cv2.drawContours(vis, contours, -1, CONTOUR_COLOR, CONTOUR_THICKNESS)

    for i, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["box"]
        cx, cy = det["center"]
        label = det["label"]

        # clip only for drawing, not for crop logic
        x1d = max(0, x1)
        y1d = max(0, y1)
        x2d = min(vis.shape[1] - 1, x2)
        y2d = min(vis.shape[0] - 1, y2)

        cv2.rectangle(vis, (x1d, y1d), (x2d, y2d), BOX_COLOR, BOX_THICKNESS)
        cv2.drawMarker(
            vis,
            (int(round(cx)), int(round(cy))),
            CENTER_COLOR,
            markerType=cv2.MARKER_CROSS,
            markerSize=MARKER_SIZE,
            thickness=MARKER_THICKNESS,
        )

        text = f"{i}:{label}"
        draw_label_with_bg(vis, text, x1d + 4, max(24, y1d + 24))

    Image.fromarray(vis).save(out_path)


def main():
    ensure_dir(OUT_DIR)

    image_paths = find_v_images(RAW_DIR)
    if not image_paths:
        print(f"No v* images found in {RAW_DIR}")
        return

    print(f"[Device] {DEVICE}")
    cp_model = build_cellpose_model()
    clf_models = load_fold_models()

    current_cp_model = latest_cellpose_model(CELLPOSE_PROJECT_DIR) or CELLPOSE_BASE_MODEL

    for img_path in image_paths:
        rid = img_path.stem
        print(f"\n[Application] {rid}")

        img_rgb = read_rgb(img_path)

        masks, flows, styles = cp_model.eval(
            img_rgb,
            channels=[0, 0],
            diameter=CELLPOSE_DIAMETER,
            flow_threshold=CELLPOSE_FLOW_THRESHOLD,
            cellprob_threshold=CELLPOSE_CELLPROB_THRESHOLD,
            min_size=CELLPOSE_MIN_SIZE,
        )

        masks = remove_border_truncated_nuclei_by_purple(img_rgb, masks)

        detections = []
        valid_labels = sorted([x for x in np.unique(masks) if x > 0])
        print(f"[Cellpose] detected cells: {len(valid_labels)}")

        for lid in valid_labels:
            c = mask_centroid(masks, lid)
            if c is None:
                continue

            cx, cy = c

            # strictly matched to your cellcrop_feulgen padding crop
            crop_rgb = crop_with_padding_rgb(img_rgb, cx, cy, BOX_SIZE)

            x = make_classifier_input(crop_rgb)
            pred_idx, probs_mean, votes = vote_predict(clf_models, x)
            label = IDX_TO_LABEL[pred_idx]

            x1, y1, x2, y2 = classifier_box_coords(cx, cy, BOX_SIZE)

            detections.append(
                {
                    "cell_id": int(lid),
                    "center": [float(cx), float(cy)],
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "label": label,
                    "label_idx": int(pred_idx),
                    "votes": [int(v) for v in votes],
                    "mean_probs": [float(v) for v in probs_mean.tolist()],
                }
            )

        draw_results(img_rgb, masks, detections, OUT_DIR / f"{rid}_application.png")

        with open(OUT_DIR / f"{rid}_application.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image_id": rid,
                    "device": str(DEVICE),
                    "cellpose_model": current_cp_model,
                    "classifier": "ResNet34_pretrained",
                    "box_size": BOX_SIZE,
                    "detections": detections,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    print(f"\nDone. Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()