import re
import sys
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cellpose import models, core, utils


# ============================================================
# Config
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "Dataset_raw" / "100times"
PROJECT_DIR = BASE_DIR / "Cellpose" / "HeLa_100times"

BASE_MODEL = "cpsam"

DIAMETER = 600
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
MIN_SIZE = 2000

BORDER_MARGIN = 2
DAPI_PERCENTILE = 75
MIN_DAPI_PIXELS = 40

GUIDE_DILATE = 4
GUIDE_MIN_CONTACT_PIXELS = 10
GUIDE_BRIGHT_PERCENTILE = 99.7
GUIDE_SCORE_PERCENTILE = 99
GUIDE_MAX_SUGGESTIONS = 8

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.1
N_EPOCHS = 10
TRAIN_BATCH_SIZE = 1
MIN_TRAIN_MASKS = 1

VALID_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


# ============================================================
# Utilities
# ============================================================
def natural_key(path_or_str):
    s = str(path_or_str.stem if isinstance(path_or_str, Path) else path_or_str)
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def is_holdout_id(name: str):
    return re.fullmatch(r"v\d+", name, flags=re.IGNORECASE) is not None


def iter_files(input_dir: Path):
    return sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS],
        key=natural_key
    )


def read_gray(path: Path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def read_rgb(path: Path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_rgb(path: Path, img):
    Image.fromarray(img).save(path)


def read_mask(path: Path):
    arr = np.array(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D: {path}")
    return arr.astype(np.int32)


def save_mask(path: Path, mask: np.ndarray):
    Image.fromarray(mask.astype(np.uint16)).save(path)


def relabel_sequential(mask):
    out = np.zeros_like(mask, dtype=np.int32)
    labs = np.unique(mask)
    labs = labs[labs > 0]
    for new_id, old_id in enumerate(labs, start=1):
        out[mask == old_id] = new_id
    return out


def yes_no(prompt, default=True):
    suffix = "[Y/n]" if default else "[y/N]"
    s = input(f"{prompt} {suffix}: ").strip().lower()
    if s == "":
        return default
    return s in {"y", "yes"}


def latest_run_idx(project_dir: Path):
    if not project_dir.exists():
        return 0
    nums = []
    for p in project_dir.iterdir():
        if p.is_dir():
            m = re.match(r"^Run_(\d+)$", p.name)
            if m:
                nums.append(int(m.group(1)))
    return max(nums) if nums else 0


def make_run_dirs(run_dir: Path):
    for name in ["images", "predictions", "corrected", "train", "models"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def get_previous_model(project_dir: Path):
    last_idx = latest_run_idx(project_dir)
    if last_idx == 0:
        return BASE_MODEL

    prev_models = project_dir / f"Run_{last_idx:02d}" / "models"
    if not prev_models.exists():
        return BASE_MODEL

    candidates = sorted(prev_models.iterdir(), key=natural_key)
    return str(candidates[-1]) if candidates else BASE_MODEL


def normalize_image(img, gamma_val=1.0, low_pct=1.0, high_pct=99.5):
    p1 = np.percentile(img, low_pct)
    p99 = np.percentile(img, high_pct)
    if p99 <= p1:
        p99 = p1 + 1.0
    arr = (img - p1) / (p99 - p1)
    arr = np.clip(arr, 0.0, 1.0)
    arr = arr ** gamma_val
    return (arr * 255).astype(np.uint8)


def normalize_dapi_100x(img):
    return normalize_image(img, gamma_val=1.05, low_pct=1, high_pct=98.8)


def find_image_pairs(input_dir: Path):
    files = iter_files(input_dir)
    dna_map, tub_map = {}, {}

    for f in files:
        stem = f.stem
        m_dna = re.match(r"^(.*)_DNA$", stem, flags=re.IGNORECASE)
        m_tub = re.match(r"^(.*)_Tubulin$", stem, flags=re.IGNORECASE)
        if m_dna:
            rid = m_dna.group(1)
            if not is_holdout_id(rid):
                dna_map[rid] = f
        elif m_tub:
            rid = m_tub.group(1)
            if not is_holdout_id(rid):
                tub_map[rid] = f

    common = sorted(set(dna_map) & set(tub_map), key=natural_key)
    return [(rid, dna_map[rid], tub_map[rid]) for rid in common]


def build_merge_image_100x(dna_path: Path, tub_path: Path):
    dna = read_gray(dna_path)
    tub = read_gray(tub_path)
    tub8 = normalize_image(tub, gamma_val=1.0, low_pct=1, high_pct=99.5)
    dna8 = normalize_dapi_100x(dna)

    h, w = dna8.shape
    img3 = np.zeros((h, w, 3), dtype=np.uint8)
    img3[:, :, 1] = tub8
    img3[:, :, 2] = dna8
    return img3, dna8, tub8


def get_mask_path(run_dir: Path, img_path: Path):
    return run_dir / "corrected" / f"{img_path.stem}_masks.tif"


def valid_labels(mask):
    labs = np.unique(mask)
    return set(int(x) for x in labs if x > 0)


# ============================================================
# DAPI-truncation filter
# ============================================================
def remove_border_truncated_nuclei_by_dapi(
    dapi_img,
    mask,
    margin=BORDER_MARGIN,
    dapi_percentile=DAPI_PERCENTILE,
    min_dapi_pixels=MIN_DAPI_PIXELS,
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
# Tubulin guidance
# ============================================================
def dilate_mask(binary, radius):
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(binary.astype(np.uint8), kernel, iterations=1).astype(bool)


def find_merge_suggestions(mask, tub_img):
    labs = sorted([x for x in np.unique(mask) if x > 0])
    if len(labs) < 2:
        return []

    bright_thr = np.percentile(tub_img, GUIDE_BRIGHT_PERCENTILE)
    binaries = {lid: (mask == lid) for lid in labs}
    dilated = {lid: dilate_mask(binaries[lid], GUIDE_DILATE) for lid in labs}

    suggestions = []

    for i in range(len(labs)):
        a = labs[i]
        for j in range(i + 1, len(labs)):
            b = labs[j]

            zone = dilated[a] & dilated[b]
            if zone.sum() < GUIDE_MIN_CONTACT_PIXELS:
                continue

            bridge_vals = tub_img[zone]
            if bridge_vals.size == 0:
                continue

            score = float(np.percentile(bridge_vals, GUIDE_SCORE_PERCENTILE))
            peak = float(bridge_vals.max())

            if peak >= bright_thr:
                suggestions.append({
                    "pair": (a, b),
                    "score": score,
                    "peak": peak,
                    "pixels": int(zone.sum()),
                })

    suggestions.sort(key=lambda x: (x["peak"], x["score"], x["pixels"]), reverse=True)
    return suggestions[:GUIDE_MAX_SUGGESTIONS]


# ============================================================
# Prediction
# ============================================================
def make_model(model_path):
    try:
        use_gpu = core.use_gpu()
    except Exception:
        use_gpu = False

    if model_path == BASE_MODEL:
        return models.CellposeModel(gpu=use_gpu, model_type=BASE_MODEL)
    return models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)


def copy_images_to_run(run_dir: Path):
    dst_dir = run_dir / "images"
    pairs = find_image_pairs(RAW_DIR)

    if not pairs:
        print(f"No eligible image pairs in {RAW_DIR}")
        return

    skipped = []
    files = iter_files(RAW_DIR)
    for f in files:
        stem = f.stem
        m = re.match(r"^(.*)_(DNA|Tubulin)$", stem, flags=re.IGNORECASE)
        if m and is_holdout_id(m.group(1)):
            skipped.append(f.name)
    if skipped:
        print(f"[Info] Holdout images skipped: {sorted(skipped)}")

    for rid, dna_path, tub_path in pairs:
        out_path = dst_dir / f"{rid}_merge.png"
        if not out_path.exists():
            img3, _, _ = build_merge_image_100x(dna_path, tub_path)
            save_rgb(out_path, img3)


def predict_run(run_dir: Path, model_path):
    model = make_model(model_path)
    img_dir = run_dir / "images"
    pred_dir = run_dir / "predictions"

    for img_path in sorted(img_dir.glob("*_merge.png"), key=natural_key):
        rid = img_path.stem.replace("_merge", "")
        dna_candidates = [p for p in iter_files(RAW_DIR) if p.stem == f"{rid}_DNA"]
        if not dna_candidates:
            print(f"Skip {rid}: DNA file not found")
            continue

        img = read_rgb(img_path)
        dapi = normalize_dapi_100x(read_gray(dna_candidates[0]))

        print(f"[Predict] {img_path.name}")
        masks, flows, styles = model.eval(
            img,
            channels=[2, 3],
            diameter=DIAMETER,
            flow_threshold=FLOW_THRESHOLD,
            cellprob_threshold=CELLPROB_THRESHOLD,
            min_size=MIN_SIZE,
        )

        masks = remove_border_truncated_nuclei_by_dapi(dapi_img=dapi, mask=masks)
        save_mask(pred_dir / f"{img_path.stem}_masks.tif", masks)


def seed_corrected_from_predictions(run_dir: Path):
    pred_dir = run_dir / "predictions"
    corr_dir = run_dir / "corrected"
    for mask_path in sorted(pred_dir.glob("*_masks.tif"), key=natural_key):
        dst = corr_dir / mask_path.name
        if not dst.exists():
            shutil.copy2(mask_path, dst)


# ============================================================
# Training
# ============================================================
def rebuild_train_folder(project_dir: Path, target_run_dir: Path):
    train_dir = target_run_dir / "train"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)

    for run_dir in sorted(project_dir.glob("Run_*"), key=natural_key):
        img_dir = run_dir / "images"
        corr_dir = run_dir / "corrected"
        if not img_dir.exists() or not corr_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*_merge.png"), key=natural_key):
            mask_path = corr_dir / f"{img_path.stem}_masks.tif"
            if not mask_path.exists():
                continue

            rid = img_path.stem.replace("_merge", "")
            dna_candidates = [p for p in iter_files(RAW_DIR) if p.stem == f"{rid}_DNA"]
            if not dna_candidates:
                continue
            dapi = normalize_dapi_100x(read_gray(dna_candidates[0]))
            mask = read_mask(mask_path)
            mask = remove_border_truncated_nuclei_by_dapi(dapi_img=dapi, mask=mask)

            shutil.copy2(img_path, train_dir / img_path.name)
            save_mask(train_dir / mask_path.name, mask)


def train_on_run(run_dir: Path):
    train_dir = run_dir / "train"
    model_name = f"{run_dir.name}_hela100x"

    cmd = [
        sys.executable, "-m", "cellpose",
        "--train",
        "--dir", str(train_dir.resolve()),
        "--mask_filter", "_masks",
        "--pretrained_model", BASE_MODEL,
        "--learning_rate", str(LEARNING_RATE),
        "--weight_decay", str(WEIGHT_DECAY),
        "--n_epochs", str(N_EPOCHS),
        "--train_batch_size", str(TRAIN_BATCH_SIZE),
        "--min_train_masks", str(MIN_TRAIN_MASKS),
        "--model_name_out", model_name,
        "--verbose",
    ]

    print("\n[Train Command]")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True, cwd=str(BASE_DIR))

    src_models_dir = train_dir / "models"
    dst_models_dir = run_dir / "models"
    dst_models_dir.mkdir(parents=True, exist_ok=True)

    if src_models_dir.exists():
        for p in src_models_dir.iterdir():
            if p.name.startswith(model_name):
                dst = dst_models_dir / p.name
                if dst.exists():
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                if p.is_dir():
                    shutil.copytree(p, dst)
                else:
                    shutil.copy2(p, dst)


# ============================================================
# Editor
# ============================================================
def mask_centroid(mask, label_id):
    ys, xs = np.where(mask == label_id)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def merge_two_labels(mask, a, b):
    if a <= 0 or b <= 0 or a == b:
        return mask
    out = mask.copy()
    out[out == b] = a
    return relabel_sequential(out)


def delete_one_label(mask, label_id):
    if label_id <= 0:
        return mask
    out = mask.copy()
    out[out == label_id] = 0
    return relabel_sequential(out)


def build_overlay(img_rgb, mask, highlight_labels=None, show_ids=True):
    if highlight_labels is None:
        highlight_labels = set()

    overlay = img_rgb.copy()
    labels = np.unique(mask)
    labels = labels[labels > 0]

    for lid in labels:
        region = (mask == lid)
        color = np.array([(37 * lid) % 255, (97 * lid) % 255, (173 * lid) % 255], dtype=np.uint8)
        alpha = 0.28
        overlay[region] = (
            (1 - alpha) * overlay[region].astype(np.float32) + alpha * color.astype(np.float32)
        ).astype(np.uint8)

    outlines = utils.masks_to_outlines(mask)
    overlay[outlines] = [255, 255, 0]

    for lid in highlight_labels:
        region = (mask == lid)
        overlay[region] = (
            0.5 * overlay[region].astype(np.float32) + 0.5 * np.array([255, 0, 0], dtype=np.float32)
        ).astype(np.uint8)

    if show_ids:
        for lid in labels:
            c = mask_centroid(mask, lid)
            if c is None:
                continue
            x, y = int(round(c[0])), int(round(c[1]))
            cv2.putText(overlay, str(lid), (x - 8, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


def put_help_text(img, img_name, idx, total, run_name, show_help=True):
    if not show_help:
        return
    lines = [
        f"{run_name}   [{idx}/{total}] {img_name}",
        "Commands in terminal:",
        "  m A B   -> merge A and B",
        "  d A     -> delete A",
        "  a K     -> accept suggestion K",
        "  s       -> recompute / show suggestions",
        "  u       -> undo",
        "  i       -> toggle ID text",
        "  t       -> toggle help text",
        "  n       -> save and next image",
        "  q       -> save and quit current round",
    ]
    y0 = 24
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (10, y0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)


def show_image_blocking(img_name, img_rgb, mask, idx, total, run_name,
                        highlight_labels=None, show_ids=True, show_help=True):
    canvas = build_overlay(img_rgb, mask, highlight_labels=highlight_labels, show_ids=show_ids)
    put_help_text(canvas, img_name, idx, total, run_name, show_help=show_help)
    cv2.imshow("HeLa 100x Cellpose HITL Editor", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def parse_command(s):
    parts = s.strip().split()
    if not parts:
        return None, []

    op = parts[0].lower()
    args = parts[1:]

    if op in {"u", "s", "n", "q", "i", "h", "t"}:
        return op, []

    if op == "d":
        if len(args) != 1:
            return None, []
        try:
            return op, [int(args[0])]
        except ValueError:
            return None, []

    if op == "m":
        if len(args) != 2:
            return None, []
        try:
            return op, [int(args[0]), int(args[1])]
        except ValueError:
            return None, []

    if op == "a":
        if len(args) != 1:
            return None, []
        try:
            return op, [int(args[0])]
        except ValueError:
            return None, []

    return None, []


def print_command_help():
    print("\nCommands:")
    print("  m A B   : merge cell A and B")
    print("  d A     : delete cell A")
    print("  a K     : accept suggestion K")
    print("  s       : recompute/show suggestions")
    print("  u       : undo")
    print("  i       : toggle ID text")
    print("  t       : toggle help text")
    print("  n       : save current image and go to next")
    print("  q       : save current image and quit this round")
    print("  h       : show this help\n")


def print_suggestions(suggestions):
    if not suggestions:
        print("No merge suggestions.")
        return
    print("Suggested merges:")
    for k, item in enumerate(suggestions, start=1):
        a, b = item["pair"]
        print(f"  #{k}: merge ({a}, {b})   peak_tub={item['peak']:.1f}   score={item['score']:.1f}   contact_px={item['pixels']}")


def edit_one_image(run_dir: Path, img_path: Path, idx: int, total: int):
    mask_path = get_mask_path(run_dir, img_path)
    if not mask_path.exists():
        print(f"Skip {img_path.name}: mask not found")
        return "next"

    rid = img_path.stem.replace("_merge", "")
    dna_candidates = [p for p in iter_files(RAW_DIR) if p.stem == f"{rid}_DNA"]
    tub_candidates = [p for p in iter_files(RAW_DIR) if p.stem == f"{rid}_Tubulin"]
    if not dna_candidates or not tub_candidates:
        print(f"Skip {img_path.name}: paired raw channels not found")
        return "next"

    img_rgb = read_rgb(img_path)
    dapi = normalize_dapi_100x(read_gray(dna_candidates[0]))
    tub = normalize_image(read_gray(tub_candidates[0]), gamma_val=1.0, low_pct=1, high_pct=99.5)

    mask = read_mask(mask_path)
    mask = remove_border_truncated_nuclei_by_dapi(dapi_img=dapi, mask=mask)

    undo_stack = []
    show_ids = True
    show_help = True
    highlight_labels = set()
    suggestions = find_merge_suggestions(mask, tub)

    cv2.namedWindow("HeLa 100x Cellpose HITL Editor", cv2.WINDOW_NORMAL)
    print_command_help()

    while True:
        show_image_blocking(
            img_path.name, img_rgb, mask, idx, total, run_dir.name,
            highlight_labels=highlight_labels, show_ids=show_ids, show_help=show_help
        )

        labels_now = sorted(valid_labels(mask))
        print(f"\n[{idx}/{total}] {img_path.name}")
        print(f"Existing labels ({len(labels_now)}):")
        print(labels_now[:60] if len(labels_now) > 60 else labels_now)
        print_suggestions(suggestions)

        cmd = input("Input command: ").strip()
        op, args = parse_command(cmd)

        if op is None:
            print("Invalid command. Input h for help.")
            continue

        if op == "h":
            print_command_help()
            continue

        if op == "s":
            suggestions = find_merge_suggestions(mask, tub)
            highlight_labels = set()
            continue

        if op == "i":
            show_ids = not show_ids
            continue

        if op == "t":
            show_help = not show_help
            continue

        if op == "u":
            if undo_stack:
                mask = undo_stack.pop()
                mask = remove_border_truncated_nuclei_by_dapi(dapi_img=dapi, mask=mask)
                suggestions = find_merge_suggestions(mask, tub)
                highlight_labels = set()
            else:
                print("Nothing to undo.")
            continue

        if op == "d":
            a = args[0]
            if a not in valid_labels(mask):
                print(f"Label {a} does not exist.")
                continue
            undo_stack.append(mask.copy())
            mask = delete_one_label(mask, a)
            mask = remove_border_truncated_nuclei_by_dapi(dapi_img=dapi, mask=mask)
            suggestions = find_merge_suggestions(mask, tub)
            highlight_labels = set()
            continue

        if op == "m":
            a, b = args
            current = valid_labels(mask)
            if a not in current or b not in current or a == b:
                print("Invalid labels for merge.")
                continue
            undo_stack.append(mask.copy())
            mask = merge_two_labels(mask, a, b)
            mask = remove_border_truncated_nuclei_by_dapi(dapi_img=dapi, mask=mask)
            suggestions = find_merge_suggestions(mask, tub)
            highlight_labels = {a}
            continue

        if op == "a":
            k = args[0]
            if k < 1 or k > len(suggestions):
                print("Suggestion index out of range.")
                continue
            a, b = suggestions[k - 1]["pair"]
            current = valid_labels(mask)
            if a not in current or b not in current:
                suggestions = find_merge_suggestions(mask, tub)
                print("That suggestion is no longer valid.")
                continue
            undo_stack.append(mask.copy())
            mask = merge_two_labels(mask, a, b)
            mask = remove_border_truncated_nuclei_by_dapi(dapi_img=dapi, mask=mask)
            suggestions = find_merge_suggestions(mask, tub)
            highlight_labels = {a}
            print(f"Accepted suggestion #{k}: merged ({a}, {b})")
            continue

        if op == "n":
            mask = remove_border_truncated_nuclei_by_dapi(dapi_img=dapi, mask=mask)
            save_mask(mask_path, mask)
            return "next"

        if op == "q":
            mask = remove_border_truncated_nuclei_by_dapi(dapi_img=dapi, mask=mask)
            save_mask(mask_path, mask)
            return "quit"


def edit_run_interactively(run_dir: Path):
    image_paths = sorted((run_dir / "images").glob("*_merge.png"), key=natural_key)
    total = len(image_paths)

    print(f"\nEditing masks in: {run_dir}")
    print("Mode: command line by label IDs + tubulin guidance")

    for idx, img_path in enumerate(image_paths, start=1):
        flag = edit_one_image(run_dir, img_path, idx, total)
        if flag == "quit":
            break

    cv2.destroyAllWindows()


# ============================================================
# Main loop
# ============================================================
def init_new_run():
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    new_idx = latest_run_idx(PROJECT_DIR) + 1
    run_dir = PROJECT_DIR / f"Run_{new_idx:02d}"
    make_run_dirs(run_dir)

    copy_images_to_run(run_dir)
    prev_model = get_previous_model(PROJECT_DIR)
    print(f"\n[Init] New round: {run_dir.name}")
    print(f"[Init] Using inference model: {prev_model}")

    predict_run(run_dir, prev_model)
    seed_corrected_from_predictions(run_dir)
    return run_dir


def run_one_round():
    run_dir = init_new_run()
    edit_run_interactively(run_dir)

    if yes_no(f"\nTrain {run_dir.name} now?", default=True):
        rebuild_train_folder(PROJECT_DIR, run_dir)
        train_on_run(run_dir)
        print(f"\n[Done] Training finished for {run_dir.name}")
    else:
        print(f"\n[Skip] {run_dir.name} was edited but not trained.")

    return run_dir


def main():
    if not RAW_DIR.exists():
        print(f"RAW_DIR not found: {RAW_DIR}")
        return

    pairs = find_image_pairs(RAW_DIR)
    print("========================================")
    print("HeLa 100x Cellpose HITL all-in-one")
    print("========================================")
    print(f"Raw dir: {RAW_DIR}")
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Eligible image pairs: {len(pairs)}")
    print(f"DAPI truncation margin: {BORDER_MARGIN}")
    print("")

    while True:
        run_dir = run_one_round()
        if not yes_no("\nContinue to next round?", default=True):
            print(f"\nStopped by user. Latest round: {run_dir.name}")
            break


if __name__ == "__main__":
    main()