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

RAW_DIR = BASE_DIR / "Dataset_raw" / "Feulgen_40times"
PROJECT_DIR = BASE_DIR / "Cellpose" / "Feulgen_40times"

BASE_MODEL = "cpsam"

DIAMETER = 80
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
MIN_SIZE = 60

# 只判断“紫色核区域”是否碰边
BORDER_MARGIN = 2
PURPLE_PERCENTILE = 75
MIN_PURPLE_PIXELS = 20

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


def is_holdout_image(path_or_name):
    """
    文件名主体恰好为 v1, v2, v3, ... 时，视为独立检验集。
    例如：
        v1.tif   -> True
        v12.png  -> True
        sample1.tif -> False
        v1_extra.tif -> False
    """
    stem = path_or_name.stem if isinstance(path_or_name, Path) else Path(path_or_name).stem
    return re.fullmatch(r"v\d+", stem, flags=re.IGNORECASE) is not None


def iter_images(input_dir: Path, exclude_holdout=True):
    files = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ]
    if exclude_holdout:
        files = [p for p in files if not is_holdout_image(p)]
    return sorted(files, key=natural_key)


def read_rgb(path: Path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def read_mask(path: Path):
    arr = np.array(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D label image: {path}")
    return arr.astype(np.int32)


def save_mask(path: Path, mask: np.ndarray):
    Image.fromarray(mask.astype(np.uint16)).save(path)


def relabel_sequential(mask):
    out = np.zeros_like(mask, dtype=np.int32)
    labels = np.unique(mask)
    labels = labels[labels > 0]
    for new_id, old_id in enumerate(labels, start=1):
        out[mask == old_id] = new_id
    return out


def latest_run_idx(project_dir: Path):
    if not project_dir.exists():
        return 0
    nums = []
    for p in project_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"^Run_(\d+)$", p.name)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) if nums else 0


def make_run_dirs(run_dir: Path):
    for name in ["images", "predictions", "corrected", "train", "models"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def yes_no(prompt, default=True):
    suffix = "[Y/n]" if default else "[y/N]"
    s = input(f"{prompt} {suffix}: ").strip().lower()
    if s == "":
        return default
    return s in {"y", "yes"}


def get_previous_model(project_dir: Path):
    last_idx = latest_run_idx(project_dir)
    if last_idx == 0:
        return BASE_MODEL

    prev_models = project_dir / f"Run_{last_idx:02d}" / "models"
    if not prev_models.exists():
        return BASE_MODEL

    candidates = sorted(prev_models.iterdir(), key=natural_key)
    return str(candidates[-1]) if candidates else BASE_MODEL


def copy_images_to_run(run_dir: Path):
    dst_dir = run_dir / "images"
    selected = iter_images(RAW_DIR, exclude_holdout=True)

    if not selected:
        print(f"No eligible training images found in {RAW_DIR}")
        return

    skipped = [
        p.name for p in iter_images(RAW_DIR, exclude_holdout=False)
        if is_holdout_image(p)
    ]
    if skipped:
        print(f"[Info] Holdout images skipped: {skipped}")

    for img_path in selected:
        dst = dst_dir / img_path.name
        if not dst.exists():
            shutil.copy2(img_path, dst)


def get_mask_path(run_dir: Path, img_path: Path):
    return run_dir / "corrected" / f"{img_path.stem}_masks.tif"


def valid_labels(mask):
    labels = np.unique(mask)
    return set(int(x) for x in labels if x > 0)


# ============================================================
# Purple nucleus border truncation filter
# ============================================================
def remove_border_truncated_nuclei_by_purple(
    img_rgb,
    mask,
    margin=BORDER_MARGIN,
    purple_percentile=PURPLE_PERCENTILE,
    min_purple_pixels=MIN_PURPLE_PIXELS,
):
    """
    仅删除“紫色核区域被边界截断”的实例，而不是简单删除碰边实例。
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("img_rgb must be HxWx3")
    if img_rgb.shape[:2] != mask.shape:
        raise ValueError("img_rgb and mask must have the same spatial shape")

    h, w = mask.shape
    out = mask.copy()

    rgb = img_rgb.astype(np.float32)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # 紫色分数：红蓝高、绿色低
    purple_score = 0.5 * (r + b) - g

    labels = np.unique(mask)
    labels = labels[labels > 0]

    margin = max(int(margin), 1)

    for lid in labels:
        region = (mask == lid)
        scores = purple_score[region]
        if scores.size == 0:
            continue

        thr = np.percentile(scores, purple_percentile)
        purple_region = region & (purple_score >= thr)

        if purple_region.sum() < min_purple_pixels:
            continue

        touches_border = (
            purple_region[:margin, :].any()
            or purple_region[h - margin:, :].any()
            or purple_region[:, :margin].any()
            or purple_region[:, w - margin:].any()
        )

        if touches_border:
            out[out == lid] = 0

    return relabel_sequential(out)


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


def predict_run(run_dir: Path, model_path):
    model = make_model(model_path)
    img_dir = run_dir / "images"
    pred_dir = run_dir / "predictions"

    for img_path in iter_images(img_dir, exclude_holdout=False):
        if is_holdout_image(img_path):
            continue

        print(f"[Predict] {img_path.name}")
        img = read_rgb(img_path)
        masks, flows, styles = model.eval(
            img,
            channels=[0, 0],
            diameter=DIAMETER,
            flow_threshold=FLOW_THRESHOLD,
            cellprob_threshold=CELLPROB_THRESHOLD,
            min_size=MIN_SIZE,
        )

        masks = remove_border_truncated_nuclei_by_purple(
            img_rgb=img,
            mask=masks,
            margin=BORDER_MARGIN,
            purple_percentile=PURPLE_PERCENTILE,
            min_purple_pixels=MIN_PURPLE_PIXELS,
        )
        save_mask(pred_dir / f"{img_path.stem}_masks.tif", masks)


def seed_corrected_from_predictions(run_dir: Path):
    pred_dir = run_dir / "predictions"
    corr_dir = run_dir / "corrected"

    for mask_path in sorted(pred_dir.glob("*_masks.tif"), key=natural_key):
        stem0 = mask_path.stem.replace("_masks", "")
        if is_holdout_image(stem0):
            continue
        dst = corr_dir / mask_path.name
        if not dst.exists():
            shutil.copy2(mask_path, dst)


# ============================================================
# Training set assembly
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

        for img_path in iter_images(img_dir, exclude_holdout=False):
            if is_holdout_image(img_path):
                continue

            mask_path = corr_dir / f"{img_path.stem}_masks.tif"
            if not mask_path.exists():
                continue

            img_rgb = read_rgb(img_path)
            mask = read_mask(mask_path)
            mask = remove_border_truncated_nuclei_by_purple(
                img_rgb=img_rgb,
                mask=mask,
                margin=BORDER_MARGIN,
                purple_percentile=PURPLE_PERCENTILE,
                min_purple_pixels=MIN_PURPLE_PIXELS,
            )

            shutil.copy2(img_path, train_dir / img_path.name)
            save_mask(train_dir / mask_path.name, mask)

def train_on_run(run_dir: Path):
    train_dir = run_dir / "train"
    model_name = f"{run_dir.name}_feulgen"

    cmd = [
        sys.executable, "-m", "cellpose",
        "--train",
        "--dir", str(train_dir.resolve()),
        "--mask_filter", "_masks",   # 关键修正：不要写成 _masks.tif
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
# Mask editing by ID commands
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
        color = np.array(
            [(37 * lid) % 255, (97 * lid) % 255, (173 * lid) % 255],
            dtype=np.uint8
        )
        alpha = 0.28
        overlay[region] = (
            (1 - alpha) * overlay[region].astype(np.float32) +
            alpha * color.astype(np.float32)
        ).astype(np.uint8)

    outlines = utils.masks_to_outlines(mask)
    overlay[outlines] = [255, 255, 0]

    for lid in highlight_labels:
        region = (mask == lid)
        overlay[region] = (
            0.5 * overlay[region].astype(np.float32) +
            0.5 * np.array([255, 0, 0], dtype=np.float32)
        ).astype(np.uint8)

    if show_ids:
        for lid in labels:
            c = mask_centroid(mask, lid)
            if c is None:
                continue
            x, y = int(round(c[0])), int(round(c[1]))
            cv2.putText(
                overlay,
                str(lid),
                (x - 8, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return overlay


def put_help_text(img, img_name, idx, total, run_name, show_help=True):
    if not show_help:
        return

    lines = [
        f"{run_name}   [{idx}/{total}] {img_name}",
        "Commands in terminal:",
        "  m A B   -> merge A and B",
        "  d A     -> delete A",
        "  u       -> undo",
        "  r       -> refresh display",
        "  i       -> toggle ID text",
        "  t       -> toggle help text",
        "  n       -> save and next image",
        "  q       -> save and quit current round",
    ]
    y0 = 24
    for i, txt in enumerate(lines):
        cv2.putText(
            img,
            txt,
            (10, y0 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


def show_image_blocking(window_name, img_rgb, mask, idx, total, run_name,
                        highlight_labels=None, show_ids=True, show_help=True):
    canvas = build_overlay(
        img_rgb,
        mask,
        highlight_labels=highlight_labels,
        show_ids=show_ids,
    )
    put_help_text(canvas, window_name, idx, total, run_name, show_help=show_help)
    cv2.imshow("Feulgen Cellpose HITL Editor", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def parse_command(s):
    parts = s.strip().split()
    if not parts:
        return None, []

    op = parts[0].lower()
    args = parts[1:]

    if op in {"u", "r", "n", "q", "i", "h", "t"}:
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

    return None, []


def print_command_help():
    print("\nCommands:")
    print("  m A B   : merge cell A and B")
    print("  d A     : delete cell A")
    print("  u       : undo last operation")
    print("  r       : refresh image")
    print("  i       : toggle ID text on image")
    print("  t       : toggle help text on image")
    print("  n       : save current image and go to next")
    print("  q       : save current image and quit this round")
    print("  h       : show this help\n")


def edit_one_image(run_dir: Path, img_path: Path, idx: int, total: int):
    mask_path = get_mask_path(run_dir, img_path)
    if not mask_path.exists():
        print(f"Skip {img_path.name}: mask not found -> {mask_path}")
        return "next"

    img_rgb = read_rgb(img_path)
    mask = read_mask(mask_path)
    mask = remove_border_truncated_nuclei_by_purple(
        img_rgb=img_rgb,
        mask=mask,
        margin=BORDER_MARGIN,
        purple_percentile=PURPLE_PERCENTILE,
        min_purple_pixels=MIN_PURPLE_PIXELS,
    )

    undo_stack = []
    show_ids = True
    show_help = True
    highlight_labels = set()

    cv2.namedWindow("Feulgen Cellpose HITL Editor", cv2.WINDOW_NORMAL)
    print_command_help()

    while True:
        show_image_blocking(
            img_path.name,
            img_rgb,
            mask,
            idx,
            total,
            run_dir.name,
            highlight_labels=highlight_labels,
            show_ids=show_ids,
            show_help=show_help,
        )

        labels_now = sorted(valid_labels(mask))
        print(f"\n[{idx}/{total}] {img_path.name}")
        print(f"Existing labels ({len(labels_now)}):")
        if len(labels_now) <= 60:
            print(labels_now)
        else:
            print(labels_now[:60], "...")

        cmd = input("Input command: ").strip()
        op, args = parse_command(cmd)

        if op is None:
            print("Invalid command. Input h for help.")
            continue

        if op == "h":
            print_command_help()
            continue

        if op == "r":
            highlight_labels = set()
            continue

        if op == "i":
            show_ids = not show_ids
            print(f"show_ids = {show_ids}")
            continue

        if op == "t":
            show_help = not show_help
            print(f"show_help = {show_help}")
            continue

        if op == "u":
            if undo_stack:
                mask = undo_stack.pop()
                mask = remove_border_truncated_nuclei_by_purple(
                    img_rgb=img_rgb,
                    mask=mask,
                    margin=BORDER_MARGIN,
                    purple_percentile=PURPLE_PERCENTILE,
                    min_purple_pixels=MIN_PURPLE_PIXELS,
                )
                highlight_labels = set()
                print("Undo done.")
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
            mask = remove_border_truncated_nuclei_by_purple(
                img_rgb=img_rgb,
                mask=mask,
                margin=BORDER_MARGIN,
                purple_percentile=PURPLE_PERCENTILE,
                min_purple_pixels=MIN_PURPLE_PIXELS,
            )
            highlight_labels = set()
            print(f"Deleted {a}.")
            continue

        if op == "m":
            a, b = args
            current = valid_labels(mask)
            if a not in current or b not in current:
                print(f"Label not found. Existing labels include: {sorted(list(current))[:20]} ...")
                continue
            if a == b:
                print("Cannot merge the same label.")
                continue
            undo_stack.append(mask.copy())
            mask = merge_two_labels(mask, a, b)
            mask = remove_border_truncated_nuclei_by_purple(
                img_rgb=img_rgb,
                mask=mask,
                margin=BORDER_MARGIN,
                purple_percentile=PURPLE_PERCENTILE,
                min_purple_pixels=MIN_PURPLE_PIXELS,
            )
            highlight_labels = set()
            print(f"Merged {a} and {b}.")
            continue

        if op == "n":
            mask = remove_border_truncated_nuclei_by_purple(
                img_rgb=img_rgb,
                mask=mask,
                margin=BORDER_MARGIN,
                purple_percentile=PURPLE_PERCENTILE,
                min_purple_pixels=MIN_PURPLE_PIXELS,
            )
            save_mask(mask_path, mask)
            print("Saved. Next image.")
            return "next"

        if op == "q":
            mask = remove_border_truncated_nuclei_by_purple(
                img_rgb=img_rgb,
                mask=mask,
                margin=BORDER_MARGIN,
                purple_percentile=PURPLE_PERCENTILE,
                min_purple_pixels=MIN_PURPLE_PIXELS,
            )
            save_mask(mask_path, mask)
            print("Saved. Quit current round editor.")
            return "quit"


def edit_run_interactively(run_dir: Path):
    image_paths = iter_images(run_dir / "images", exclude_holdout=False)
    image_paths = [p for p in image_paths if not is_holdout_image(p)]
    total = len(image_paths)

    print(f"\nEditing masks in: {run_dir}")
    print("Mode: command line by label IDs")
    print("You watch the image window, then type commands in terminal.")

    for idx, img_path in enumerate(image_paths, start=1):
        flag = edit_one_image(run_dir, img_path, idx, total)
        if flag == "quit":
            break

    cv2.destroyAllWindows()


# ============================================================
# One round
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


# ============================================================
# Main loop
# ============================================================
def main():
    if not RAW_DIR.exists():
        print(f"RAW_DIR not found: {RAW_DIR}")
        return

    all_imgs = iter_images(RAW_DIR, exclude_holdout=False)
    train_imgs = [p.name for p in all_imgs if not is_holdout_image(p)]
    holdout_imgs = [p.name for p in all_imgs if is_holdout_image(p)]

    print("========================================")
    print("Feulgen Cellpose HITL all-in-one")
    print("========================================")
    print(f"Raw images: {RAW_DIR}")
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Eligible training images: {len(train_imgs)}")
    print(f"Holdout images skipped: {len(holdout_imgs)}")
    print(f"BORDER_MARGIN = {BORDER_MARGIN}")
    print(f"PURPLE_PERCENTILE = {PURPLE_PERCENTILE}")
    print(f"MIN_PURPLE_PIXELS = {MIN_PURPLE_PIXELS}")
    if holdout_imgs:
        print("Holdout list:", holdout_imgs)
    print("")

    while True:
        run_dir = run_one_round()

        if not yes_no("\nContinue to next round?", default=True):
            print("\nStopped by user.")
            print(f"Latest round: {run_dir.name}")
            break


if __name__ == "__main__":
    main()