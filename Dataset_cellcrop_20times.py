import re
from pathlib import Path
import csv
import cv2
import numpy as np
from PIL import Image

INPUT_DIR = Path("./Dataset_raw/20times")
OUTPUT_DIR = Path("./Dataset/Dataset_20times")
BOX_SIZE = 160
DISPLAY_GAMMA = 1.00

class MouseState:
    def __init__(self):
        self.x = None
        self.y = None
        self.clicked = False
        self.click_x = None
        self.click_y = None

def cell_distinguisher_20times():
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    history_file = input_dir / "history.tsv"
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = find_image_pairs(input_dir)
    if not pairs:
        print(f"No valid image pairs found in {input_dir}")
        return

    done_ids = load_history_ids(history_file)
    redo_processed = True
    if done_ids:
        redo_processed = confirm_yes_no(f"Reprocess handled groups in {input_dir}? [Y/N]: ")

    if not redo_processed:
        pairs = [p for p in pairs if p["id"] not in done_ids]

    if not pairs:
        print(f"No image groups to process in {input_dir}")
        return

    save_counter = next_cell_index(output_dir)

    for pair in pairs:
        dna = read_gray(pair["dna_path"])
        tub = read_gray(pair["tub_path"])

        if dna.shape != tub.shape:
            print(f"Skip {pair['raw_id']}: size mismatch")
            continue

        tub8 = normalize_image(tub, DISPLAY_GAMMA, 1, 99.5)
        dna_main = normalize_dna_20x(dna)

        merged_main = np.zeros((dna.shape[0], dna.shape[1], 3), dtype=np.uint8)
        merged_main[:, :, 0] = dna_main
        merged_main[:, :, 1] = tub8

        # 20x 主界面显示 merge
        base_img = merged_main.copy()

        window_name = f"20x group: {pair['raw_id']}"
        mouse = MouseState()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback, mouse)

        print(f"\nProcessing {pair['raw_id']}")
        print("Move mouse: preview box")
        print("Left click: select one cell")
        print("Press Enter: finish this image")

        while True:
            disp = base_img.copy()

            if mouse.x is not None and mouse.y is not None:
                draw_box(disp, mouse.x, mouse.y, BOX_SIZE, color=(255, 255, 0), thickness=1, dashed=True)
                cv2.drawMarker(
                    disp,
                    (int(mouse.x), int(mouse.y)),
                    (255, 255, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=10,
                    thickness=1,
                )

            cv2.imshow(window_name, disp)
            key = cv2.waitKey(20) & 0xFF

            if key == 13:
                break

            if mouse.clicked:
                cx = mouse.click_x
                cy = mouse.click_y
                mouse.clicked = False

                if not box_inside(dna.shape, cx, cy, BOX_SIZE):
                    print("Crop box exceeds boundary. Skipped.")
                    continue

                temp_disp = base_img.copy()
                draw_box(temp_disp, cx, cy, BOX_SIZE, color=(255, 255, 0), thickness=2, dashed=False)
                cv2.drawMarker(
                    temp_disp,
                    (int(cx), int(cy)),
                    (255, 255, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=10,
                    thickness=2,
                )
                cv2.imshow(window_name, temp_disp)

                tub_crop = crop_cell(tub, cx, cy, BOX_SIZE)
                dna_crop = crop_cell(dna, cx, cy, BOX_SIZE)

                keep_cell, phase = review_cell_20times(tub_crop, dna_crop)
                if not keep_cell:
                    continue

                prefix = f"{save_counter:04d}_{phase}"
                tub_out = output_dir / f"{prefix}_Tubulin.png"
                dna_out = output_dir / f"{prefix}_DNA.png"

                Image.fromarray(normalize_image(tub_crop, 1.0, 1, 99.5)).save(tub_out)
                Image.fromarray(normalize_dna_20x(dna_crop)).save(dna_out)

                append_history_record(
                    history_file=history_file,
                    raw_id=pair["raw_id"],
                    phase=phase,
                    cx=int(round(cx)),
                    cy=int(round(cy)),
                    box_size=int(BOX_SIZE),
                    output_prefix=prefix,
                    input_dir=str(input_dir),
                    output_dir=str(output_dir),
                )

                draw_box(base_img, cx, cy, BOX_SIZE, color=(0, 255, 255), thickness=2, dashed=False)
                cv2.drawMarker(
                    base_img,
                    (int(cx), int(cy)),
                    (0, 255, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=10,
                    thickness=2,
                )
                cv2.putText(
                    base_img,
                    str(save_counter),
                    (int(cx) - 10, int(cy) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                save_counter += 1

        mark_group_done(history_file, pair["id"])
        cv2.destroyWindow(window_name)

    cv2.destroyAllWindows()


def mouse_callback(event, x, y, flags, param):
    mouse = param
    mouse.x = x
    mouse.y = y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse.clicked = True
        mouse.click_x = x
        mouse.click_y = y


def draw_box(img, cx, cy, box_size, color=(255, 255, 0), thickness=1, dashed=False):
    half = box_size // 2
    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = x1 + box_size
    y2 = y1 + box_size

    if dashed:
        draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness)
        draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness)
        draw_dashed_line(img, (x2, y2), (x1, y2), color, thickness)
        draw_dashed_line(img, (x1, y2), (x1, y1), color, thickness)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def natural_key(s):
    parts = re.split(r"(\d+)", str(s))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def draw_dashed_line(img, p1, p2, color, thickness=1, dash_len=8):
    x1, y1 = p1
    x2, y2 = p2
    dist = int(np.hypot(x2 - x1, y2 - y1))
    if dist == 0:
        return
    for i in range(0, dist, dash_len * 2):
        a = i / dist
        b = min(i + dash_len, dist) / dist
        xa = int(round(x1 + (x2 - x1) * a))
        ya = int(round(y1 + (y2 - y1) * a))
        xb = int(round(x1 + (x2 - x1) * b))
        yb = int(round(y1 + (y2 - y1) * b))
        cv2.line(img, (xa, ya), (xb, yb), color, thickness)


def find_image_pairs(input_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in exts]

    dna_map = {}
    tub_map = {}
    raw_map = {}

    for f in files:
        stem = f.stem
        m_dna = re.match(r"^(.*)_DNA$", stem, flags=re.IGNORECASE)
        m_tub = re.match(r"^(.*)_Tubulin$", stem, flags=re.IGNORECASE)

        if m_dna:
            raw_id = m_dna.group(1)
            dna_map[raw_id] = f
            raw_map[raw_id] = raw_id
        elif m_tub:
            raw_id = m_tub.group(1)
            tub_map[raw_id] = f
            raw_map[raw_id] = raw_id

    common = sorted(set(dna_map) & set(tub_map), key=natural_key)

    return [
        {
            "id": rid,
            "raw_id": raw_map[rid],
            "dna_path": dna_map[rid],
            "tub_path": tub_map[rid],
        }
        for rid in common
    ]


def read_gray(path: Path):
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32)


def normalize_image(img, gamma_val=1.0, low_pct=1.0, high_pct=99.5):
    p1 = np.percentile(img, low_pct)
    p99 = np.percentile(img, high_pct)
    if p99 <= p1:
        p99 = p1 + 1.0
    arr = (img.astype(np.float32) - p1) / (p99 - p1)
    arr = np.clip(arr, 0.0, 1.0)
    arr = arr ** gamma_val
    return (arr * 255).astype(np.uint8)


def normalize_dna_20x(img):
    black_pct = 30
    white_pct = 99
    gamma_val = 0.9

    p_black = np.percentile(img, black_pct)
    p_white = np.percentile(img, white_pct)
    if p_white <= p_black:
        p_white = p_black + 1.0

    arr = (img.astype(np.float32) - p_black) / (p_white - p_black)
    arr = np.clip(arr, 0.0, 1.0)
    arr = arr ** gamma_val
    return (arr * 255).astype(np.uint8)


def box_inside(img_shape, cx, cy, box_size):
    h, w = img_shape
    half = box_size // 2
    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = x1 + box_size - 1
    y2 = y1 + box_size - 1
    return x1 >= 0 and y1 >= 0 and x2 < w and y2 < h


def crop_cell(img, cx, cy, box_size):
    half = box_size // 2
    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = x1 + box_size
    y2 = y1 + box_size
    return img[y1:y2, x1:x2]


def review_cell_20times(tub_crop, dna_crop):
    tub8 = normalize_image(tub_crop, 1.0, 1, 99.5)
    dna8 = normalize_dna_20x(dna_crop)

    merged = np.zeros((dna8.shape[0], dna8.shape[1], 3), dtype=np.uint8)
    merged[:, :, 0] = dna8
    merged[:, :, 1] = tub8

    tub_show = cv2.cvtColor(tub8, cv2.COLOR_GRAY2BGR)
    dna_show = cv2.cvtColor(dna8, cv2.COLOR_GRAY2BGR)

    h, w = dna8.shape
    gap = 20
    canvas = np.ones((h + 50, w * 3 + gap * 2, 3), dtype=np.uint8) * 230
    canvas[50:50 + h, 0:w] = merged
    canvas[50:50 + h, w + gap:w * 2 + gap] = tub_show
    canvas[50:50 + h, w * 2 + gap * 2:w * 3 + gap * 2] = dna_show

    cv2.putText(canvas, "Merged", (w // 2 - 40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    cv2.putText(canvas, "Tubulin", (w + gap + w // 2 - 35, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    cv2.putText(canvas, "DNA", (w * 2 + gap * 2 + w // 2 - 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)

    preview_name = "20x Candidate preview"
    cv2.namedWindow(preview_name, cv2.WINDOW_NORMAL)
    cv2.imshow(preview_name, canvas)
    cv2.waitKey(1)

    while True:
        s = input("Keep this cell? [Y/N]: ").strip().upper()
        if s in {"Y", "N"}:
            keep = s == "Y"
            break
        print("Please input Y or N.")

    phase = ""
    if keep:
        valid = {"I", "P", "M", "A", "T"}
        while True:
            phase = input("Cell-cycle phase [I/P/M/A/T]: ").strip().upper()
            if phase in valid:
                break
            print("Please input I, P, M, A, or T.")

    cv2.destroyWindow(preview_name)
    return keep, phase


def load_history_ids(history_file: Path):
    if not history_file.exists():
        return set()

    done = set()
    with open(history_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("type") == "done_group":
                done.add(row.get("raw_id", ""))
    return done


def append_history_record(history_file: Path, raw_id: str, phase: str, cx: int, cy: int,
                          box_size: int, output_prefix: str, input_dir: str, output_dir: str):
    write_header = not history_file.exists()
    with open(history_file, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if write_header:
            writer.writerow([
                "type", "raw_id", "phase", "cx", "cy", "box_size",
                "output_prefix", "input_dir", "output_dir"
            ])
        writer.writerow([
            "crop", raw_id, phase, cx, cy, box_size,
            output_prefix, input_dir, output_dir
        ])


def mark_group_done(history_file: Path, raw_id: str):
    write_header = not history_file.exists()
    with open(history_file, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if write_header:
            writer.writerow([
                "type", "raw_id", "phase", "cx", "cy", "box_size",
                "output_prefix", "input_dir", "output_dir"
            ])
        writer.writerow(["done_group", raw_id, "", "", "", "", "", "", ""])


def confirm_yes_no(prompt_text):
    while True:
        s = input(prompt_text).strip().upper()
        if s in {"Y", "N"}:
            return s == "Y"
        print("Please input Y or N.")


def next_cell_index(output_dir: Path):
    files = list(output_dir.glob("*_Tubulin.png")) + list(output_dir.glob("*_DNA.png"))
    nums = []
    for f in files:
        m = re.match(r"^(\d+)_", f.name)
        if m:
            nums.append(int(m.group(1)))
    return 1 if not nums else max(nums) + 1


if __name__ == "__main__":
    cell_distinguisher_20times()