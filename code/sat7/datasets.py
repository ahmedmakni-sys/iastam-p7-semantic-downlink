"""The large-scene datasets the paper recommends (section VIII-C): DOTA, HRSID, any COCO export.

Airbus tiles are 768 px and already cut, so they cannot show what SAHI does on a real wide
scene. DOTA (optical, large aerial scenes, class "ship") and HRSID (SAR ships, COCO json) can.
These loaders turn each format into the pipeline's own ground truth (``sahi.Detection``,
top-left corner) so ``scripts/paper_b0_b4.py --source dota|coco`` runs the full B0-B4 campaign
on them unchanged.

The annotation parsers are pure functions, testable without downloading tens of GB. They are
ported from oblhdj/iastam-p7-semantic-downlink (commit d69f85d, MIT), converted from centre-form
tuples to ``Detection``. The detector was trained on Airbus optical tiles: on a new domain it
needs fine-tuning first, so report its recall on these sets as what it is.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from .sahi import Detection

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _isnum(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_dota_lines(lines, classes=("ship",)) -> list[Detection]:
    """DOTA ``x1 y1 x2 y2 x3 y3 x4 y4 category difficult`` -> axis-aligned boxes.

    Header lines (``imagesource:``, ``gsd:``) and blanks are skipped; the oriented quad becomes
    its axis-aligned bounding box. ``classes=None`` keeps every category.
    """
    want = None if classes is None else {c.lower() for c in classes}
    out = []
    for ln in lines:
        p = ln.strip().split()
        if len(p) < 9 or not _isnum(p[0]):
            continue
        if want is not None and p[8].lower() not in want:
            continue
        xs = [float(p[i]) for i in (0, 2, 4, 6)]
        ys = [float(p[i]) for i in (1, 3, 5, 7)]
        out.append(Detection(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), 1.0, 0))
    return out


def parse_yolo_lines(lines, width: int, height: int, classes=None) -> list[Detection]:
    """YOLO ``class cx cy w h`` (normalised) -> pixel boxes; ``classes`` keeps those ids."""
    out = []
    for ln in lines:
        p = ln.strip().split()
        if len(p) < 5 or not _isnum(p[1]):
            continue
        cls = int(float(p[0]))
        if classes is not None and cls not in classes:
            continue
        cx, cy = float(p[1]) * width, float(p[2]) * height
        w, h = float(p[3]) * width, float(p[4]) * height
        out.append(Detection(cx - w / 2, cy - h / 2, w, h, 1.0, 0))
    return out


def coco_ground_truth(coco: dict, category_names=("ship",)) -> dict[str, list[Detection]]:
    """COCO dict -> {file_name: boxes}. Used for HRSID and any xView-style COCO export."""
    want_ids = None
    if category_names is not None:
        want = {c.lower() for c in category_names}
        want_ids = {c["id"] for c in coco.get("categories", []) if c["name"].lower() in want}
    by_id: dict[int, list[Detection]] = {}
    for a in coco.get("annotations", []):
        if want_ids is not None and a.get("category_id") not in want_ids:
            continue
        x, y, w, h = a["bbox"]
        by_id.setdefault(a["image_id"], []).append(Detection(x, y, w, h, 1.0, 0))
    return {im.get("file_name", str(im["id"])): by_id.get(im["id"], [])
            for im in coco.get("images", [])}


def iter_dota(img_dir: Path, ann_dir: Path, classes=("ship",), with_ships_only: bool = True):
    """Yield (name, image, gt) for every DOTA image that has an annotation file."""
    img_dir, ann_dir = Path(img_dir), Path(ann_dir)
    for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXT):
        ann = ann_dir / f"{img_path.stem}.txt"
        if not ann.exists():
            continue
        gts = parse_dota_lines(ann.read_text().splitlines(), classes)
        if with_ships_only and not gts:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is not None:
            yield img_path.name, img, gts


def iter_coco(coco_json: Path, img_dir: Path, category_names=("ship",),
              with_ships_only: bool = True):
    """Yield (name, image, gt) for every image of a COCO json (e.g. HRSID)."""
    gt = coco_ground_truth(json.loads(Path(coco_json).read_text()), category_names)
    for name, boxes in sorted(gt.items()):
        if with_ships_only and not boxes:
            continue
        img = cv2.imread(str(Path(img_dir) / name), cv2.IMREAD_COLOR)
        if img is not None:
            yield name, img, boxes
