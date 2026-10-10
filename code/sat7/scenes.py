"""Large scenes for the paper's B0-B4 campaign (I in R^{H x W x C}, eq. 1).

The paper's pipeline starts from a *large* image that must be tiled. Sources:

* ``swaths_from_manifest`` -- REAL: the 24 swaths of 4 x 4 real Airbus test tiles (3072 px,
  716 ships) defined by ``wp15_manifest.json`` from oblhdj/iastam-p7-semantic-downlink. Same
  recipe, same tiles, so a run here is directly comparable with that repository's REAL SAHI
  measurements (wp16). The tile pixels and labels are real; the seams between tiles are not a
  real sensor swath (stated in every output).
* ``airbus_mosaic`` -- REAL: k x k random real test tiles (k=2 -> 1536 px).
* ``sat7.datasets`` -- the large-scene sets the paper recommends (DOTA, HRSID / COCO).
* ``synthetic_scene`` -- SIM: a generated sea with ships of known geometry, for running the
  whole chain on a machine with no dataset. Never quoted as a measured result.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .sahi import Detection
from .synthetic import make_tile


def synthetic_scene(size: int = 2048, n_ships: int = 20, seed: int = 0, cloud_cover: float = 0.0):
    img, boxes = make_tile(n_ships=n_ships, cloud_cover=cloud_cover, size=size, seed=seed)
    return img, [Detection(b.x, b.y, b.w, b.h, 1.0, 0) for b in boxes]


def _yolo_labels(path: Path, tile: int) -> list[Detection]:
    """Tile-local ground truth, clipped to the tile (a few Airbus labels overhang by < 1 px)."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().split("\n"):
        v = line.split()
        if len(v) == 5:
            c, cx, cy, w, h = int(v[0]), *(float(z) * tile for z in v[1:])
            x0, y0 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
            x1, y1 = min(float(tile), cx + w / 2), min(float(tile), cy + h / 2)
            if x1 > x0 and y1 > y0:
                out.append(Detection(x0, y0, x1 - x0, y1 - y0, 1.0, c))
    return out


def _read_tile(path: Path, tile: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img if img.shape[:2] == (tile, tile) else cv2.resize(img, (tile, tile))


def stitch(layout: list[tuple[int, int, str]], img_dir: Path, lbl_dir: Path, tile: int = 768,
           stride: int | None = None):
    """Place tiles at (row, col) on one canvas; returns (image, gt in scene coordinates)."""
    stride = stride or tile
    n_side = max(max(r, c) for r, c, _ in layout) + 1
    side = (n_side - 1) * stride + tile
    scene = np.zeros((side, side, 3), np.uint8)
    gts = []
    for r, c, name in layout:
        x0, y0 = c * stride, r * stride
        scene[y0:y0 + tile, x0:x0 + tile] = _read_tile(img_dir / name, tile)
        gts += [d.to_global(x0, y0) for d in _yolo_labels(lbl_dir / f"{Path(name).stem}.txt", tile)]
    return scene, gts


def airbus_mosaic(files: list[Path], labels_dir: Path, k: int = 2, tile: int = 768):
    """Stitch k*k tiles (row-major) into one scene; returns (image, gt detections)."""
    if len(files) != k * k:
        raise ValueError(f"need {k * k} tiles")
    layout = [(n // k, n % k, f.name) for n, f in enumerate(files)]
    return stitch(layout, files[0].parent, labels_dir, tile)


def swaths_from_manifest(manifest: Path, data_dir: Path, split: str | None = None):
    """Yield (swath_id, image, gt) for every swath of a wp15 manifest, in manifest order."""
    man = json.loads(Path(manifest).read_text())
    split = split or man.get("split", "test")
    img_dir = Path(data_dir) / "images" / split
    lbl_dir = Path(data_dir) / "labels" / split
    tile, stride = int(man.get("tile_px", 768)), int(man.get("stride_px", man.get("tile_px", 768)))
    for sw in man["swaths"]:
        layout = [(int(r), int(c), str(name)) for r, c, name in sw["source_tiles"]]
        img, gts = stitch(layout, img_dir, lbl_dir, tile, stride)
        yield sw["swath_id"], img, gts
