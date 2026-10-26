"""
jet_multimask — rasterize LabelMe jet annotations into a 15-channel mask stack.

Each labeled geometry type becomes its own binary channel, so the U-Net can
predict every annotation simultaneously (one sigmoid mask per channel):

    idx  label             geometry    drawn as
    ---  ----------------  ----------  --------------------------------
     0   jet_skeleton      linestrip   thick polyline
     1   jet_left_edge     linestrip   thick polyline
     2   jet_right_edge    linestrip   thick polyline
     3   jet_start         point       filled disk
     4   jet_end           point       filled disk
     5   diameter_10pct    line        thick segment
     6   diameter_20pct    line        thick segment
     7   diameter_30pct    line        thick segment
     8   diameter_40pct    line        thick segment
     9   diameter_50pct    line        thick segment
    10   diameter_60pct    line        thick segment
    11   diameter_70pct    line        thick segment
    12   diameter_80pct    line        thick segment
    13   diameter_90pct    line        thick segment
    14   diameter_100pct   line        thick segment

Masks are built at the JSON's native resolution; the caller resizes them
alongside the image. Keeping line/point strokes a few px wide ensures thin
features survive the ~3x horizontal downsample when resizing to 416 wide.
"""

import json
import numpy as np
import cv2

# Channel order — index in this list == channel index in the mask stack.
CHANNELS = [
    "jet_skeleton",
    "jet_left_edge",
    "jet_right_edge",
    "jet_start",
    "jet_end",
    "diameter_10pct",
    "diameter_20pct",
    "diameter_30pct",
    "diameter_40pct",
    "diameter_50pct",
    "diameter_60pct",
    "diameter_70pct",
    "diameter_80pct",
    "diameter_90pct",
    "diameter_100pct",
]
NUM_CLASSES = len(CHANNELS)
CHANNEL_INDEX = {name: i for i, name in enumerate(CHANNELS)}

# Stroke sizes (in native-resolution px) per geometry type.
LINESTRIP_THICKNESS = 5   # skeleton + edges
DIAMETER_THICKNESS  = 3   # diameter segments
POINT_RADIUS        = 5   # jet_start / jet_end disks


def build_masks(data, point_radius=POINT_RADIUS,
                linestrip_thickness=LINESTRIP_THICKNESS,
                diameter_thickness=DIAMETER_THICKNESS):
    """
    Build a (NUM_CLASSES, H, W) uint8 mask stack (0/255) from a LabelMe dict.

    Channels with no matching shape in the JSON stay all-zero.
    """
    h = data["imageHeight"]
    w = data["imageWidth"]
    masks = np.zeros((NUM_CLASSES, h, w), dtype=np.uint8)

    for shape in data.get("shapes", []):
        label = shape.get("label")
        ch = CHANNEL_INDEX.get(label)
        if ch is None:
            continue

        pts = np.array(shape["points"], dtype=np.int32)
        if len(pts) == 0:
            continue

        m = masks[ch]

        if label in ("jet_start", "jet_end"):
            cx, cy = int(pts[0][0]), int(pts[0][1])
            cv2.circle(m, (cx, cy), point_radius, 255, thickness=-1, lineType=cv2.LINE_AA)

        elif label.startswith("diameter_"):
            if len(pts) >= 2:
                cv2.line(m, tuple(pts[0]), tuple(pts[1]), 255,
                         thickness=diameter_thickness, lineType=cv2.LINE_AA)

        else:  # jet_skeleton / jet_left_edge / jet_right_edge
            if len(pts) >= 2:
                cv2.polylines(m, [pts], isClosed=False, color=255,
                              thickness=linestrip_thickness, lineType=cv2.LINE_AA)

    return masks


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # Quick self-test: rasterize one frame and report per-channel pixel counts.
    import sys
    from pathlib import Path

    sample = Path(r"c:\Users\nmana\Documents\VSCode\OpenCV MEW Jet Analysis"
                  r"\data\barrothandendpoints\train")
    jsons = sorted(sample.glob("*.json"))
    if not jsons:
        print("No JSONs found for self-test.")
        sys.exit(0)

    data = load_json(jsons[0])
    masks = build_masks(data)
    print(f"Self-test on {jsons[0].name}  ({data['imageWidth']}x{data['imageHeight']})")
    print(f"Mask stack shape: {masks.shape}  (expect ({NUM_CLASSES}, H, W))")
    for i, name in enumerate(CHANNELS):
        px = int((masks[i] > 0).sum())
        flag = "" if px else "   <- empty in this frame"
        print(f"  ch{i:2d} {name:16s}: {px:6d} px{flag}")
