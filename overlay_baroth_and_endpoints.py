"""
Merge BarothJSONData labels + only jet_start/jet_end from labeled folder.

For each frame in BarothJSONData:
  - Keep all shapes EXCEPT jet_start and jet_end
  - Replace jet_start and jet_end with the ones from labeled/ (if available)
  - If no matching labeled JSON, keep Baroth's original jet_start/jet_end

Output: data/barrothandendpoints/{train,val,test}/ — LabelMe-compatible JSON + PNG pairs
"""

import json
import shutil
from pathlib import Path

BASE = Path(r"c:\Users\nmana\Documents\VSCode\OpenCV MEW Jet Analysis\data")
BAROTH_DIR  = BASE / "BarothJSONData"
LABELED_DIR = BASE / "data" / "labeled_og"
OUT_DIR     = BASE / "barrothandendpoints"

ENDPOINT_LABELS = {"jet_start", "jet_end"}


def merge_json(baroth_data: dict, labeled_data: dict | None) -> dict:
    baroth_shapes = baroth_data.get("shapes", [])

    if labeled_data is None:
        # No labeled counterpart — keep Baroth as-is
        merged_shapes = baroth_shapes
    else:
        # Remove jet_start / jet_end from Baroth, then append from labeled
        non_endpoint = [s for s in baroth_shapes if s["label"] not in ENDPOINT_LABELS]
        labeled_endpoints = [s for s in labeled_data.get("shapes", [])
                             if s["label"] in ENDPOINT_LABELS]
        merged_shapes = non_endpoint + labeled_endpoints

    return {
        "version":     baroth_data.get("version", "5.5.0"),
        "flags":       baroth_data.get("flags", {}),
        "shapes":      merged_shapes,
        "imagePath":   baroth_data["imagePath"],
        "imageData":   None,
        "imageHeight": baroth_data["imageHeight"],
        "imageWidth":  baroth_data["imageWidth"],
    }


def process_split(split: str) -> None:
    baroth_split  = BAROTH_DIR  / split
    labeled_split = LABELED_DIR / split
    out_split     = OUT_DIR     / split
    out_split.mkdir(parents=True, exist_ok=True)

    json_files = sorted(baroth_split.glob("*.json"))
    print(f"[{split}] {len(json_files)} frames")

    for json_path in json_files:
        stem = json_path.stem  # e.g. "frame_000059"

        img_src = baroth_split / f"{stem}.png"
        if not img_src.exists():
            print(f"  SKIP (no image): {stem}.png")
            continue

        with open(json_path) as f:
            baroth_data = json.load(f)

        labeled_json = labeled_split / f"{stem}.json"
        labeled_data = None
        if labeled_json.exists():
            with open(labeled_json) as f:
                labeled_data = json.load(f)

        merged = merge_json(baroth_data, labeled_data)

        # Write merged JSON
        out_json = out_split / f"{stem}.json"
        with open(out_json, "w") as f:
            json.dump(merged, f, indent=2)

        # Copy image
        shutil.copy2(img_src, out_split / f"{stem}.png")

    print(f"  -> {out_split}")


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        process_split(split)
    print("Done.")
