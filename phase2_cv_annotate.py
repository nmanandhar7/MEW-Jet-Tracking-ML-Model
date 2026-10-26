import sys
import os
import argparse
import pickle
import lzma
import csv
import copy
import time

import cv2 as cv
import numpy as np

BAROTH_PYTHON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "MEW_Computer_Vision_PublicationArchive-main",
    "MEW_Computer_Vision_PublicationArchive-main",
    "Python",
)
sys.path.insert(0, BAROTH_PYTHON)

import CV_functions as cvt
import image_annotation as cva

DIA_PCT = [10, 25, 50, 75, 99]


def list_frames(source_dir):
    frames = []
    for name in sorted(os.listdir(source_dir)):
        if not (name.endswith(".png") or name.endswith(".jpg")):
            continue
        try:
            parts = name.split("_")
            num = int(parts[1].split(".")[0]) if len(parts) > 1 else int(name.split(".")[0])
            frames.append((num, name))
        except (IndexError, ValueError):
            pass
    return sorted(frames)


def estimate_init_from_annotations(annotations_csv, img_h, img_w):
    """Derive CV init parameters from existing annotations.csv when CV_init.pkl is incompatible."""
    x1_vals, y1_vals, y2_vals = [], [], []
    with open(annotations_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                x1_vals.append(float(row["x1"]))
                y1_vals.append(float(row["y1"]))
                y2_vals.append(float(row["y2"]))
            except (KeyError, ValueError):
                pass
    if not x1_vals:
        return None

    nozzle_x = int(np.median(x1_vals))
    nozzle_y = int(np.median(y1_vals))
    hw = 15  # half-width estimate for nozzle exit (pixels) — adjust if CV fails
    upper = max(0, nozzle_y - 5)
    lower = min(img_h - 5, int(np.percentile(y2_vals, 95)) + 5)

    return {
        "blur_kernel": 3,
        "thres_kernel": 31,
        "collector_left": img_h,
        "collector_right": img_h,
        "nozzle_mid_point": [[nozzle_x], [nozzle_y]],
        "left_nozzle_line_points": [[nozzle_x - hw, nozzle_x - hw], [upper, nozzle_y]],
        "right_nozzle_line_points": [[nozzle_x + hw, nozzle_x + hw], [upper, nozzle_y]],
        "upper_cut_off": upper,
        "lower_cut_off": lower,
    }


ROI_X_PAD = 150  # pixels each side of nozzle x to crop before CV processing


def _crop_init(init_dict, roi_x1):
    """Return a copy of init_dict with all x-coordinates shifted left by roi_x1."""
    d = copy.deepcopy(init_dict)
    d["nozzle_mid_point"][0][0] -= roi_x1
    for key in ("left_nozzle_line_points", "right_nozzle_line_points"):
        d[key][0] = [x - roi_x1 for x in d[key][0]]
    return d


def _shift_contour_x(contour, dx):
    """Add dx to the x-channel of a (y_array, x_array) contour tuple."""
    if not isinstance(contour, tuple):
        return contour
    ys, xs = contour
    return (ys, np.asarray(xs) + dx)


def _rotate_contour(contour, M):
    """Apply 2×3 affine matrix M to a (y_array, x_array) contour tuple."""
    ys = np.asarray(contour[0], dtype=float)
    xs = np.asarray(contour[1], dtype=float)
    pts = np.stack([xs, ys, np.ones(len(xs))])  # 3×N
    rot = M @ pts                                # 2×N
    return (rot[1], rot[0])                      # back to (ys, xs)


def _build_angle_lookup(annotations_csv, cv_metrics_csv):
    """Return {frame_name: tilt_right_deg} from both CSV sources.

    Sign convention: positive = jet tilts right (+x direction).
    annotations.csv uses this convention directly.
    cv_metrics.csv uses the opposite sign (Baroth atan2 convention), so we negate.
    annotations.csv takes priority when both have the same frame.
    """
    lookup = {}
    if os.path.exists(cv_metrics_csv):
        with open(cv_metrics_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] == "ok":
                    try:
                        lookup[row["frame"]] = -float(row["angle_deg"])
                    except ValueError:
                        pass
    if os.path.exists(annotations_csv):
        with open(annotations_csv, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    lookup[row["filename"]] = float(row["angle_deg"])
                except (KeyError, ValueError):
                    pass
    return lookup


ROTATION_THRESHOLD_DEG = 10.0  # only rotate when |tilt| exceeds this


def process_frame(img, init_dict, tilt_right_deg=0.0):
    """Run full CV pipeline on one grayscale frame. Returns (shape, status, measurements).

    For steep-angle jets (|tilt_right_deg| > ROTATION_THRESHOLD_DEG), rotates the ROI
    so the jet is vertical before running Baroth, then rotates all contours back.
    """
    # --- Crop image to narrow ROI around nozzle ---
    nozzle_x = init_dict["nozzle_mid_point"][0][0]
    roi_x1 = max(0, nozzle_x - ROI_X_PAD)
    roi_x2 = min(img.shape[1], nozzle_x + ROI_X_PAD)
    img_roi = img[:, roi_x1:roi_x2]
    init_roi = _crop_init(init_dict, roi_x1)

    # --- Optional rotation to align steep-angle jet with vertical axis ---
    do_rotate = abs(tilt_right_deg) > ROTATION_THRESHOLD_DEG
    if do_rotate:
        cx = float(init_roi["nozzle_mid_point"][0][0])
        cy = float(init_roi["nozzle_mid_point"][1][0])
        # Rotate image CW by tilt_right_deg to stand the jet upright.
        # OpenCV positive angle = CCW, so we pass -tilt_right_deg for CW.
        M_fwd = cv.getRotationMatrix2D((cx, cy), -tilt_right_deg, 1.0)
        M_inv = cv.getRotationMatrix2D((cx, cy),  tilt_right_deg, 1.0)
        img_roi = cv.warpAffine(img_roi, M_fwd,
                                (img_roi.shape[1], img_roi.shape[0]),
                                flags=cv.INTER_LINEAR,
                                borderMode=cv.BORDER_REPLICATE)
        # Collector band rotates out of frame at steep angles — extend lower cutoff
        init_roi = copy.deepcopy(init_roi)
        init_roi["lower_cut_off"] = img_roi.shape[0] - 2

    # Two-stage erosion before calling Baroth:
    # 1) (3,1) horizontal: widens the thin jet from 1-2px to ~4px so it survives
    #    Baroth's internal 3x3 MORPH_OPEN (which removes structures <3px wide).
    # 2) (1,7) vertical: bridges gaps up to 3 rows in the jet body WITHOUT
    #    spreading the substrate/collector dark band far enough upward to trigger
    #    Baroth's "|delta count| > 30" early-stop rule.  (1,13) spread the
    #    collector 6 rows upward, causing stops 5-9 rows above the actual jet tip.
    img_roi = cv.erode(img_roi, cv.getStructuringElement(cv.MORPH_RECT, (3, 1)))
    img_roi = cv.erode(img_roi, cv.getStructuringElement(cv.MORPH_RECT, (1, 7)))

    (
        _thres, contour_status,
        left_s, right_s,
        left_c, right_c,
        _, _, _, _,
        _all_contours, _,
        _y_min_l, _y_min_r,
    ) = cvt.getJetContoursHorizontalLineScan(
        img_roi,
        collector_left=init_roi["collector_left"],
        collector_right=init_roi["collector_right"],
        nozzleleftlowy=init_roi["left_nozzle_line_points"][1][1],
        nozzlerightlowy=init_roi["right_nozzle_line_points"][1][1],
        nozzleleftlowx=init_roi["left_nozzle_line_points"][0][1],
        nozzlerightlowx=init_roi["right_nozzle_line_points"][0][1],
        Threshold_kernel=init_roi["thres_kernel"],
        Blur_kernel=init_roi["blur_kernel"],
        upper_cut_off=init_roi["upper_cut_off"],
        lower_cut_off=init_roi["lower_cut_off"],
    )
    if contour_status != "Ok":
        return None, contour_status, {}

    # --- Un-rotate contours back to original ROI orientation ---
    # warpAffine uses M_fwd as a destination→source lookup, so features at
    # source position p appear in destination at M_fwd(p). Applying M_fwd to
    # the destination contour coordinates recovers the original source positions.
    if do_rotate:
        left_c  = _rotate_contour(left_c,  M_fwd)
        right_c = _rotate_contour(right_c, M_fwd)
        left_s  = _rotate_contour(left_s,  M_fwd)
        right_s = _rotate_contour(right_s, M_fwd)

    # --- Translate contour/midline x-coords back to full-frame space ---
    left_c  = _shift_contour_x(left_c,  roi_x1)
    right_c = _shift_contour_x(right_c, roi_x1)
    left_s  = _shift_contour_x(left_s,  roi_x1)
    right_s = _shift_contour_x(right_s, roi_x1)

    midline, midline_s, midline_status = cvt.getMidlinefromContours(left_s, right_s)
    if midline_status != "Ok":
        return None, midline_status, {}

    shape = {
        "left_contour": left_c,
        "right_contour": right_c,
        "left_contour_smoothed": left_s,
        "right_contour_smoothed": right_s,
        "midline_coordinates": midline,
        "midline_coordinates_smoothed": midline_s,
    }

    # Arc length along midline in pixels
    all_lengths, _ = cvt.getLengthAlongMidline(midline, pixel_size=1.0)
    total_length = float(all_lengths[-1])

    # Midline indices at requested % positions
    dia_indices = []
    for pct in DIA_PCT:
        target = (pct / 100.0) * total_length
        hit = np.where(all_lengths >= target - 0.05)[0]
        dia_indices.append(int(hit[0]) + 1 if len(hit) > 0 else -1)

    diameters, _, contour_dia_idx, _ = cvt.getFibreDiameterIntersect(
        init_dict["nozzle_mid_point"], midline, left_c, right_c,
        dia_indices, pixel_size=1.0, approx_dist=5,
    )

    # Jet angle at the tip (last midline point)
    tip_idx = len(midline[0]) - 1
    jet_angles, _ = cvt.getJetAngle(init_dict["nozzle_mid_point"], midline, [tip_idx])

    measurements = {
        "angle_deg": float(jet_angles[0]),
        "total_length_px": total_length,
        "diameters_px": [float(d) for d in diameters],
        "contour_dia_indices": contour_dia_idx,
        "dia_midline_indices": dia_indices,
        "tip_midline_idx": tip_idx,
    }
    return shape, "Ok", measurements


def annotate_frame(img, shape, measurements, init_dict):
    """Draw CV overlays onto a BGR copy of the frame."""
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cimg = cva.annotateEdes(cimg, shape)
    cimg = cva.annotateMidline(cimg, shape)
    cva.drawDiameterLines(
        cimg,
        measurements["diameters_px"],
        shape["left_contour"],
        shape["right_contour"],
        measurements["contour_dia_indices"],
        text=False,
    )
    tip_idx = measurements["tip_midline_idx"]
    if tip_idx != -1:
        cva.drawAngleLines(
            cimg,
            [measurements["angle_deg"]],
            [tip_idx],
            init_dict["nozzle_mid_point"],
            shape["midline_coordinates"],
            text=False,
        )
    return cimg


def make_mask(img_shape, shape):
    """
    Single-channel uint8 mask:
      0 = background
      1 = left jet edge
      2 = right jet edge
      3 = centerline
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pairs = (
        (1, shape["left_contour"]),
        (2, shape["right_contour"]),
        (3, shape["midline_coordinates"]),
    )
    for label, contour in pairs:
        ys = np.asarray(contour[0], dtype=int)
        xs = np.asarray(contour[1], dtype=int)
        valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        mask[ys[valid], xs[valid]] = label
    return mask


def main():
    ap = argparse.ArgumentParser(description="Run Baroth CV pipeline to generate ML training annotations")
    ap.add_argument("--source", default="data/filtered_frames", help="Input frames directory")
    ap.add_argument("--output", default="data", help="Output base directory")
    ap.add_argument("--init", default="CV_init.pkl", help="CV calibration pickle file")
    ap.add_argument("--show", action="store_true", help="Display each annotated frame (debug)")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N frames (0 = all)")
    args = ap.parse_args()

    out_annotated = os.path.join(args.output, "annotated")
    out_shapes = os.path.join(args.output, "cv_shapes")
    out_masks = os.path.join(args.output, "masks")
    for d in (out_annotated, out_shapes, out_masks):
        os.makedirs(d, exist_ok=True)

    # --- Load CV init ---
    init_dict = None
    if os.path.exists(args.init):
        with open(args.init, "rb") as f:
            init_dict = pickle.load(f)

    # --- Validate init against actual frame dimensions ---
    frames = list_frames(args.source)
    if not frames:
        print(f"No frames found in {args.source}")
        return

    sample = cv.imread(os.path.join(args.source, frames[0][1]), cv.IMREAD_GRAYSCALE)
    if sample is None:
        print("Could not read a sample frame to determine image dimensions.")
        return
    img_h, img_w = sample.shape

    nozzle_x_in_init = init_dict["nozzle_mid_point"][0][0] if init_dict else img_w + 1
    if init_dict is None or nozzle_x_in_init > img_w:
        print(f"CV_init.pkl nozzle x={nozzle_x_in_init} is outside frame width={img_w}.")
        print("Auto-estimating CV init from annotations.csv ...")
        ann_csv = os.path.join(args.output, "annotations.csv")
        if not os.path.exists(ann_csv):
            print(f"ERROR: {ann_csv} not found. Run initialisation mode (-M i) to create a proper CV_init.pkl.")
            return
        init_dict = estimate_init_from_annotations(ann_csv, img_h, img_w)
        if init_dict is None:
            print("ERROR: Could not estimate init — annotations.csv may be empty.")
            return
        est_path = "CV_init_estimated.pkl"
        with open(est_path, "wb") as f:
            pickle.dump(init_dict, f)
        print(f"Saved estimated init -> {est_path}")
        print("TIP: If results look wrong, run the Baroth initialisation GUI on a filtered frame:")
        print(f"  python \"{BAROTH_PYTHON}/main.py\" -M i -I data/filtered_frames -o .")

    print(f"Init: nozzle @ ({init_dict['nozzle_mid_point'][0][0]}, {init_dict['nozzle_mid_point'][1][0]}), "
          f"upper_cut={init_dict['upper_cut_off']}, lower_cut={init_dict['lower_cut_off']}")

    # --- Angle lookup for rotation correction ---
    angle_lookup = _build_angle_lookup(
        os.path.join(args.output, "annotations.csv"),
        os.path.join(args.output, "cv_metrics.csv"),
    )
    print(f"Angle lookup: {len(angle_lookup)} frames "
          f"(rotation applied when |tilt| > {ROTATION_THRESHOLD_DEG}°)")

    # --- Metrics CSV ---
    metrics_path = os.path.join(args.output, "cv_metrics.csv")
    header = (
        ["frame", "status", "total_length_px", "angle_deg"]
        + [f"dia_{p}pct_px" for p in DIA_PCT]
    )
    metrics_f = open(metrics_path, "w", newline="")
    metrics_w = csv.writer(metrics_f)
    metrics_w.writerow(header)

    if args.show:
        cv.namedWindow("Annotated", cv.WINDOW_GUI_EXPANDED)

    if args.limit > 0:
        frames = frames[: args.limit]

    n_ok = n_fail = 0
    tic = time.time()

    for _frame_num, frame_name in frames:
        img = cv.imread(os.path.join(args.source, frame_name), cv.IMREAD_GRAYSCALE)
        if img is None:
            continue

        tilt = angle_lookup.get(frame_name, 0.0)
        shape, status, meas = process_frame(img, init_dict, tilt_right_deg=tilt)

        if status == "Ok":
            n_ok += 1

            cimg = annotate_frame(img, shape, meas, init_dict)
            cv.imwrite(os.path.join(out_annotated, frame_name), cimg)

            mask = make_mask(img.shape, shape)
            cv.imwrite(os.path.join(out_masks, frame_name), mask)

            stem = os.path.splitext(frame_name)[0]
            with lzma.open(os.path.join(out_shapes, stem + ".xz"), "wb") as fp:
                pickle.dump(shape, fp)

            row = [frame_name, "ok", f"{meas['total_length_px']:.2f}", f"{meas['angle_deg']:.3f}"]
            row += [f"{d:.2f}" if d != -1 else "-1" for d in meas["diameters_px"]]
            metrics_w.writerow(row)

            if args.show:
                cv.imshow("Annotated", cimg)
                cv.waitKey(1)
        else:
            n_fail += 1
            metrics_w.writerow([frame_name, status] + ["-1"] * (len(header) - 2))

    metrics_f.close()
    if args.show:
        cv.destroyAllWindows()

    elapsed = time.time() - tic
    total = n_ok + n_fail
    print(f"\n{n_ok}/{total} frames annotated in {elapsed:.1f}s "
          f"({elapsed / max(1, total):.2f}s/frame)")
    print(f"  Annotated images  -> {out_annotated}/")
    print(f"  Segmentation masks -> {out_masks}/")
    print(f"  Shape files       -> {out_shapes}/")
    print(f"  Metrics           -> {metrics_path}")


if __name__ == "__main__":
    main()
