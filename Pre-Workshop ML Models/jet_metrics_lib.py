"""
jet_metrics_lib.py
==================
Single source of truth for jet shape metrics and video overlay drawing.

Everything here works from an ordered list of skeleton points
(top-to-bottom, in pixel coordinates), e.g.:
    [(x0, y0), (x1, y1), ..., (xN, yN)]

Two public functions:
    compute_all_metrics(points) -> dict   # all geometric metrics
    draw_annotations(frame, skeleton, metrics) -> annotated frame

And two public constants:
    METRIC_SCHEMA   # ordered list of metric column names (for CSV headers)
    KEY_METRICS     # which 3 metrics get burned onto the video

NOTE on width/diameter:
    Width metrics are intentionally NOT here. The U-Net was trained on masks
    that dilate the skeleton to a constant thickness, so its predicted mask
    has no real width information. Width would require either retraining on
    true-width masks or measuring directly from the raw frame.
"""

import numpy as np
import cv2


# ── Which metrics go in the CSV, and in what order ───────────────────────────
METRIC_SCHEMA = [
    "arc_length_px",         # total length of the jet path
    "chord_length_px",       # straight-line nozzle->contact distance
    "straightness_ratio",    # arc / chord  (1.0 = perfectly straight)
    "chord_angle_deg",       # overall jet angle from vertical (nozzle->contact)
    "launch_angle_deg",      # angle from vertical at the nozzle
    "contact_angle_deg",     # angle from vertical at the substrate
    "net_turn_deg",          # contact_angle - launch_angle (signed total bend)
    "total_turning_deg",     # sum of |angle changes| along the jet (whip/wiggle)
    "max_deflection_px",     # furthest the jet strays from the chord line
    "deflection_pos_pct",    # where that max deflection is (0%=nozzle,100%=substrate)
    "mean_curvature_invpx",  # average |curvature| along the jet
    "max_curvature_invpx",   # peak |curvature|
    "min_radius_px",         # tightest bend radius (1 / max curvature)
    "bending_energy",        # integral of curvature^2 along the jet
    "n_skeleton_pts",        # number of points in the skeleton
]


# ── Which 3 metrics get drawn on the video (label, metric key, format) ───────
# To change what's shown on the video, edit these three lines.
KEY_METRICS = [
    ("Contact angle", "contact_angle_deg",  "{:+.1f} deg"),
    ("Straightness",  "straightness_ratio", "{:.2f}"),
    ("Jet length",    "arc_length_px",      "{:.0f} px"),
]


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _angle_from_vertical(dx, dy):
    """
    Angle of a vector from straight-down vertical, in degrees.
      straight down (dx=0, dy>0) -> 0
      leaning right               -> positive
      leaning left                -> negative
    """
    return float(np.degrees(np.arctan2(dx, dy)))


def _wrap_deg(a):
    """Wrap an angle difference into (-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


def _point_to_line_dist(pt, a, b):
    """Perpendicular distance from pt to the infinite line through a and b."""
    x0, y0 = pt
    x1, y1 = a
    x2, y2 = b
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = np.hypot(y2 - y1, x2 - x1)
    return float(num / den) if den > 0 else 0.0


# ── Main metrics computation ─────────────────────────────────────────────────

def compute_all_metrics(points):
    """
    Compute the full geometric metric suite from ordered skeleton points.
    Returns a dict keyed by METRIC_SCHEMA names. Missing/undefined values
    are returned as "" (empty string) so CSV rows always line up.
    """
    # Default everything to empty
    out = {k: "" for k in METRIC_SCHEMA}

    if not points or len(points) < 3:
        out["n_skeleton_pts"] = len(points) if points else 0
        return out

    pts = np.array(points, dtype=float)

    # Segment vectors and lengths
    diffs = np.diff(pts, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    seg_len = np.where(seg_len == 0, 1e-9, seg_len)  # guard zero-length

    arc_length   = float(seg_len.sum())
    chord_vec    = pts[-1] - pts[0]
    chord_length = float(np.hypot(*chord_vec))
    straightness = arc_length / chord_length if chord_length > 1e-9 else 1.0

    # Cumulative arc position (0 at nozzle, arc_length at substrate)
    arc_pos = np.concatenate([[0.0], np.cumsum(seg_len)])

    # Angles
    chord_angle  = _angle_from_vertical(chord_vec[0], chord_vec[1])
    launch_angle = _angle_from_vertical(diffs[0, 0],  diffs[0, 1])
    contact_angle = _angle_from_vertical(diffs[-1, 0], diffs[-1, 1])
    net_turn = _wrap_deg(contact_angle - launch_angle)

    # Per-segment direction angles -> total turning (sum of |wrapped deltas|)
    seg_angles = np.degrees(np.arctan2(diffs[:, 0], diffs[:, 1]))
    total_turning = float(sum(abs(_wrap_deg(seg_angles[i + 1] - seg_angles[i]))
                              for i in range(len(seg_angles) - 1)))

    # Max deflection from the chord, and where it occurs
    max_defl = 0.0
    max_defl_idx = 0
    for i in range(1, len(pts) - 1):
        d = _point_to_line_dist(pts[i], pts[0], pts[-1])
        if d > max_defl:
            max_defl = d
            max_defl_idx = i
    defl_pos_pct = 100.0 * arc_pos[max_defl_idx] / arc_length if arc_length > 0 else 0.0

    # Signed Menger curvature at each interior point (1/px)
    curvatures = []
    ds_list = []
    for i in range(1, len(pts) - 1):
        A, B, C = pts[i - 1], pts[i], pts[i + 1]
        cross = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
        area = 0.5 * cross
        dAB = np.hypot(*(B - A))
        dBC = np.hypot(*(C - B))
        dCA = np.hypot(*(A - C))
        denom = dAB * dBC * dCA
        kappa = (4.0 * area / denom) if denom > 1e-9 else 0.0
        curvatures.append(kappa)
        ds_list.append(0.5 * (dAB + dBC))

    curv = np.array(curvatures) if curvatures else np.array([0.0])
    ds   = np.array(ds_list)   if ds_list   else np.array([0.0])

    mean_curv = float(np.mean(np.abs(curv)))
    max_curv  = float(np.max(np.abs(curv)))
    min_radius = (1.0 / max_curv) if max_curv > 1e-6 else ""
    bending_energy = float(np.sum((curv ** 2) * ds))

    # Fill output
    out["arc_length_px"]        = round(arc_length, 2)
    out["chord_length_px"]      = round(chord_length, 2)
    out["straightness_ratio"]   = round(straightness, 4)
    out["chord_angle_deg"]      = round(chord_angle, 2)
    out["launch_angle_deg"]     = round(launch_angle, 2)
    out["contact_angle_deg"]    = round(contact_angle, 2)
    out["net_turn_deg"]         = round(net_turn, 2)
    out["total_turning_deg"]    = round(total_turning, 2)
    out["max_deflection_px"]    = round(max_defl, 2)
    out["deflection_pos_pct"]   = round(defl_pos_pct, 1)
    out["mean_curvature_invpx"] = round(mean_curv, 6)
    out["max_curvature_invpx"]  = round(max_curv, 6)
    out["min_radius_px"]        = round(min_radius, 1) if min_radius != "" else ""
    out["bending_energy"]       = round(bending_energy, 6)
    out["n_skeleton_pts"]       = len(points)

    return out


# ── Video overlay drawing ────────────────────────────────────────────────────

def _draw_text_box(img, lines, origin=(8, 8), bg=(0, 0, 0), alpha=0.45,
                   font_scale=0.55, text_color=(255, 255, 255)):
    """Draw a translucent box with text lines at the top-left."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    pad = 7
    line_h = int(24 * font_scale / 0.55)

    # Measure widest line
    max_w = 0
    for ln in lines:
        (tw, th), _ = cv2.getTextSize(ln, font, font_scale, thickness)
        max_w = max(max_w, tw)

    x0, y0 = origin
    box_w = max_w + 2 * pad
    box_h = len(lines) * line_h + 2 * pad

    # Translucent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), bg, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Text
    for i, ln in enumerate(lines):
        ty = y0 + pad + (i + 1) * line_h - 6
        cv2.putText(img, ln, (x0 + pad, ty), font, font_scale,
                    text_color, thickness, cv2.LINE_AA)


def draw_annotations(frame, skeleton, metrics=None,
                     skel_color=(0, 230, 0), thickness=2):
    """
    Draw the jet skeleton and burn the 3 key metrics onto the frame.

    Args:
        frame    : BGR image (numpy array)
        skeleton : list of (x, y) points, or None
        metrics  : dict from compute_all_metrics, or None/empty

    Returns:
        Annotated BGR image (a copy; the input is not modified).
    """
    out = frame.copy()

    # Skeleton polyline + endpoints
    if skeleton and len(skeleton) > 1:
        pts = np.array(skeleton, dtype=np.int32)
        cv2.polylines(out, [pts], isClosed=False, color=skel_color,
                      thickness=thickness, lineType=cv2.LINE_AA)
        cv2.circle(out, tuple(skeleton[0]),  4, (255, 120, 0), -1)  # nozzle (cyan-blue)
        cv2.circle(out, tuple(skeleton[-1]), 4, (0,  80, 255), -1)  # contact (orange)

    # Metric overlay
    if metrics:
        lines = []
        for label, key, fmt in KEY_METRICS:
            val = metrics.get(key, "")
            if val == "" or val is None:
                continue
            try:
                lines.append(f"{label}: {fmt.format(float(val))}")
            except Exception:
                lines.append(f"{label}: {val}")
        if lines:
            _draw_text_box(out, lines)
    else:
        _draw_text_box(out, ["No jet detected"], bg=(0, 0, 120), alpha=0.5)

    return out


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build a synthetic curved jet (quarter-circle-ish) to sanity-check
    t = np.linspace(0, 1, 25)
    xs = 600 + 80 * (t ** 2)          # curves right as it descends
    ys = 20 + 160 * t
    pts = [(float(x), float(y)) for x, y in zip(xs, ys)]

    m = compute_all_metrics(pts)
    print("Metrics on synthetic curved jet:")
    for k in METRIC_SCHEMA:
        print(f"  {k:22s}: {m[k]}")
