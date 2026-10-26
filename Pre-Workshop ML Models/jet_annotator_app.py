"""
Jet Annotator — GUI App
========================
A simple desktop app to run your trained U-Net on a video.

Pick a video, click Process, and the app will:
  - run the U-Net on every frame to detect the jet
  - overlay the jet skeleton on each frame
  - save an annotated MP4
  - save a metrics.csv with per-frame jet measurements

Requirements:
  - Must sit in your project root, alongside:
        phase4b_unet_model.py   (model architecture)
        phase7b_infer_video.py  (inference helpers)
        checkpoints/unet_best.pth (trained weights)
  - Tkinter (ships with Python — nothing to install)
  - Pillow is optional; if installed you also get a live preview.

Run:
    python jet_annotator_app.py
"""

import sys
import os
import csv
import time
import queue
import threading
import traceback
import subprocess
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
# torch is only needed when loading a .pth model; ONNX models don't require it.
try:
    import torch
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

# Inference helpers — single source of truth, reused from Phase 7b.
# (phase7b imports the JetUNet architecture from phase4b for us.)
import phase7b_infer_video as infer

# Optional live preview via Pillow
try:
    from PIL import Image, ImageTk
    PIL_OK = True
except Exception:
    PIL_OK = False


# CSV schema — pulled from the metrics library so it always matches the
# full metric suite. frame_idx/status are prepended by us.
CSV_SCHEMA = ["frame_idx", "status"] + list(infer.METRIC_SCHEMA)

DEFAULT_CHECKPOINT = "checkpoints/unet_mobilenet.onnx"
DEFAULT_OUTPUT_DIR = "inference"


def open_in_file_browser(path):
    """Open a folder in the OS file browser."""
    path = str(path)
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception as e:
        messagebox.showerror("Could not open folder", str(e))


class JetAnnotatorApp:
    def __init__(self, root, initial_video=None):
        self.root = root
        root.title("Jet Annotator")
        root.geometry("760x720")
        root.minsize(680, 600)

        # Worker communication
        self.msg_queue = queue.Queue()
        self.worker = None
        self.stop_event = threading.Event()
        self.crop = None      # (x, y, w, h) in full-frame coords, or None for full frame
        self.last_output_dir = None
        self._imgtk = None  # keep a ref so preview isn't garbage-collected

        self._build_ui()
        self._poll_queue()

        # Pre-fill a video if one was passed (e.g. dragged onto the app icon)
        if initial_video:
            self.video_var.set(initial_video)

    # ── UI construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}
        main = ttk.Frame(self.root, padding=14)
        main.pack(fill="both", expand=True)

        # Header
        ttk.Label(main, text="Jet Annotator",
                  font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(main,
                  text="Run the trained U-Net on a video to detect and measure the polymer jet.",
                  foreground="#555").pack(anchor="w", pady=(0, 10))

        # ── Inputs frame ─────────────────────────────────────────────────────
        inputs = ttk.LabelFrame(main, text="Inputs", padding=10)
        inputs.pack(fill="x", **pad)
        inputs.columnconfigure(1, weight=1)

        # Video
        ttk.Label(inputs, text="Video file:").grid(row=0, column=0, sticky="w")
        self.video_var = tk.StringVar()
        ttk.Entry(inputs, textvariable=self.video_var).grid(
            row=0, column=1, sticky="ew", padx=6)
        ttk.Button(inputs, text="Browse…", command=self._browse_video).grid(
            row=0, column=2)

        # Checkpoint
        ttk.Label(inputs, text="Model file:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.ckpt_var = tk.StringVar(value=DEFAULT_CHECKPOINT)
        ttk.Entry(inputs, textvariable=self.ckpt_var).grid(
            row=1, column=1, sticky="ew", padx=6, pady=(8, 0))
        ttk.Button(inputs, text="Browse…", command=self._browse_ckpt).grid(
            row=1, column=2, pady=(8, 0))

        # Output dir
        ttk.Label(inputs, text="Output folder:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.out_var = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        ttk.Entry(inputs, textvariable=self.out_var).grid(
            row=2, column=1, sticky="ew", padx=6, pady=(8, 0))
        ttk.Button(inputs, text="Browse…", command=self._browse_out).grid(
            row=2, column=2, pady=(8, 0))

        # ── Options frame ────────────────────────────────────────────────────
        options = ttk.LabelFrame(main, text="Options", padding=10)
        options.pack(fill="x", **pad)
        options.columnconfigure(1, weight=1)

        ttk.Label(options, text="Detection confidence:").grid(row=0, column=0, sticky="w")
        self.conf_var = tk.DoubleVar(value=0.5)
        self.conf_scale = ttk.Scale(options, from_=0.1, to=0.9,
                                    variable=self.conf_var, orient="horizontal",
                                    command=self._update_conf_label)
        self.conf_scale.grid(row=0, column=1, sticky="ew", padx=8)
        self.conf_label = ttk.Label(options, text="0.50", width=5)
        self.conf_label.grid(row=0, column=2)

        ttk.Label(options, text="Frame scale:").grid(row=1, column=0, sticky="w")
        self.scale_var = tk.DoubleVar(value=1.0)
        self.scale_scale = ttk.Scale(options, from_=0.25, to=1.0,
                                     variable=self.scale_var, orient="horizontal",
                                     command=self._update_scale_label)
        self.scale_scale.grid(row=1, column=1, sticky="ew", padx=8)
        self.scale_label = ttk.Label(options, text="1.00", width=5)
        self.scale_label.grid(row=1, column=2)

        # Crop controls
        ttk.Label(options, text="Crop region:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        crop_btns = ttk.Frame(options)
        crop_btns.grid(row=2, column=1, sticky="w", padx=8, pady=(8, 0))
        ttk.Button(crop_btns, text="Select…", command=self._select_crop).pack(side="left")
        ttk.Button(crop_btns, text="Clear", command=self._clear_crop).pack(side="left", padx=(6, 0))
        self.crop_label = ttk.Label(options, text="full frame")
        self.crop_label.grid(row=2, column=2, sticky="w", pady=(8, 0))

        # Auto-scale to ~Dalton pixel budget (aspect-preserving)
        self.autoscale_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options,
            text="Auto-scale to ~Dalton resolution (keep aspect ratio)",
            variable=self.autoscale_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))

        self.save_video_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text="Save annotated video (MP4)",
                        variable=self.save_video_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.save_csv_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text="Save metrics (CSV)",
                        variable=self.save_csv_var).grid(
            row=2, column=0, columnspan=2, sticky="w")

        # ── Action buttons ───────────────────────────────────────────────────
        actions = ttk.Frame(main)
        actions.pack(fill="x", **pad)
        self.process_btn = ttk.Button(actions, text="Process Video",
                                      command=self._start_processing)
        self.process_btn.pack(side="left")
        self.stop_btn = ttk.Button(actions, text="Stop",
                                   command=self._stop_processing, state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        self.open_btn = ttk.Button(actions, text="Open Output Folder",
                                   command=self._open_output, state="disabled")
        self.open_btn.pack(side="right")

        # ── Progress ─────────────────────────────────────────────────────────
        prog = ttk.Frame(main)
        prog.pack(fill="x", **pad)
        self.progress = ttk.Progressbar(prog, mode="determinate")
        self.progress.pack(fill="x")
        self.progress_label = ttk.Label(prog, text="Idle")
        self.progress_label.pack(anchor="w", pady=(4, 0))

        # ── Preview ──────────────────────────────────────────────────────────
        if PIL_OK:
            prev_frame = ttk.LabelFrame(main, text="Preview", padding=6)
            prev_frame.pack(fill="x", **pad)
            self.preview_label = ttk.Label(prev_frame,
                text="Preview will appear here during processing.",
                anchor="center", foreground="#888")
            self.preview_label.pack(fill="x")

        # ── Log ──────────────────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(main, text="Log", padding=6)
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_text = tk.Text(log_frame, height=7, wrap="word",
                                state="disabled", font=("Consolas", 9))
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scroll.set)

    # ── Small UI helpers ─────────────────────────────────────────────────────
    def _update_conf_label(self, _=None):
        self.conf_label.config(text=f"{self.conf_var.get():.2f}")

    def _update_scale_label(self, _=None):
        self.scale_label.config(text=f"{self.scale_var.get():.2f}")

    def _select_crop(self):
        """Open the first frame of the selected video and let the user drag a crop box."""
        path = self.video_var.get()
        if not path or not Path(path).exists():
            messagebox.showerror("No video", "Select a valid video first.")
            return
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Read error", "Could not read the first frame of the video.")
            return

        H, W = frame.shape[:2]
        # Downscale for on-screen selection if the frame is larger than the screen
        disp_scale = min(1.0, 1200.0 / max(W, H))
        if disp_scale < 1.0:
            disp = cv2.resize(frame, (int(W * disp_scale), int(H * disp_scale)))
        else:
            disp = frame.copy()

        try:
            roi = cv2.selectROI(
                "Drag a box, then press ENTER to confirm (C to cancel)",
                disp, showCrosshair=False)
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # flush the window close on some platforms
        except Exception as e:
            messagebox.showerror(
                "Crop unavailable",
                "This OpenCV build has no GUI support for region selection.\n\n"
                f"Details: {e}")
            return

        x, y, w, h = roi
        if w == 0 or h == 0:
            return  # cancelled / empty selection

        # Map the selection back to full-resolution coordinates and clamp
        inv = 1.0 / disp_scale
        x, y, w, h = int(x * inv), int(y * inv), int(w * inv), int(h * inv)
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))

        self.crop = (x, y, w, h)
        self.crop_label.config(text=f"{w}x{h} @ ({x},{y})")

    def _clear_crop(self):
        self.crop = None
        self.crop_label.config(text="full frame")

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                       ("All files", "*.*")])
        if path:
            self.video_var.set(path)

    def _browse_ckpt(self):
        path = filedialog.askopenfilename(
            title="Select model (.onnx or .pth)",
            filetypes=[("Model files", "*.onnx *.pth *.pt"),
                       ("ONNX model", "*.onnx"),
                       ("PyTorch checkpoint", "*.pth *.pt"),
                       ("All files", "*.*")])
        if path:
            self.ckpt_var.set(path)

    def _browse_out(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.out_var.set(path)

    def _log(self, text):
        self.log_text.config(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _open_output(self):
        if self.last_output_dir:
            open_in_file_browser(self.last_output_dir)

    # ── Processing control ───────────────────────────────────────────────────
    def _start_processing(self):
        video = self.video_var.get().strip()
        ckpt  = self.ckpt_var.get().strip()
        out   = self.out_var.get().strip()

        if not video or not Path(video).exists():
            messagebox.showerror("Missing video", "Please select a valid video file.")
            return
        if not ckpt or not Path(ckpt).exists():
            messagebox.showerror("Missing model",
                f"Model file not found:\n{ckpt}\n\n"
                f"Export an ONNX model with quantize_onnx.py, or select a .pth checkpoint.")
            return
        if not out:
            out = DEFAULT_OUTPUT_DIR
            self.out_var.set(out)

        Path(out).mkdir(parents=True, exist_ok=True)
        self.last_output_dir = out

        # Reset UI state
        self.stop_event.clear()
        self.process_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.open_btn.config(state="disabled")
        self.progress.config(value=0)
        self.progress_label.config(text="Starting…")
        self._log(f"Input : {video}")
        self._log(f"Model : {ckpt}")
        self._log(f"Output: {out}")

        # Launch worker thread
        self.worker = threading.Thread(
            target=self._worker_run,
            args=(video, ckpt, out,
                  self.save_video_var.get(),
                  self.save_csv_var.get(),
                  float(self.conf_var.get()),
                  float(self.scale_var.get()),
                  self.crop,
                  bool(self.autoscale_var.get())),
            daemon=True)
        self.worker.start()

    def _stop_processing(self):
        self.stop_event.set()
        self.progress_label.config(text="Stopping…")
        self.stop_btn.config(state="disabled")

    # ── Worker thread (no Tk calls here — only queue messages) ────────────────
    def _worker_run(self, video_path, ckpt_path, out_dir, save_video, save_csv,
                    confidence, scale, crop, autoscale):
        try:
            self.msg_queue.put(("status", "Loading model…"))

            # Apply confidence threshold to the shared inference module
            infer.CONFIDENCE = confidence

            model = infer.load_model(ckpt_path, scale=scale)
            self.msg_queue.put(("log", f"Model loaded ({model.backend})."))

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.msg_queue.put(("error", "Could not open the video file."))
                return

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.msg_queue.put(("log",
                f"Video: {total} frames, {fps:.1f} fps, {w}x{h}"))
            self.msg_queue.put(("total", total))

            # Resolve crop against this video's resolution (clamp; a crop set on a
            # different-resolution video still behaves sensibly).
            if crop:
                cx, cy, cw, ch = crop
                cx = max(0, min(cx, w - 1))
                cy = max(0, min(cy, h - 1))
                cw = max(1, min(cw, w - cx))
                ch = max(1, min(ch, h - cy))
                crop = (cx, cy, cw, ch)
                eff_w, eff_h = cw, ch
                self.msg_queue.put(("log", f"Cropping to {cw}x{ch} at ({cx},{cy})"))
            else:
                eff_w, eff_h = w, h

            # Auto-scale to ~Dalton pixel budget (aspect-preserving). Overrides the
            # manual Frame scale slider when enabled. Never upscales.
            if autoscale:
                target_area = 1262 * 228   # Dalton reference frame area
                s = min(1.0, (target_area / float(eff_w * eff_h)) ** 0.5)
                model.scale = s
                self.msg_queue.put(("log",
                    f"Auto-scale to ~Dalton: scale={s:.3f} "
                    f"(inference at ~{int(eff_w*s)}x{int(eff_h*s)})"))
            elif scale != 1.0:
                self.msg_queue.put(("log", f"Frame scale: {scale:.2f}"))

            writer = None
            mp4_path = None
            if save_video:
                mp4_path = Path(out_dir) / (Path(video_path).stem + "_annotated.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (eff_w, eff_h))
                if not writer.isOpened():
                    self.msg_queue.put(("log",
                        "Warning: could not open video writer; skipping MP4."))
                    writer = None
                    mp4_path = None

            metrics_rows = []
            idx = 0
            ok  = 0
            start_t = time.time()

            while True:
                if self.stop_event.is_set():
                    self.msg_queue.put(("log", "Stopped by user."))
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # Crop before anything else so only the cropped area is analyzed
                if crop:
                    cx, cy, cw, ch = crop
                    frame = frame[cy:cy + ch, cx:cx + cw]

                mask = infer.predict_mask(model, frame)
                skeleton = infer.extract_skeleton_from_mask(mask)

                if skeleton:
                    metrics = infer.compute_all_metrics(skeleton)
                    status = "ok"
                    ok += 1
                else:
                    metrics = {}
                    status = "failed"

                annotated = infer.draw_annotations(frame, skeleton, metrics)
                if writer is not None:
                    writer.write(annotated)

                metrics_rows.append({"frame_idx": idx, "status": status, **metrics})
                idx += 1

                # Progress + preview every few frames (and on the first frame)
                if idx == 1 or idx % 5 == 0 or idx == total:
                    elapsed = time.time() - start_t
                    rate = idx / elapsed if elapsed > 0 else 0
                    eta = (total - idx) / rate if (rate > 0 and total > 0) else 0
                    self.msg_queue.put(("progress", (idx, total, rate, eta)))

                if PIL_OK and (idx == 1 or idx % 10 == 0):
                    pw = annotated.shape[1]
                    prev_scale = 640 / pw if pw > 640 else 1.0
                    if prev_scale != 1.0:
                        prev = cv2.resize(
                            annotated,
                            (int(pw * prev_scale), int(annotated.shape[0] * prev_scale)))
                    else:
                        prev = annotated.copy()
                    self.msg_queue.put(("preview", prev))

            cap.release()
            if writer is not None:
                writer.release()

            # Write metrics CSV
            csv_path = None
            if save_csv and metrics_rows:
                csv_path = Path(out_dir) / "metrics.csv"
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer_csv = csv.DictWriter(
                        f, fieldnames=CSV_SCHEMA, restval="", extrasaction="ignore")
                    writer_csv.writeheader()
                    writer_csv.writerows(metrics_rows)

            self.msg_queue.put(("done", {
                "frames": idx,
                "ok": ok,
                "mp4": str(mp4_path) if mp4_path else None,
                "csv": str(csv_path) if csv_path else None,
                "stopped": self.stop_event.is_set(),
            }))

        except Exception:
            self.msg_queue.put(("error", traceback.format_exc()))

    # ── Queue polling (runs on main thread) ──────────────────────────────────
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()

                if kind == "status":
                    self.progress_label.config(text=payload)

                elif kind == "log":
                    self._log(payload)

                elif kind == "total":
                    total = payload
                    if total and total > 0:
                        self.progress.config(mode="determinate", maximum=total, value=0)
                    else:
                        self.progress.config(mode="indeterminate")
                        self.progress.start(12)

                elif kind == "progress":
                    idx, total, rate, eta = payload
                    if total and total > 0:
                        self.progress.config(value=idx)
                        eta_str = time.strftime("%M:%S", time.gmtime(eta)) if eta else "--:--"
                        self.progress_label.config(
                            text=f"Frame {idx}/{total}   "
                                 f"{rate:.1f} fps   ETA {eta_str}")
                    else:
                        self.progress_label.config(
                            text=f"Frame {idx}   {rate:.1f} fps")

                elif kind == "preview":
                    self._show_preview(payload)

                elif kind == "done":
                    self._on_done(payload)

                elif kind == "error":
                    self._on_error(payload)

        except queue.Empty:
            pass

        self.root.after(80, self._poll_queue)

    def _show_preview(self, bgr_frame):
        if not PIL_OK:
            return
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        im.thumbnail((660, 280))
        self._imgtk = ImageTk.PhotoImage(im)
        self.preview_label.config(image=self._imgtk, text="")

    def _on_done(self, info):
        try:
            self.progress.stop()
        except Exception:
            pass
        if info.get("stopped"):
            self.progress_label.config(text="Stopped.")
        else:
            if self.progress["mode"] == "determinate":
                self.progress.config(value=self.progress["maximum"])
            self.progress_label.config(text="Done.")

        self._log(f"Processed {info['frames']} frames "
                  f"({info['ok']} with a detected jet).")
        if info.get("mp4"):
            self._log(f"Annotated video -> {info['mp4']}")
        if info.get("csv"):
            self._log(f"Metrics CSV     -> {info['csv']}")

        self.process_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.open_btn.config(state="normal")

        if not info.get("stopped"):
            messagebox.showinfo("Finished",
                f"Done — {info['ok']}/{info['frames']} frames had a detected jet.\n\n"
                f"Output saved to:\n{self.last_output_dir}")

    def _on_error(self, tb_text):
        try:
            self.progress.stop()
        except Exception:
            pass
        self.progress_label.config(text="Error.")
        self._log("ERROR:\n" + tb_text)
        self.process_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        messagebox.showerror("Processing failed",
            "Something went wrong. See the log for details.")


def main():
    # Optional: a video path passed on the command line (e.g. dragged onto the
    # app icon, or via "Open with"). Pre-fills the video field if given.
    initial_video = None
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        initial_video = sys.argv[1]

    root = tk.Tk()
    # Use a modern theme if available
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    JetAnnotatorApp(root, initial_video=initial_video)
    root.mainloop()


if __name__ == "__main__":
    main()
