"""
Live camera image viewer for the tilt test setup.

Usage (from the repo root):
    python -m ui.cameraview --exposure 10000

Keyboard shortcuts:
  p  pause / resume acquisition
  i  toggle image between linear and log scale
  w  save current frame as PNG
"""

import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from TiltTest.lab.Thorlabs.CS126MU import CS126MU


def _block_average(image, factor):
    h, w = image.shape
    h2, w2 = h // factor, w // factor
    return image[:h2 * factor, :w2 * factor].reshape(h2, factor, w2, factor).mean(axis=(1, 3))


# ---------------------------------------------------------------------------
# UI class  (runs on the main thread)
# ---------------------------------------------------------------------------

class CameraViewUI:

    def __init__(self, exposure_us, display_bin=1):
        self._exposure_us  = exposure_us
        self._display_bin  = display_bin
        self._lock        = threading.Lock()
        self._dirty       = threading.Event()
        self._pending     = None
        self._paused      = False
        self._log_scale   = False
        self._last_image  = None

        self.fig, self._ax = plt.subplots(figsize=(9, 7), layout="constrained")
        self._update_title()

        self._ax.set_xlabel("X (px)")
        self._ax.set_ylabel("Y (px)")
        self._ax.set_aspect("equal", adjustable="datalim")

        self._img_artist = None
        self._img_cb     = None

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _update_title(self):
        pause_label = "  |  PAUSED" if self._paused else ""
        self.fig.suptitle(
            f"Camera View  |  Exposure {self._exposure_us} µs"
            f"{pause_label}"
            f"  |  'p' Pause  |  'i' Log  |  'w' Save",
            fontsize=10,
        )

    def _apply_norm(self):
        norm = LogNorm() if self._log_scale else Normalize()
        self._img_artist.set_norm(norm)
        self._img_cb.update_normal(self._img_artist)
        scale_label = "log" if self._log_scale else "linear"
        self._ax.set_title(f"[{scale_label}]", fontsize=8)

    # ------------------------------------------------------------------
    # Called from acquisition thread
    # ------------------------------------------------------------------

    def add_frame(self, image):
        with self._lock:
            self._pending = image
        self._dirty.set()

    # ------------------------------------------------------------------
    # Called from main thread only
    # ------------------------------------------------------------------

    def _refresh(self):
        if not self._dirty.is_set():
            return
        self._dirty.clear()

        with self._lock:
            image = self._pending
            self._pending = None

        if image is None or self._paused:
            return

        self._last_image = image
        img_disp = _block_average(image, self._display_bin) if self._display_bin > 1 else image

        if self._img_artist is None:
            self._img_artist = self._ax.imshow(
                img_disp, cmap="inferno", origin="upper", interpolation="nearest",
            )
            self._img_cb = self.fig.colorbar(
                self._img_artist, ax=self._ax,
                location="right", pad=0.02, fraction=0.046,
            )
            self._img_cb.set_label("Counts", fontsize=8)
            self._img_cb.ax.tick_params(labelsize=7)
            self._apply_norm()
        else:
            self._img_artist.set_data(img_disp)
            self._apply_norm()

        self.fig.canvas.draw()

    def run(self, interval):
        plt.ion()
        plt.show(block=False)
        while plt.fignum_exists(self.fig.number):
            self._refresh()
            self.fig.canvas.flush_events()
            time.sleep(interval)

    def _on_key(self, event):
        if event.key == "p":
            self._paused = not self._paused
            self._update_title()
            self.fig.canvas.draw()

        elif event.key == "i":
            self._log_scale = not self._log_scale
            if self._img_artist is not None:
                self._apply_norm()
                self.fig.canvas.draw()

        elif event.key == "w":
            self._save_png()

    def _save_png(self):
        if self._last_image is None:
            print("[cameraview] no frame to save", file=sys.stderr)
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(f"frame_{stamp}.png")
        plt.imsave(path, self._last_image, cmap="inferno")
        print(f"[cameraview] saved frame → {path.resolve()}")


# ---------------------------------------------------------------------------
# Acquisition thread  (runs in background)
# ---------------------------------------------------------------------------

class AcquisitionThread(threading.Thread):
    def __init__(self, ui, cam, stack, interval):
        super().__init__(daemon=True)
        self._ui         = ui
        self._cam        = cam
        self._stack      = stack
        self._interval   = interval
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            frame_start = time.monotonic()
            try:
                image = self._cam.acquireStack(self._stack)
            except (RuntimeError, ValueError, TimeoutError) as exc:
                print(f"[cameraview] frame error: {exc}", file=sys.stderr)
            else:
                self._ui.add_frame(image)

            elapsed = time.monotonic() - frame_start
            remaining = self._interval - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live camera image viewer for tilt test"
    )
    parser.add_argument(
        "--exposure", type=int, required=True,
        help="Camera exposure time in microseconds",
    )
    parser.add_argument(
        "--timeout", type=int, default=15000,
        help="Frame poll timeout in ms (default: 15000)",
    )
    parser.add_argument(
        "--interval", type=float, default=0.5,
        help="Minimum seconds between frames (default: 0.5)",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--stack", type=int, default=1,
        help="Number of frames to average per display update (default: 1)",
    )
    parser.add_argument(
        "--display-bin", type=int, default=1, dest="display_bin",
        help="Spatial binning factor for display (default: 1, no binning)",
    )
    args = parser.parse_args()

    ui = CameraViewUI(args.exposure, display_bin=args.display_bin)

    cam = CS126MU(camera_index=args.camera)
    cam.setExposure(args.exposure)
    cam.pollTimeout(args.timeout)

    with cam:
        acq = AcquisitionThread(
            ui=ui,
            cam=cam,
            stack=args.stack,
            interval=args.interval,
        )
        acq.start()
        try:
            ui.run(interval=0.05)
        except KeyboardInterrupt:
            pass
        finally:
            acq.stop()
            acq.join(timeout=5)


if __name__ == "__main__":
    main()
