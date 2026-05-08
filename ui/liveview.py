"""
Live Gaussian spot tracker for the tilt test setup.

Usage (from the repo root):
    python -m ui.liveview --exposure 10000

Keyboard shortcuts:
  c      clear all accumulated data and reset the clock
  p      pause / resume acquisition
  i      toggle camera image between linear and log scale
  space  add a vertical marker line on all time-domain plots
  w      write all time-domain data to a CSV file
  m      toggle image recording (saves every raw frame to a folder as .npy)
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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit

from TiltTest.lab.Thorlabs.CS126MU import CS126MU


# ---------------------------------------------------------------------------
# 2-D Gaussian model  (no rotation, separate σ_x / σ_y, flat background)
# ---------------------------------------------------------------------------

def _gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, background):
    x, y = xy
    exponent = (x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)
    return (background + amplitude * np.exp(-exponent)).ravel()


def _block_average(image, factor):
    h, w = image.shape
    h2, w2 = h // factor, w // factor
    return image[:h2 * factor, :w2 * factor].reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def fit_spot(image, bin_factor):
    """Fit a 2-D Gaussian; spatially bins by *bin_factor* for speed, returns coords in original pixel units."""
    img = _block_average(image, bin_factor)

    # Background subtracted image is only used for initial parameter guess!
    bg_guess = float(np.percentile(img, 5))
    img_sub = np.clip(img - bg_guess, 0, None)
    total = img_sub.sum()
    if total == 0:
        raise RuntimeError("Image is blank after background subtraction")

    y_idx, x_idx = np.indices(img.shape)

    # Moment-based initial guesses
    x0_g = float((x_idx * img_sub).sum() / total)
    y0_g = float((y_idx * img_sub).sum() / total)
    amp_g = float(img_sub.max())
    sx_g = max(float(np.sqrt(((x_idx - x0_g) ** 2 * img_sub).sum() / total)), 1.0)
    sy_g = max(float(np.sqrt(((y_idx - y0_g) ** 2 * img_sub).sum() / total)), 1.0)

    p0 = [amp_g, x0_g, y0_g, sx_g, sy_g, bg_guess]
    xy = (x_idx.ravel(), y_idx.ravel())
    popt, _ = curve_fit(_gaussian_2d, xy, img.ravel(), p0=p0, maxfev=1000)

    amplitude, x0, y0, sigma_x, sigma_y, background = popt
    peak = amplitude + background  # absolute peak counts, physically bounded by bit depth
    # Scale back to original pixel units
    scale = float(bin_factor)
    return x0 * scale, y0 * scale, abs(sigma_x) * scale, abs(sigma_y) * scale, peak, amplitude, background


# ---------------------------------------------------------------------------
# UI class  (runs on the main thread)
# ---------------------------------------------------------------------------

class LiveViewUI:
    _IMG_DISPLAY_FACTOR = 8

    def __init__(self, exposure_us, stack, plate_scale_mm, output_path=None):
        self.plate_scale_mm = plate_scale_mm
        self._output_path = output_path
        self._lock  = threading.Lock()
        self._dirty = threading.Event()
        self._segment = 0
        self._segment_start_time = 0.0

        # Latest measurement snapshot written by acquisition thread,
        # read by main thread in _refresh().
        self._pending = None

        self._paused       = False
        self._log_scale    = False
        self._last_img_disp = None
        self._exposure_us  = exposure_us
        self._stack        = stack
        self._recording    = False
        if output_path is not None:
            stem = Path(output_path).stem
        else:
            stem = datetime.now().strftime("tilt_%Y%m%d_%H%M%S")
        self._record_dir = Path(f"{stem}_frames")

        # --- Figure -----------------------------------------------------------
        self.fig = plt.figure(figsize=(13, 11), layout="constrained")
        self._update_title()
        gs = GridSpec(4, 2, figure=self.fig, height_ratios=[2, 2, 1, 1])
        ax_xy    = self.fig.add_subplot(gs[0:2, 0])
        ax_x     = self.fig.add_subplot(gs[2:4, 0])
        ax_y     = ax_x.twinx()
        ax_img   = self.fig.add_subplot(gs[0:2, 1])
        ax_sigma = self.fig.add_subplot(gs[2:4, 1], sharex=ax_x)
        ax_amp   = ax_sigma.twinx()

        ax_xy.set_xlabel("X (mm)")
        ax_xy.set_ylabel("Y (mm)")
        ax_xy.set_title("Spot position")
        ax_xy.set_aspect("equal", adjustable="datalim")
        ax_xy.invert_yaxis()
        self._dbg_stats = ax_xy.text(
            0.01, 0.01, "", transform=ax_xy.transAxes,
            fontsize=7, family="monospace", va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            zorder=5,
        )

        ax_img.set_title("Camera image", fontsize=8)
        ax_img.set_aspect("equal", adjustable="datalim")
        ax_img.set_xlabel("X (mm)", fontsize=7)
        ax_img.set_ylabel("Y (mm)", fontsize=7)
        ax_img.tick_params(labelsize=6)
        self._img_artist = None
        self._img_cb     = None
        (self._img_pt,) = ax_img.plot([], [], "+", color="red", markersize=10, markeredgewidth=1.5)
        self._img_el = Ellipse((0, 0), width=1, height=1, fill=False, edgecolor="red", linewidth=1.2)
        ax_img.add_patch(self._img_el)

        ax_x.set_xlabel("Time (s)")
        ax_x.set_ylabel("X (mm)", color="steelblue")
        ax_x.tick_params(axis="y", labelcolor="steelblue")
        ax_x.set_title("Spot X / Y vs time")

        ax_y.set_ylabel("Y (mm)", color="tomato")
        ax_y.tick_params(axis="y", labelcolor="tomato")

        ax_sigma.set_xlabel("Time (s)")
        ax_sigma.set_ylabel("σ (mm)")
        ax_sigma.set_title("Spot σ & amplitude vs time")

        ax_amp.set_ylabel("Amplitude (counts)", color="seagreen")
        ax_amp.tick_params(axis="y", labelcolor="seagreen")

        self._hist_sc  = ax_xy.scatter([], [], s=25, alpha=0.8, linewidths=0, zorder=1)
        self._curr_sc  = ax_xy.scatter([], [], s=80, zorder=2, edgecolors="black", linewidths=1.0)
        self._sat_text = ax_xy.text(
            0.5, 0.5, "SATURATED", transform=ax_xy.transAxes,
            fontsize=22, fontweight="bold", color="red", alpha=0.35,
            ha="center", va="center", rotation=30,
            visible=False, zorder=0,
        )
        _mk = dict(marker=".", markersize=5)
        (self._line_x,)   = ax_x.plot([], [], color="steelblue", lw=1.2, **_mk)
        (self._line_y,)   = ax_y.plot([], [], color="tomato",    lw=1.2, **_mk)
        (self._line_sx,)  = ax_sigma.plot([], [], color="steelblue", lw=1.2, label="σ_x", **_mk)
        (self._line_sy,)  = ax_sigma.plot([], [], color="tomato",    lw=1.2, label="σ_y", **_mk)
        ax_sigma.legend(fontsize=8, loc="upper right")
        (self._line_amp,) = ax_amp.plot([], [], color="seagreen", lw=1.2, **_mk)

        self._ax_xy    = ax_xy
        self._ax_x     = ax_x
        self._ax_y     = ax_y
        self._ax_img   = ax_img
        self._ax_sigma = ax_sigma
        self._ax_amp   = ax_amp
        self._time_axes = (ax_x, ax_sigma)

        # --- Data storage (main thread only) ----------------------------------
        self._times    = []
        self._segments = []
        self._xs       = []
        self._ys       = []
        self._sigma_xs = []
        self._sigma_ys = []
        self._amps     = []
        self._span_artists = []

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _update_title(self):
        stack_label = f"  |  Stack {self._stack}" if self._stack > 1 else ""
        pause_label = "  |  PAUSED" if self._paused else ""
        rec_label   = "  |  REC●" if self._recording else ""
        self.fig.suptitle(
            f"Tilt Test Live View  |  Exposure {self._exposure_us} µs"
            f"{stack_label}{pause_label}{rec_label}"
            f"  |  Segment {self._segment}"
            f"  |  'c' Clear  |  'p' Pause  |  'i' Log  |  'space' Add Segment  |  'w' Save  |  'm' Record",
            fontsize=10,
        )

    def _apply_img_norm(self, img_disp):
        """Apply linear or log normalisation to the image artist and update the colorbar."""
        norm = LogNorm() if self._log_scale else Normalize()
        self._img_artist.set_norm(norm)
        self._img_cb.update_normal(self._img_artist)
        scale_label = "log" if self._log_scale else "linear"
        self._ax_img.set_title(f"Camera image  [{scale_label}]", fontsize=8)

    # ------------------------------------------------------------------
    # Called from acquisition thread
    # ------------------------------------------------------------------

    def add_measurement(self, t, x0, y0, sx, sy, amp, fit_amp, fit_bg,
                        image, saturated, sat_level):
        with self._lock:
            self._pending = dict(
                t=t, x0=x0, y0=y0, sx=sx, sy=sy, amp=amp,
                fit_amp=fit_amp, fit_bg=fit_bg,
                image=image, saturated=saturated, sat_level=sat_level,
            )
        self._dirty.set()

    # ------------------------------------------------------------------
    # Called from main thread only
    # ------------------------------------------------------------------

    def _refresh(self):
        """Update all artists from buffered data. Must be called from main thread."""
        if not self._dirty.is_set():
            return
        self._dirty.clear()

        with self._lock:
            p = self._pending
            self._pending = None

        if p is None or self._paused:
            return

        t, x0, y0, sx, sy   = p["t"], p["x0"], p["y0"], p["sx"], p["sy"]
        amp, fit_amp, fit_bg = p["amp"], p["fit_amp"], p["fit_bg"]
        image, saturated, sat_level = p["image"], p["saturated"], p["sat_level"]

        if self._recording:
            fname = (
                f"seg{self._segment:04d}"
                f"_exp{self._exposure_us}us"
                f"_stack{self._stack}"
                f"_t{t:.3f}s.npy"
            )
            np.save(self._record_dir / fname, image)

        psm = self.plate_scale_mm

        self._times.append(t)
        self._segments.append(self._segment)
        self._xs.append(x0 * psm)
        self._ys.append(y0 * psm)
        self._sigma_xs.append(sx * psm)
        self._sigma_ys.append(sy * psm)
        self._amps.append(amp)

        # XY scatter: colored by segment
        if len(self._xs) > 1:
            self._hist_sc.set_offsets(np.c_[self._xs[:-1], self._ys[:-1]])
            colors = [plt.cm.tab10(s % 10) if isinstance(s, int) else (0, 0, 0, 0)
                      for s in self._segments[:-1]]
            self._hist_sc.set_facecolor(colors)
        self._curr_sc.set_offsets([[self._xs[-1], self._ys[-1]]])
        self._curr_sc.set_facecolor([plt.cm.tab10(self._segment % 10)])
        self._ax_xy.relim()
        self._ax_xy.update_datalim(np.c_[self._xs, self._ys])
        self._ax_xy.autoscale_view()

        self._line_x.set_data(self._times, self._xs)
        self._ax_x.relim(); self._ax_x.autoscale_view()

        self._line_y.set_data(self._times, self._ys)
        self._ax_y.relim(); self._ax_y.autoscale_view()

        self._line_sx.set_data(self._times, self._sigma_xs)
        self._line_sy.set_data(self._times, self._sigma_ys)
        self._ax_sigma.relim(); self._ax_sigma.autoscale_view()

        self._line_amp.set_data(self._times, self._amps)
        self._ax_amp.relim(); self._ax_amp.autoscale_view()

        # Update the debug status
        self._sat_text.set_visible(saturated)
        self._dbg_stats.set_text(
            f"img max={image.max():.0f}  mean={image.mean():.1f}  median={np.median(image):.1f}\n"
            f"sat_level={sat_level}  sat_px={(image >= sat_level).sum()}\n"
            f"fit amp={fit_amp:.1f}  bg={fit_bg:.1f}  peak={amp:.1f}"
        )

        # Update the live view image
        img_disp = _block_average(image, self._IMG_DISPLAY_FACTOR)
        self._last_img_disp = img_disp
        h_px, w_px = image.shape
        extent_mm = [0, w_px * psm, h_px * psm, 0]
        if self._img_artist is None:
            self._img_artist = self._ax_img.imshow(
                img_disp, cmap="inferno", origin="upper",
                extent=extent_mm, interpolation="nearest",
            )
            self._ax_img.set_aspect("equal", adjustable="datalim")
            self._img_cb = self.fig.colorbar(
                self._img_artist, ax=self._ax_img,
                location="top", pad=0.02, fraction=0.046,
            )
            self._img_cb.set_label("Counts", fontsize=7)
            self._img_cb.ax.tick_params(labelsize=6)
            self._apply_img_norm(img_disp)
        else:
            self._img_artist.set_data(img_disp)
            self._apply_img_norm(img_disp)

        # Update the fitted spot position and sigma elipse
        cx_mm, cy_mm = x0 * psm, y0 * psm
        self._img_pt.set_data([cx_mm], [cy_mm])
        self._img_el.set_center((cx_mm, cy_mm))
        self._img_el.set_width(2 * sx * psm)
        self._img_el.set_height(2 * sy * psm)

        self.fig.canvas.draw()

    def run(self, interval):
        """Main-thread event loop. Returns when the figure is closed."""
        plt.ion()
        plt.show(block=False)
        while plt.fignum_exists(self.fig.number):
            self._refresh()
            self.fig.canvas.flush_events()
            time.sleep(interval)

    def _on_key(self, event):
        if event.key == "c":
            # Clear all the data
            self._times.clear(); self._segments.clear()
            self._xs.clear(); self._ys.clear()
            self._sigma_xs.clear(); self._sigma_ys.clear(); self._amps.clear()
            self._segment = 0
            self._pending = None
            self._hist_sc.set_offsets(np.empty((0, 2)))
            self._curr_sc.set_offsets(np.empty((0, 2)))
            for line in (self._line_x, self._line_y,
                         self._line_sx, self._line_sy, self._line_amp):
                line.set_data([], [])
            for artist in self._span_artists:
                artist.remove()
            self._span_artists.clear()
            self._segment_start_time = 0.0
            for ax in (self._ax_xy, self._ax_y, self._ax_amp, *self._time_axes):
                ax.relim()
                ax.autoscale_view()
            self._update_title()
            self.fig.canvas.draw()

        elif event.key == "p":
            if self._paused and self._times:  # resuming: break the curves
                nan = float("nan")
                self._times.append(nan); self._segments.append(nan)
                self._xs.append(nan); self._ys.append(nan)
                self._sigma_xs.append(nan); self._sigma_ys.append(nan); self._amps.append(nan)
            self._paused = not self._paused
            self._update_title()
            self.fig.canvas.draw()

        elif event.key == "i":
            self._log_scale = not self._log_scale
            if self._img_artist is not None and self._last_img_disp is not None:
                self._apply_img_norm(self._last_img_disp)
                self.fig.canvas.draw()

        elif event.key == " ":
            # Fill background of completed segment and start a new one
            x_pos = self._times[-1] if self._times else 0.0
            color = plt.cm.tab10(self._segment % 10)
            for ax in self._time_axes:
                self._span_artists.append(
                    ax.axvspan(self._segment_start_time, x_pos,
                               facecolor=color, alpha=0.15, linewidth=0, zorder=0)
                )
            self._segment_start_time = x_pos
            self._segment += 1
            self._update_title()
            self.fig.canvas.draw()

        elif event.key == "w":
            self._save_csv()

        elif event.key == "m":
            self._recording = not self._recording
            if self._recording:
                self._record_dir.mkdir(parents=True, exist_ok=True)
                print(f"[liveview] recording started → {self._record_dir.resolve()}")
            else:
                print(f"[liveview] recording stopped  → {self._record_dir}")
            self._update_title()
            self.fig.canvas.draw()

    def _save_csv(self):
        if not self._times:
            print("[liveview] no data to save", file=sys.stderr)
            return
        if self._output_path is not None:
            path = Path(self._output_path).with_suffix(".csv")
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(f"tilt_{stamp}.csv")
        data = np.column_stack([self._times, self._segments, self._xs, self._ys,
                                self._sigma_xs, self._sigma_ys, self._amps])
        np.savetxt(path, data, delimiter=",",
                   header="time_s,segment,x_mm,y_mm,sigma_x_mm,sigma_y_mm,amplitude_counts",
                   comments="")
        print(f"[liveview] saved {len(self._times)} rows → {path.resolve()}")

        pdf_path = path.with_suffix(".pdf")
        orig_size = self.fig.get_size_inches()
        self.fig.set_size_inches(16, 9)  # 0.8 × Full HD (1536 × 864 px at 96 dpi)
        self.fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=150)
        self.fig.set_size_inches(*orig_size)
        print(f"[liveview] saved figure  → {pdf_path.resolve()}")


# ---------------------------------------------------------------------------
# Acquisition thread  (runs in background)
# ---------------------------------------------------------------------------

class AcquisitionThread(threading.Thread):
    def __init__(self, ui, cam, bin_factor, sat_level, sat_fraction, stack, interval):
        super().__init__(daemon=True)
        self._ui           = ui
        self._cam          = cam
        self._bin_factor   = bin_factor
        self._sat_level    = sat_level
        self._sat_fraction = sat_fraction
        self._stack        = stack
        self._interval     = interval
        self._stop_event   = threading.Event()
        self._t0           = time.monotonic()

    def stop(self):
        self._stop_event.set()

    def elapsed(self):
        return time.monotonic() - self._t0

    def run(self):
        while not self._stop_event.is_set():
            frame_start = time.monotonic()
            try:
                image = self._cam.acquireStack(self._stack)
                saturated = (
                    float((image >= self._sat_level).sum()) / image.size
                    >= self._sat_fraction
                )
                x0, y0, sx, sy, amp, fit_amp, fit_bg = fit_spot(image, self._bin_factor)
            except (RuntimeError, ValueError, TimeoutError) as exc:
                print(f"[liveview] frame error: {exc}", file=sys.stderr)
            else:
                t = self.elapsed()
                self._ui.add_measurement(
                    t, x0, y0, sx, sy, amp, fit_amp, fit_bg,
                    image, saturated, self._sat_level,
                )

            elapsed = time.monotonic() - frame_start
            remaining = self._interval - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live Gaussian spot tracker for tilt test"
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
        "--bin", type=int, default=4, dest="bin_factor",
        help="Spatial binning factor used for Gaussian fit (default: 4)",
    )
    parser.add_argument(
        "--plate-scale", type=float, default=85.9, dest="plate_scale_um",
        help="Plate scale in µm/px (default: 85.9)",
    )
    parser.add_argument(
        "--stack", type=int, default=1, dest="stack",
        help="Number of frames to average per measurement (default: 1)",
    )
    parser.add_argument(
        "--sat-fraction", type=float, default=1e-6, dest="sat_fraction",
        help="Fraction of pixels at full scale that triggers saturation warning (default: 1e-6)",
    )
    parser.add_argument(
        "--sat-level", type=int, default=None, dest="sat_level",
        help="Override saturation level in counts (default: 2^bit_depth - 1)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path stem for 'w' save; .csv and .pdf are appended automatically (default: tilt_YYYYMMDD_HHMMSS)",
    )
    args = parser.parse_args()
    plate_scale_mm = args.plate_scale_um * 1e-3

    ui = LiveViewUI(args.exposure, args.stack, plate_scale_mm, output_path=args.output)

    cam = CS126MU(camera_index=args.camera)
    cam.setExposure(args.exposure)
    cam.pollTimeout(args.timeout)

    with cam:
        sat_level = args.sat_level if args.sat_level is not None else (1 << cam.bit_depth) - 1
        acq = AcquisitionThread(
            ui=ui,
            cam=cam,
            bin_factor=args.bin_factor,
            sat_level=sat_level,
            sat_fraction=args.sat_fraction,
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
