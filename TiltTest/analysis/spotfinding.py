"""
Spot-finding methods for the tilt test live view.

Two interchangeable estimators, both returning a SpotMeasurement:
  - GaussianSpotFinder:  2-D Gaussian fit (amplitude, sigma_x, sigma_y, background).
  - ThresholdSpotFinder: fixed-level threshold + connected-component mask,
                         circle fit to the mask boundary.
"""

import abc
from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.ndimage
import scipy.optimize


def _block_average(image, factor):
    if factor <= 1:
        return image
    h, w = image.shape
    h2, w2 = h // factor, w // factor
    return image[:h2 * factor, :w2 * factor].reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def _block_all(mask, factor):
    """Block-reduce a boolean mask: a binned pixel is True only if every raw pixel it covers is."""
    if factor <= 1:
        return mask
    h, w = mask.shape
    h2, w2 = h // factor, w // factor
    return mask[:h2 * factor, :w2 * factor].reshape(h2, factor, w2, factor).all(axis=(1, 3))


@dataclass
class SpotMeasurement:
    x0: float
    y0: float
    sigma_x: float  # Gaussian: sigma along image x-axis. Threshold: ellipse semi-major axis.
    sigma_y: float  # Gaussian: sigma along image y-axis. Threshold: ellipse semi-minor axis.
    peak: float
    amplitude: float
    background: float
    theta: float = 0.0  # rotation of sigma_x axis from image x-axis, radians. 0 unless fitted.


class SpotFinder(abc.ABC):
    """
    clip=(left, bottom, right, top) excludes that many raw pixels from each
    edge of the frame from spot finding entirely -- e.g. where the target
    screen's physical edge clips the spot and the image shows background
    instead. Excluded pixels are dropped from the fit rather than treated as
    background, in both subclasses.
    """

    def __init__(self, bin_factor=1, clip=(0, 0, 0, 0)):
        self.bin_factor = bin_factor
        self.clip_left, self.clip_bottom, self.clip_right, self.clip_top = clip
        self._valid_mask_cache = None  # (raw_shape, binned_mask)

    @abc.abstractmethod
    def find(self, image: np.ndarray) -> SpotMeasurement:
        """Locate the spot in *image* (raw pixel units in, raw pixel units out)."""

    def _binned_valid_mask(self, raw_shape):
        """Valid-pixel mask at the binned image's resolution, or None if nothing is clipped."""
        left, bottom, right, top = self.clip_left, self.clip_bottom, self.clip_right, self.clip_top
        if not any((left, bottom, right, top)):
            return None
        if self._valid_mask_cache is None or self._valid_mask_cache[0] != raw_shape:
            h, w = raw_shape
            mask = np.ones(raw_shape, dtype=bool)
            if top:
                mask[:top, :] = False
            if bottom:
                mask[h - bottom:, :] = False
            if left:
                mask[:, :left] = False
            if right:
                mask[:, w - right:] = False
            self._valid_mask_cache = (raw_shape, _block_all(mask, self.bin_factor))
        return self._valid_mask_cache[1]


# ---------------------------------------------------------------------------
# Gaussian fit
# ---------------------------------------------------------------------------

def _gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, background):
    x, y = xy
    exponent = (x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)
    return (background + amplitude * np.exp(-exponent)).ravel()


class GaussianSpotFinder(SpotFinder):
    def find(self, image):
        img = _block_average(image, self.bin_factor)
        valid = self._binned_valid_mask(image.shape)
        if valid is None:
            valid = np.ones(img.shape, dtype=bool)

        # Background subtracted image is only used for initial parameter guess!
        bg_guess = float(np.percentile(img[valid], 5))
        img_sub = np.clip(img - bg_guess, 0, None)
        img_sub[~valid] = 0.0  # excluded pixels contribute nothing to the moment guesses
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
        # Excluded pixels are dropped from the fit itself (not just zeroed), since
        # zeroing would tell curve_fit they're background, biasing the result.
        xy = (x_idx[valid].ravel(), y_idx[valid].ravel())
        popt, _ = scipy.optimize.curve_fit(_gaussian_2d, xy, img[valid].ravel(), p0=p0, maxfev=1000)

        amplitude, x0, y0, sigma_x, sigma_y, background = popt
        peak = amplitude + background  # absolute peak counts, physically bounded by bit depth
        scale = float(self.bin_factor)
        return SpotMeasurement(
            x0=x0 * scale, y0=y0 * scale,
            sigma_x=abs(sigma_x) * scale, sigma_y=abs(sigma_y) * scale,
            peak=peak, amplitude=amplitude, background=background,
        )


# ---------------------------------------------------------------------------
# Threshold + ellipse fit
# ---------------------------------------------------------------------------

def _fit_ellipse(x, y):
    """
    Direct least-squares ellipse fit (Fitzgibbon, Pilu & Fisher 1999).

    Fits the general conic A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0 under the
    ellipse-specific constraint B^2 - 4AC < 0, via a generalized eigenvalue
    problem. Returns (x0, y0, a_major, a_minor, theta), theta = angle of the
    major axis from the x-axis, radians.
    """
    # Center and scale the points first: raw pixel coordinates are large enough
    # (up to ~1e3) that the x^2/y^2 columns of the design matrix badly condition
    # the scatter matrix otherwise.
    x_mean, y_mean = x.mean(), y.mean()
    xc, yc = x - x_mean, y - y_mean
    scale = max(float(np.std(xc)), float(np.std(yc)), 1.0)
    xc, yc = xc / scale, yc / scale

    D = np.column_stack([xc ** 2, xc * yc, yc ** 2, xc, yc, np.ones_like(xc)])
    S = D.T @ D
    constraint = np.zeros((6, 6))
    constraint[0, 2] = constraint[2, 0] = 2.0
    constraint[1, 1] = -1.0

    eigvals, eigvecs = scipy.linalg.eig(S, constraint)
    eigvals = eigvals.real
    valid = np.isfinite(eigvals) & (eigvals > 1e-10)
    if not np.any(valid):
        raise RuntimeError("Ellipse fit failed: no valid ellipse-specific solution")
    coeffs = eigvecs[:, np.nonzero(valid)[0][0]].real
    A, B, C, Dc, E, F = coeffs

    center_mat = np.array([[2 * A, B], [B, 2 * C]])
    x0n, y0n = np.linalg.solve(center_mat, [-Dc, -E])
    F0 = A * x0n ** 2 + B * x0n * y0n + C * y0n ** 2 + Dc * x0n + E * y0n + F

    M = np.array([[A, B / 2], [B / 2, C]])
    evals, evecs = np.linalg.eigh(M)
    axes_sq = -F0 / evals
    if np.any(axes_sq <= 0):
        raise RuntimeError("Ellipse fit failed: degenerate conic")
    axes = np.sqrt(axes_sq) * scale

    order = np.argsort(axes)[::-1]  # descending: major, minor
    a_major, a_minor = float(axes[order[0]]), float(axes[order[1]])
    major_vec = evecs[:, order[0]]
    theta = float(np.arctan2(major_vec[1], major_vec[0]))

    x0 = float(x0n * scale + x_mean)
    y0 = float(y0n * scale + y_mean)
    return x0, y0, a_major, a_minor, theta


class ThresholdSpotFinder(SpotFinder):
    """
    Thresholds the (binned) image at a fixed absolute count level, keeps the
    brightest connected component, and fits an ellipse to its boundary
    (the setup images the circular spot obliquely, so it projects as an
    ellipse).

    The threshold should sit between the background level and the diffuse
    spot's plateau, below any superposed specular reflection: the mask's
    spatial extent is set by the diffuse footprint regardless of specular
    amplitude, as long as the specular reflection doesn't spatially extend
    past the diffuse spot (e.g. via blooming at saturation).

    `peak`/`amplitude` report the max pixel value inside the mask, which
    will reflect a superposed specular reflection if present -- these are
    not directly comparable to GaussianSpotFinder's fitted amplitude.

    If clip excludes part of the frame (e.g. a screen edge clipping the spot),
    boundary pixels adjacent to the excluded region are dropped before fitting
    -- they mark the artificial cut, not the spot's true edge -- so the ellipse
    is fit to the remaining visible arc only.
    """

    def __init__(self, bin_factor=1, threshold=None, clip=(0, 0, 0, 0)):
        super().__init__(bin_factor, clip=clip)
        if threshold is None:
            raise ValueError("ThresholdSpotFinder requires an explicit threshold")
        self.threshold = threshold

    def find(self, image):
        img = _block_average(image, self.bin_factor)
        valid = self._binned_valid_mask(image.shape)
        if valid is None:
            valid = np.ones(img.shape, dtype=bool)

        mask = (img > self.threshold) & valid
        if not mask.any():
            raise RuntimeError(f"No pixels above threshold ({self.threshold}) in the valid region")

        labeled, n_components = scipy.ndimage.label(mask)
        if n_components > 1:
            sums = scipy.ndimage.sum(img, labeled, index=range(1, n_components + 1))
            best = int(np.argmax(sums)) + 1
            mask = labeled == best

        boundary = mask & ~scipy.ndimage.binary_erosion(mask)
        # Boundary pixels next to the excluded region are clipping artifacts
        # (a straight cut), not part of the spot's true elliptical edge.
        clipped = scipy.ndimage.binary_dilation(~valid, structure=np.ones((3, 3), dtype=bool)) & valid
        arc_boundary = boundary & ~clipped
        y_idx, x_idx = np.nonzero(arc_boundary)
        if len(x_idx) < 8:
            raise RuntimeError("Not enough unclipped boundary pixels for an ellipse fit")

        xc, yc, a_major, a_minor, theta = _fit_ellipse(x_idx.astype(float), y_idx.astype(float))

        peak = float(img[mask].max())
        outside = img[valid & ~mask]
        if outside.size:
            background = float(np.median(outside))
        else:
            background = float(np.median(img[valid])) if valid.any() else float(np.median(img))

        scale = float(self.bin_factor)
        return SpotMeasurement(
            x0=xc * scale, y0=yc * scale,
            sigma_x=a_major * scale, sigma_y=a_minor * scale,
            peak=peak, amplitude=peak - background, background=background,
            theta=theta,
        )


def build_spot_finder(method, bin_factor, threshold=None, clip=(0, 0, 0, 0)):
    if method == "gaussian":
        return GaussianSpotFinder(bin_factor=bin_factor, clip=clip)
    elif method == "threshold":
        return ThresholdSpotFinder(bin_factor=bin_factor, threshold=threshold, clip=clip)
    raise ValueError(f"Unknown spot finding method: {method!r}")
