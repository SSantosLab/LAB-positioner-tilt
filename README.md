# LAB-positioner-tilt
Scripts and codes for performing the positioner tilt test


## Tilt Test Live Viewer

The script `ui/liveview.py` implements a live Gaussian spot tracker for the positioner tilt test setup.
Acquires frames from a Thorlabs CS126MU camera, fits a 2-D Gaussian to the spot, and displays the spot position and fit parameters in real time.

### Setup

```bash
pip install -r requirements.txt
```

The Thorlabs TSI SDK (`thorlabs_tsi_sdk`) is not on PyPI and must be installed separately from the [Thorlabs Scientific Imaging SDK](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam).

On `pos-control` these requirements are usually already preinstalled, and the corresponding environment can be actived via conda:

```bash
conda activate positioner-env
```

### Usage

Run from the repository root:

```bash
python -m ui.liveview --exposure 10000
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--exposure` | *(required)* | Camera exposure time in µs |
| `--plate-scale` | `85.9` | Plate scale in µm/px |
| `--bin` | `4` | Spatial binning factor for Gaussian fit |
| `--interval` | `0.5` | Minimum seconds between frames |
| `--camera` | `0` | Camera index |

![Screenshot of the Tilt Live View Utility](_images/TiltLiveView_Screenshot.png)

### Saving data

Press `w` at any time to write all accumulated data to disk. Two files are created with the same name stem: a CSV containing one row per frame, and a PDF snapshot of the current figure. The output path can be set with `--output`; otherwise files are named `tilt_YYYYMMDD_HHMMSS.csv/.pdf` in the working directory.

The CSV columns are `time_s`, `segment`, `x_mm`, `y_mm`, `sigma_x_mm`, `sigma_y_mm`, `amplitude_counts`. The `segment` column records which data segment each measurement belongs to — segments are incremented by pressing `Space` and reset to 0 by pressing `c`. Pause/resume break-points are stored as `NaN` rows.

### Keyboard shortcuts

| Key | Action |
| --- | --- |
| `c` | Clear all accumulated data and reset the clock |
| `p` | Pause / resume acquisition |
| `i` | Toggle camera image between linear and log scale |
| `Space` | Start a new data segment. |


## Camera Live Viewer

The script `ui/cameraview.py` implements a simple live camera viewer of the Thorlabs CS126MU camera.
No spot finding is carried out, and no pixel to mm conversion is done!

### Setup

Equivalent for the `ui.liveview.py` utility above!

### Usage

Run from the repository root:

```bash
python -m ui.cameraview --exposure 10000
```

Key arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--exposure` | *(required)* | Camera exposure time in µs |
| `--stack` | `1` | Average multiple consecutive frames |
| `--camera` | `0` | Camera index |
| `--display-bin` | `1` | Spatial binning for faster image display |
