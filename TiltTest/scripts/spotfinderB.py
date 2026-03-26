import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import mahotas as mh
from scipy.ndimage import center_of_mass  
import os
from numpy import sqrt, exp, ravel, arange
from scipy import optimize
from pylab import indices
from astropy.io import fits
from PIL import Image
from matplotlib.lines import Line2D
from skimage.filters import (threshold_isodata,threshold_li,threshold_mean,threshold_minimum,threshold_otsu,threshold_triangle,threshold_yen)
from scipy.ndimage import gaussian_filter

#can also take individual images

def gauss(x, *p):
    A, mu, sigma = p
    return A * exp(-(x - mu)**2 / (2. * sigma**2))

def gaussian(bias, height, center_x, center_y, width_x, width_y):
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: bias + height * exp(-(((center_x - x) / width_x)**2 + ((center_y - y) / width_y)**2) / 2)

def moments(data):
    bias = np.min(data)
    data_this = data - bias
    total = data_this.sum()
    X, Y = indices(data.shape)
    x = (X * data_this).sum() / total
    y = (Y * data_this).sum() / total
    col = data_this[:, int(y)]
    width_x = sqrt(abs((arange(col.size) - y)**2 * col).sum() / col.sum())
    row = data_this[int(x), :]
    width_y = sqrt(abs((arange(row.size) - x)**2 * row).sum() / row.sum())
    height = data_this.max()
    return bias, height, x, y, width_x, width_y

def fitgaussian(data):
    params = moments(data)
    errorfunction = lambda p: ravel(gaussian(*p)(*indices(data.shape)) - data)
    p, _ = optimize.leastsq(errorfunction, params)
    return p

def remove_hot_pixels(image, nsigma=5):
    im_mean = np.mean(image)
    im_sig = np.std(image)
    hot_thresh = im_mean + nsigma * im_sig
    hp_img = np.copy(image).astype(np.uint32)
    low_values_indices = hp_img < hot_thresh
    hp_img[low_values_indices] = 0
    ind = zip(*np.where(hp_img > hot_thresh))
    xlimit = len(hp_img[0])
    ylimit = len(hp_img)
    for i in ind:
        if i[0] == 0 or i[0] == ylimit - 1 or i[1] == 0 or i[1] == xlimit - 1:
            image[i[0], i[1]] = np.median(image)
        else:
            neighborsum = hp_img[i[0] + 1, i[1]] + hp_img[i[0] - 1, i[1]] + hp_img[i[0], i[1] - 1] + hp_img[i[0], i[1] + 1]
            if neighborsum == 0:
                image[i[0], i[1]] = (image[i[0] + 1, i[1]] + image[i[0] - 1, i[1]] +
                                     image[i[0], i[1] - 1] + image[i[0], i[1] + 1]) / 4.
    return image

def im2bw(image, level):
    bw = np.zeros_like(image, dtype=int)
    bw[image > level] = 1
    return bw

def multiCens(img, n_centroids_to_keep=2, verbose=False, write_fits=False, no_otsu=True, save_dir='', size_fitbox=10):
    img[img < 0] = 0
    img = remove_hot_pixels(img, 7).astype(np.uint16)
    level_fraction_of_peak = 0.1
    level_frac = int(level_fraction_of_peak * np.max(img))
    level = level_frac if no_otsu else max(mh.thresholding.otsu(img), level_frac)
    bw = im2bw(img, level)
    labeled, nr_objects = mh.label(bw)
    sizes = mh.labeled.labeled_size(labeled)
    sorted_sizes_indexes = np.argsort(sizes)[::-1]
    good_spot_indexes = sorted_sizes_indexes[1:n_centroids_to_keep + 1]
    if len(good_spot_indexes) < n_centroids_to_keep and not no_otsu:
        return multiCens(img, n_centroids_to_keep, verbose, write_fits, True)

    FWHMSub, xCenSub, yCenSub, peaks = [], [], [], []
    centers = center_of_mass(labeled, labels=labeled, index=[good_spot_indexes])
    nbox = size_fitbox
    for i, x in enumerate(centers):
        x = x[0]
        px = int(round(x[1]))
        py = int(round(x[0]))
        data = img[py - nbox:py + nbox, px - nbox:px + nbox]
        params = fitgaussian(data)
        fwhm = abs(2.355 * max(params[4], params[5]))
        if fwhm < .5:
            sbox = nbox - 1
            data = img[py - sbox:py + sbox, px - sbox:px + sbox]
            params = fitgaussian(data)
            fwhm = abs(2.355 * max(params[4], params[5]))
        xCenSub.append(float(px) - float(nbox) + params[3])
        yCenSub.append(float(py) - float(nbox) + params[2])
        FWHMSub.append(fwhm)
        peaks.append(params[1])
    return xCenSub, yCenSub, peaks, FWHMSub

class SpotFinder():
    def __init__(self, file_path=None, image=None, nspots=1, print_summary=False, verbose=False,
                 max_counts=2**16 - 1, min_energy=0.3, fboxsize=7,region_file=None,img=None,
                 gaussian=False,gaussian_sig=3,gaussian_kwargs={},otsu=False,li=False,mean=False,minimum=False,
                 isodata=False,triangle=False,yen=False):
        self.version = 0.1
        self.verbose = verbose
        self.nspots = nspots
        self.max_counts = max_counts
        self.min_energy = min_energy
        self.fboxsize = fboxsize
        self.region_file = region_file
        self.img = img
        self.fits_name = file_path

        if image is not None:
            self.img = image
            if self.verbose:
                print("[SpotFinder] Using image array directly.")
        elif file_path:
            try:
                if file_path.endswith('.npy'):
                    self.img = np.load(file_path, allow_pickle=True)
                elif file_path.endswith('.fits'):
                    fits_file = fits.open(file_path)
                    self.img = fits_file[0].data
                elif file_path.endswith(('.bmp', '.png', '.jpg','.tif')):
                    image = Image.open(file_path)
                    self.img = np.array(image.convert('L'))  # Grayscale

                else:
                    raise ValueError("Unsupported file format. Use .npy, .fits, .bmp, .png, .tif, or .jpg.")
            except Exception as e:
                raise ValueError(f"Error loading file: {file_path}\n{e}")
        else:
            raise ValueError("You must provide either `file_path` or `image` to SpotFinder.")
        if gaussian:
            new_img = gaussian_filter(self.img,gaussian_sig,**gaussian_kwargs)
            self.img = new_img
        if otsu:
            self.threshold = threshold_otsu(self.img)
            self.img[self.img < self.threshold] = 0
        elif li:
            self.threshold = threshold_li(self.img)
            self.img[self.img < self.threshold] = 0
        elif mean:
            self.threshold = threshold_mean(self.img)
            self.img[self.img < self.threshold] = 0
        elif minimum:
            self.threshold = threshold_minimum(self.img)
            self.img[self.img < self.threshold] = 0
        elif triangle:
            self.threshold = threshold_triangle(self.img)
            self.img[self.img < self.threshold] = 0
        elif yen:
            self.threshold = threshold_yen(self.img)
            self.img[self.img < self.threshold] = 0
        elif isodata:
            self.threshold = threshold_isodata(self.img)
            self.img[self.img < self.threshold] = 0

    def get_centroids(self, print_summary=False, region_file=None):
        if self.img is None:
            raise ValueError("No image loaded.")

        xCenSub, yCenSub, peaks, FWHMSub = multiCens(
            self.img,
            n_centroids_to_keep=self.nspots,
            verbose=self.verbose,
            write_fits=False,
            size_fitbox=self.fboxsize
        )
        energy = [FWHMSub[i] * (peaks[i] / self.max_counts) for i in range(len(peaks))]
        sindex = sorted(range(len(peaks)), key=lambda k: peaks[k])
        peaks_sorted = [peaks[i] for i in sindex]
        x_sorted = [xCenSub[i] for i in sindex]
        y_sorted = [yCenSub[i] for i in sindex]
        fwhm_sorted = [FWHMSub[i] for i in sindex]
        energy_sorted = [energy[i] for i in sindex]

        centroids = {
            'peaks': peaks_sorted,
            'x': x_sorted,
            'y': y_sorted,
            'fwhm': fwhm_sorted,
            'energy': energy_sorted
        }

        if print_summary:
            print(" Spot  x          y         FWHM    Peak     LD")
            for i, x in enumerate(x_sorted):
                line = f"{i+1:5d} {x:9.3f} {y_sorted[i]:9.3f} {fwhm_sorted[i]:6.2f}  {peaks_sorted[i]:7.0f} {energy_sorted[i]:7.2f}"
                if energy_sorted[i] < self.min_energy:
                    line += ' *'
                print(line)

        return centroids
