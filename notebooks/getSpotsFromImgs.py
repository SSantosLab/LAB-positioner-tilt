from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from photutils.detection import find_peaks
import itertools
import os

# Compute the super dark

off_off_dir="../../ThorlabsImages/imgs/lightsOff_LEDOff"
files = os.listdir(off_off_dir)

superDark = np.zeros(np.array(Image.open(os.path.join(off_off_dir,files[0])),dtype=np.int32).shape)
for f in files:
    superDark += np.array(Image.open(os.path.join(off_off_dir,f)),dtype=np.int32)

normalizedSuperDark = superDark/len(files)

# Make a mask of the diffuser region

simpleCorners = np.array([
    [ 447,    0],
    [3197,    0],
    [ 447, 2715],
    [3197, 2715]])

xc = [447,3197]
yc = [0,2715]

mask = np.ones(np.shape(np.array(normalizedSuperDark)))
prevFirst,prevSecond = xc[-1],yc[-1]
for firstC in xc:
    for secondC in yc:
        mask[prevSecond:secondC,prevFirst:firstC] = 0
        prevSecond = secondC
    prevFirst = firstC
nativeMask = mask.astype(int).tolist()

# Get each spot picture

off_on_dir="../../ThorlabsImages/imgs/lightsOff_LEDOn"
files = os.listdir(off_on_dir)

daofind = DAOStarFinder(fwhm=250, threshold=100,brightest=1)

sources = []


for f in files:
    spotImage = np.array(Image.open(os.path.join(off_on_dir,f)),dtype=np.int32)

    sources.append(daofind(spotImage,mask=nativeMask,))

    # sourcedSpots = sources.append(sourcedSpots,np.array(sources["xcentroid","ycentroid","sharpness","roundness1","roundness2","npix","peak"],dtype=np.float32))
print(sources)
saveLoc = "darkSpots.npy"
np.save(saveLoc,sources)

print(f"File saved at {saveLoc}")
