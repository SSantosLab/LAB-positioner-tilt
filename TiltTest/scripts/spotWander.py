#!/usr/bin/env python
# coding: utf-8
import random
from skimage.feature import blob_log
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import os

baseDir = "/Users/sean/Desktop/Repos/ThorlabsImages/imgs/lightsOff_LEDOn"

allFiles = os.listdir(baseDir)

print(f"Starting {len(allFiles)} measurements")

start=200

allBlobs = []

for file in allFiles:
    img = io.imread(os.path.join(baseDir,file), as_gray=True)
    blobs = blob_log(img,
                 min_sigma=start,
                 max_sigma=start+50,
                 num_sigma=2,
                 threshold=1E-5)
    blobs[:, 2] = blobs[:, 2] * 1.414  
    allBlobs.append(blobs)

outDir = "/Users/sean/Desktop/Repos/LAB-positioner-tilt/data"
fileName = f"wander_results_{len(allFiles)}.npy"

np.save(os.path.join(outDir,fileName),allBlobs)

print(f"File saved to {os.path.join(outDir,fileName)}")
