import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time
import os
import json

pth = "/home/pos-control/Andrin/LAB-positioner-tilt/TiltTest"

sys.path.append(pth)
from lab.Thorlabs.CS126MU import CS126MU
cam1 = CS126MU()

def enhance_contrast(image_path,threshold_value = 5, save = False):
    
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    print("Original dtype: ", img.dtype)
    print("Min/Max before: ", img.min(), img.max())

    img = img.astype(np.float32) #Covert to float for scaling$
    #img = cv2.blur(img, (3,3))
    _, img = cv2.threshold(img, threshold_value, img.max(), cv2.THRESH_TOZERO)

    print("Min/Max after blurring: ", img.min(), img.max())

    #stretched = (img - img.min())*(4095/(img.max()-img.min()))
    #stretched = np.clip(stretched, 0, 4095)
    plt.imshow(img, cmap='inferno', origin='upper', norm=mpl.colors.LogNorm()) #1,100))
    plt.colorbar(label='Intensity')
    plt.title("Logarithmic Intensity Plot with threshold @" +str(threshold_value))
    plt.axis()
    plt.grid(True)
    if(save):
        plt.savefig("tilt-test/plots/" +str(time.time())+".png", dpi=300, bbox_inches='tight')
    plt.show()
    #cv2.imwrite("streched/1.tif", stretched.astype(np.uint16))
    #print("Min/Max after: ", stretched.min(), stretched.max())

def plot_tiff_logscale(image_path, save = False):
    # Load 16-bit grayscale image
    img = Image.open(image_path)
    
    if img.mode != 'I;16':
        img = img.convert('I;16')

    img_array = np.array(img, dtype=np.uint16)


    # Plot with a colormap (e.g. 'viridis', 'inferno', 'plasma')
    plt.imshow(img_array, cmap='inferno', origin='upper', norm=mpl.colors.LogNorm())
    plt.colorbar(label='Intensity')
    plt.title("Logarithmic Intensity Plot")
    plt.axis('off')
    plt.grid(True)
    if(save):
        plt.savefig(os.path.join(os.path.dirname(fullPath),str(time.time())+".png"), dpi=300, bbox_inches='tight')
    plt.show()

def load_camera_config(file_path):
    config_path = os.path.join(pth,"/data/camera_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at{config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config

class thorlabs_camera():
    def __init__(self, camera_object):
        self.dataDir = "data"
        self.camera = camera_object

    def take_images(self,axis,stagesPos,n_imgaes=5, plot = False):
        axisDir = str(axis)+"Scan"
        outDir = os.path.join(str(stagesPos),axisDir,self.dataDir,pth)
        os.makedirs(outDir,exist_ok=True)

        for k in range(n_imgaes):
            fullpath = self.camera.saveImage(outDir)
        
        if (plot):
            plot_tiff_logscale(fullpath,False)

