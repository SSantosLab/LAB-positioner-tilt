from pypylon import pylon as pypylon
from pypylon import genicam
import numpy as np

class acA3800_14um():

    def __init__(self):
        tlf = pypylon.TlFactory.GetInstance()

        available_cameras = tlf.EnumerateDevices()
        # available_ids = ...

        # This assumes we only have one camera connected.
        self._cam = None
        self._cam = pypylon.InstantCamera(tlf.CreateDevice(available_cameras[0]))
        self._cam.Open()


        # Set default configs
        self._cam.PixelFormat.Value = 'Mono12'
        self._cam.OffsetX.Value = 0
        self._cam.OffsetY.Value = 0
        self._cam.Width.Value = self._cam.Width.Max
        self._cam.Height.Value = self._cam.Height.Max

    def close(self):
        if self._cam is not None:
            self._cam.Close()
            self._cam = None


    def setExposureTime(self, exposure_us):
        self._cam.ExposureTime.Value = exposure_us


    def takeImage(self):
        self._cam.StartGrabbing(1)
        grabResult = self._cam.RetrieveResult(500, pypylon.TimeoutHandling_Return)

        image = None
        if grabResult.GrabSucceeded():
            image = grabResult.Array.astype(np.uint16)

        grabResult.Release()
        self._cam.StopGrabbing()

        return image
