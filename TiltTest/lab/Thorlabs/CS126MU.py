

import os
import tifffile
import numpy as np
from datetime import datetime

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK


class CS126MU():

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.exposure_time_us = 100
        self.poll_Timeout_ms = 2000
        self._sdk = None
        self._camera = None
        self.image_width = None
        self.image_height = None
        self.bit_depth = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def setExposure(self, us=100):
        self.exposure_time_us = us
        if self._camera is not None:
            self._camera.exposure_time_us = us
        return f"Exposure time has been set to {us} microseconds (min. 28 µs)"

    def pollTimeout(self, ms=2000):
        self.poll_Timeout_ms = ms
        if self._camera is not None:
            self._camera.image_poll_timeout_ms = ms
        return f"Image poll timeout has been set to {ms} ms"

    # ------------------------------------------------------------------
    # Persistent-session API (open once, acquire many frames, close)
    # ------------------------------------------------------------------

    def open(self):
        """Open the SDK and camera for repeated use. Call close() when done."""
        sdk = TLCameraSDK()
        try:
            cameras = sdk.discover_available_cameras()
            if not cameras:
                raise RuntimeError("No cameras detected")
            if self.camera_index >= len(cameras):
                raise RuntimeError(
                    f"Camera index {self.camera_index} out of range "
                    f"({len(cameras)} camera(s) found)"
                )
            camera = sdk.open_camera(cameras[self.camera_index])
            camera.frames_per_trigger_zero_for_unlimited = 1
            camera.exposure_time_us = self.exposure_time_us
            camera.image_poll_timeout_ms = self.poll_Timeout_ms
            self.image_width = camera.image_width_pixels
            self.image_height = camera.image_height_pixels
            self.bit_depth = camera.bit_depth
            camera.arm(2)
        except Exception:
            sdk.dispose()
            raise
        self._sdk = sdk
        self._camera = camera

    def close(self):
        """Disarm and release the camera and SDK."""
        if self._camera is not None:
            try:
                self._camera.disarm()
            except Exception:
                pass
            try:
                self._camera.dispose()
            except Exception:
                pass
            self._camera = None
        if self._sdk is not None:
            try:
                self._sdk.dispose()
            except Exception:
                pass
            self._sdk = None

    def acquireFrame(self):
        """
        Acquire a single frame from the already-open camera.
        Returns a 2-D numpy array (height × width).
        Raises RuntimeError if the camera is not open, TimeoutError on timeout.
        """
        if self._camera is None:
            raise RuntimeError("Camera is not open. Call open() first.")
        self._camera.issue_software_trigger()
        frame = self._camera.get_pending_frame_or_null()
        if frame is None:
            raise TimeoutError("Timeout while polling for a frame")
        return np.array(frame.image_buffer).reshape(self.image_height, self.image_width)

    def acquireStack(self, n=1):
        """
        Acquire *n* frames and return their mean as a float64 array (height × width).
        For n=1 this is equivalent to acquireFrame() but always returns float64.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        acc = None
        for _ in range(n):
            img = self.acquireFrame().astype(np.float64)
            acc = img if acc is None else acc + img
        return acc / n

    # ------------------------------------------------------------------
    # Legacy single-shot API (opens and closes SDK per call)
    # ------------------------------------------------------------------

    def takeImage(self, NUMBER_OF_IMAGES = 1):

        """Takes n images and returns Data"""
        image_data = []

        with TLCameraSDK() as sdk:
            cameras = sdk.discover_available_cameras()
            if len(cameras) == 0:
                print("Error: no cameras detected!")

            with sdk.open_camera(cameras[self.camera_index]) as camera:
                #  setup the camera for continuous acquisition
                camera.frames_per_trigger_zero_for_unlimited = 0
                camera.exposure_time_us = self.exposure_time_us
                camera.image_poll_timeout_ms = self.poll_Timeout_ms  # 2 second timeout
                camera.arm(2)


                # begin acquisition
                camera.issue_software_trigger()
                frames_counted = 0
                while frames_counted < NUMBER_OF_IMAGES:
                    frame = camera.get_pending_frame_or_null()
                    if frame is None:
                        raise TimeoutError("Timeout was reached while polling for a frame, program will now exit")

                    frames_counted += 1

                    image_data.append(frame.image_buffer)

        return image_data

    def saveImage(self, directory_path, NUMBER_OF_IMAGES = 1):

        """Takes n images and saves them to the specified path as a TIFF file"""
        OUTPUT_DIRECTORY = os.path.abspath(directory_path)  # Directory the TIFFs will be saved to
        now = datetime.now()
        formatedDate = now.strftime("%Y%m%d_%H%M%S")
        FILENAME =  formatedDate+'_' + str(self.exposure_time_us) + '.tif'  # The filename of the TIFF

        TAG_BITDEPTH = 32768
        TAG_EXPOSURE = 65535


        # delete image if it exists
        if os.path.exists(OUTPUT_DIRECTORY + os.sep + FILENAME):
            os.remove(OUTPUT_DIRECTORY + os.sep + FILENAME)

        with TLCameraSDK() as sdk:
            cameras = sdk.discover_available_cameras()
            if len(cameras) == 0:
                print("Error: no cameras detected!")

            with sdk.open_camera(cameras[self.camera_index]) as camera:
                #  setup the camera for continuous acquisition
                camera.frames_per_trigger_zero_for_unlimited = 0
                camera.exposure_time_us = self.exposure_time_us
                camera.image_poll_timeout_ms = self.poll_Timeout_ms  # 2 second timeout
                camera.arm(2)

                # save these values to place in our custom TIFF tags later
                bit_depth = camera.bit_depth
                exposure = camera.exposure_time_us

                # need to save the image width and height for color processing
                image_width = camera.image_width_pixels
                image_height = camera.image_height_pixels




                # begin acquisition
                camera.issue_software_trigger()
                frames_counted = 0
                while frames_counted < NUMBER_OF_IMAGES:
                    frame = camera.get_pending_frame_or_null()
                    if frame is None:
                        raise TimeoutError("Timeout was reached while polling for a frame, program will now exit")

                    frames_counted += 1

                    image_data = frame.image_buffer


                    with tifffile.TiffWriter(OUTPUT_DIRECTORY + os.sep + FILENAME, append=True) as tiff:
                        """
                            Setting append=True here means that calling tiff.save will add the image as a page to a multipage TIFF.
                        """
                        tiff.write(data=image_data,  # np.ushort image data array from the camera
                                compression=1,   # amount of compression (0-9), by default it is uncompressed (0)
                                extratags=[(TAG_BITDEPTH, 'I', 1, bit_depth, False),  # custom TIFF tag for bit depth
                                            (TAG_EXPOSURE, 'I', 1, exposure, False)]  # custom TIFF tag for exposure
                                )
                        """
                            If compress > 0 tifffile will compress the image using zlib - deflate compression.
                            Instead of an int a str can be supplied to specify a different compression algorithm;
                                e.g. compress = 'lzma'
                            View the tifffile source or online to see what is supported.
                        """
                        """
                            The extratags parameter allows the user to specify additional tags. Programs will typically ignore
                            any tags from 32768 onward, which is where the bit depth and exposure have been placed. The
                            syntax for extra tags is (tag_code, data_type_of_value, number_of_values, value, write_once).
                            View the tifffile source for more information.
                        """


                camera.disarm()


            """
            Reading tiffs - to test that the tags from before worked, we're going to read back the tags on the first page.
            Note that custom TIFF tags are not going to be picked up by normal TIFF viewers, but can be read programmatically
            if the tag code is known.
            """
            # open file

            with tifffile.TiffFile(OUTPUT_DIRECTORY + os.sep + FILENAME) as tiff_read:
                if len(tiff_read.pages) < 1:
                    raise ValueError("No pages were found in multipage TIFF")
                page_one = tiff_read.pages[0]
                print("First Image: Bit Depth = {} bpp, Exposure Time = {} ms".format(page_one.tags[str(TAG_BITDEPTH)].value,
                                                                                    page_one.tags[str(TAG_EXPOSURE)].value/1000))
        return FILENAME
