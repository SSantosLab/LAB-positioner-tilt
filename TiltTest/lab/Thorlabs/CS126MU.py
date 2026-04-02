

import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK
from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE

class CS126MU():  


    def __init__(self):
        self.exposure_time_us = 100
        self.poll_Timeout_ms = 2000

    def __del__(self):
        try:
            if hasattr(self, "sdk") and self.sdk is not None:
                self.sdk.dispose()
        except Exception:
            pass

    def setExposure(self, us=100):
        self.exposure_time_us = us
        return f"Exposure time has been set to {us} microseconds (min.28!)" 

    def pollTimeout(self, ms = 2000):
        self.poll_Timeout_ms = ms
        return f"Image Poll Timeout has been set to {ms} milliseconds"

    def takeImage(self, NUMBER_OF_IMAGES = 1):

        """Takes n images and returns Data"""
        image_data = []

        with TLCameraSDK() as sdk:
            cameras = sdk.discover_available_cameras()
            if len(cameras) == 0:
                print("Error: no cameras detected!")

            with sdk.open_camera(cameras[0]) as camera:
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

            with sdk.open_camera(cameras[0]) as camera:
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