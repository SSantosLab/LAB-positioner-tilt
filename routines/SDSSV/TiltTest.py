import posUtils.data.dir
import posUtils.positioners.SDSSVChain
import posUtils.lab.camera

import TiltTest.config.ConfigFile
import TiltTest.analysis.spotfinding

import numpy as np
import pandas as pd

import pathlib
import datetime
import dataclasses

LOG_CAT = 'SDSS-V-TiltTest'

def RunTiltTest(config):
    """
    Run a full test routine with the SDSS-V positioner.

    Routine overview:
    - Iterate over a list of alpha values
        - Goto alpha position
        - Iterate over a list of beta values
            - Goto beta position
            - Take an image
            - Run online analysis
            - If requested: save the raw image
    - Save all the online generated data
    """

    BASE_DATA_override = config.getDefault('data:base', None)
    if BASE_DATA_override is not None:
        posUtils.data.dir.DataDir.BASE_DATA = pathlib.Path(BASE_DATA_override)

    data_dir = posUtils.data.dir.DataDir.getDir("sdss", "tilt")


    test_name = config.getDefault('meta:name', 'SDSSVBenchtopTest')
    run_dir = data_dir.createRunDir(test_name)
    logger = run_dir.createLogger(
        debug=config.getDefault('data:debug', False)
    )

    if BASE_DATA_override is not None:
        logger.warning(LOG_CAT, f'BASE_DATA overridden to: {BASE_DATA_override}')

    # Save the config file
    run_dir.saveConfig(config.dict())


    # Retrieve values from the config
    # Check now for values to exist, before initializing hardware
    SAVE_IMAGES = config.get('data:saveimages')

    EXPOSURE = config.get('camera:exposure')
    N_STACK = config.get('camera:stack')

    THRESHOLD = config.get('tilt:spot:threshold')
    CLIP = config.get('tilt:spot:clip')

    ALPHA_VALUES = config.get('tilt:positioner:alpha')
    BETA_VALUES = config.get('tilt:positioner:beta')


    # Initialize the Camera
    serial_number = config.get('camera:serial')
    camera = posUtils.lab.camera.Thorlabs_CS126MU(
        serial_number=serial_number, logger=logger
    )
    try:
        camera.setExposureTime(EXPOSURE)

        # Initialize the positioner (DUT)
        positioner_port = config.get('positioner:can:tty')
        sdssv = posUtils.positioners.SDSSVChain.SDSSV1(
            port=positioner_port, logger=logger,
        )

        # Init the spot finding algorithm
        spotfinder = TiltTest.analysis.spotfinding.ThresholdSpotFinder(
            threshold= THRESHOLD, clip=CLIP
        )

        output_data = []

        ## Run the Test Routine
        logger.info(LOG_CAT, 'Start test routine')
        for alpha in ALPHA_VALUES:
            logger.info(LOG_CAT, f'Beta circle at alpha={alpha:.3f}')

            for beta in BETA_VALUES:
                logger.info(LOG_CAT, f'Beta={beta:.3f}')
                sdssv.pos1.goto_absolute(alpha, beta)
                sdssv.pos1.wait_move()


                logger.debug(LOG_CAT, f'Take [{N_STACK}] images')
                images = camera.takeImages(N_STACK)
                image_sum = np.mean(list(images.values()), axis=0)

                # Apply spot finding
                measurement = spotfinder.find(image_sum)
                logger.debug(LOG_CAT, f'Spotfinding result: {measurement}')


                if SAVE_IMAGES:
                    image_file = f'Spot_Stack{N_STACK}_{alpha:.3f}_{beta:.3f}.npz'
                    logger.debug(LOG_CAT, f'Save raw image to [{image_file}]')
                    np.savez_compressed(
                        run_dir.data / image_file,
                        image_sum
                    )
                else:
                    image_file = None


                output_data.append({
                    'time': datetime.datetime.now(),
                    'image:file': image_file,
                    'pos:alpha': alpha,
                    'pos:beta': beta,
                    'camera:stack': N_STACK,
                    'camera:exposure': EXPOSURE,
                    'spot:threshold': TRESHOLD,
                    **{
                        f'spot:{key}': value
                            for key, value in dataclasses.asdict(measurement).items()
                    }
                })


        run_dir.saveResultDf(pd.DataFrame(output_data))

    finally:
        camera.close()


    # TODO: Do some online analysis



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Routine to run a Tilt-Test Acquisition with an SDSS-V positioner'
    )

    parser.add_argument('config', help='Config file name for this test.')
    args = parser.parse_args()

    config = TiltTest.config.ConfigFile(args.config)

    RunTiltTest(config)
