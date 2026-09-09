import posUtils.data.dir
import posUtils.lab.camera

import TiltTest.config.ConfigFile
import TiltTest.analysis.spotfinding

import numpy as np
import pandas as pd

import pathlib
import datetime
import dataclasses

LOG_CAT = 'Manual-ZeroAngle'


def RunZeroAngle(config):
    """
    Manual variant of the tilt test: the calibration cylinder is rotated by
    hand about its single axis instead of the positioner being driven.

    Routine overview:
    - Loop until the operator ends it
        - Wait for the operator to rotate the cylinder and confirm
        - Take an image
        - Run online analysis
        - If requested: save the raw image
    - Save all the online generated data
    - Fit an ellipse (the imaged circle, seen slightly off-normal) to the
      spot centers and plot it
    """

    BASE_DATA_override = config.getDefault('data:base', None)
    if BASE_DATA_override is not None:
        posUtils.data.dir.DataDir.BASE_DATA = pathlib.Path(BASE_DATA_override)

    data_dir = posUtils.data.dir.DataDir.getDir("calib", "tilt")


    test_name = config.getDefault('meta:name', 'ManualZeroAngle')
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


    # Initialize the Camera
    serial_number = config.get('camera:serial')
    camera = posUtils.lab.camera.Thorlabs_CS126MU(
        serial_number=serial_number, logger=logger
    )
    try:
        camera.setExposureTime(EXPOSURE)

        # Init the spot finding algorithm
        spotfinder = TiltTest.analysis.spotfinding.ThresholdSpotFinder(
            threshold= THRESHOLD, clip=CLIP
        )

        output_data = []

        ## Run the Test Routine
        logger.info(LOG_CAT, 'Start test routine')
        logger.info(LOG_CAT, 'Rotate the cylinder and press <Enter> to capture, "q" to finish')

        index = 0
        while True:
            answer = input(f'[{index}] capture (<Enter>) / finish (q): ').strip()
            if answer.lower() in ('q', 'quit', 'exit'):
                break

            logger.info(LOG_CAT, f'Capture index={index}')

            logger.debug(LOG_CAT, f'Take [{N_STACK}] images')
            images = camera.takeImages(N_STACK)
            image_sum = np.mean(list(images.values()), axis=0)

            # Apply spot finding
            measurement = spotfinder.find(image_sum)
            logger.debug(LOG_CAT, f'Spotfinding result: {measurement}')


            if SAVE_IMAGES:
                image_file = f'Spot_Stack{N_STACK}_{index:03d}.npz'
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
                'index': index,
                'camera:stack': N_STACK,
                'camera:exposure': EXPOSURE,
                'spot:threshold': THRESHOLD,
                **{
                    f'spot:{key}': value
                        for key, value in dataclasses.asdict(measurement).items()
                }
            })

            index += 1

        data_df = pd.DataFrame(output_data)
        run_dir.saveResultDf(data_df, 'ZeroAngleData')

    finally:
        camera.close()


    ## Fit the traced circle (imaged as an ellipse) and plot it
    #
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.scatter(
        data_df['spot:x0'], data_df['spot:y0'], marker='x',
        label='Spot centers'
    )

    try:
        # The cylinder point traces a circle; imaged slightly off-normal it
        # projects to an ellipse. Reuse the spotfinder's conic fit (temporary --
        # to be pulled into a proper analysis module later).
        # Note: _fit_ellipse throws on perfectly noise-free input; fine for real data.
        x0, y0, a_major, a_minor, theta = TiltTest.analysis.spotfinding._fit_ellipse(
            data_df['spot:x0'].to_numpy(dtype=float),
            data_df['spot:y0'].to_numpy(dtype=float),
        )
        radius = float(np.sqrt(a_major * a_minor))  # effective circle radius [pix]
        logger.info(
            LOG_CAT,
            f'Ellipse fit: center=({x0:.2f}, {y0:.2f}), '
            f'a={a_major:.2f} pix, b={a_minor:.2f} pix, '
            f'radius={radius:.2f} pix, theta={np.degrees(theta):.2f} deg'
        )

        t = np.linspace(0, 2 * np.pi, 256)
        ct, st = np.cos(theta), np.sin(theta)
        ex, ey = a_major * np.cos(t), a_minor * np.sin(t)
        ax.plot(
            x0 + ex * ct - ey * st, y0 + ex * st + ey * ct,
            '-', color='tab:red', label='Fitted ellipse',
        )

        ax.scatter(
            [x0], [y0], marker='+', s=120, color='tab:red',
            label=f'Center ({x0:.1f}, {y0:.1f})'
        )

        # Draw the semi-major axis and label both semi-axes
        ax.plot(
            [x0, x0 + a_major * ct], [y0, y0 + a_major * st],
            '--', color='tab:red',
        )
        ax.annotate(
            f'a = {a_major:.1f} pix\nb = {a_minor:.1f} pix',
            xy=(x0 + 0.5 * a_major * ct, y0 + 0.5 * a_major * st),
            xytext=(8, 8), textcoords='offset points',
        )
    except RuntimeError as e:
        print(e)

    ax.set_aspect('equal')
    ax.grid()
    ax.legend()

    ax.set_xlabel('X [pix]')
    ax.set_ylabel('Y [pix]')

    fig.tight_layout()
    run_dir.saveFig(fig, 'ZeroAngleCircle.pdf')



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Routine to run a manual Zero-Angle acquisition with a rotated calibration cylinder'
    )

    parser.add_argument('config', help='Config file name for this test.')
    args = parser.parse_args()

    config = TiltTest.config.ConfigFile(args.config)

    RunZeroAngle(config)
