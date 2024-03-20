import logging

import click
import numpy as np
import os

from .. import reduction
from .observatory import Observatory

logger = logging.getLogger(__name__)

"""
TODO: make flag that allows user make masters or not
- write docstrings
"""


@click.command(
    epilog="""Check out the documentation at
                https://pyscope.readthedocs.io/ for more
                information."""
)
@click.option(
    "-o",
    "--observatory",
    type=click.Path(exists=True),
    default="./config/observatory.cfg",
    show_default=True,
    help="Path to observatory configuration file.",
)
@click.option(
    "-c",
    "--camera",
    type=click.Choice(["ccd", "cmos"]),
    default="ccd",
    show_default=True,
    help="Camera type.",
)
@click.option(
    "-d",
    "--dark-exposures",
    type=float,
    multiple=True,
    default=[0.1, 1, 10, 100],
    show_default=True,
    help="Dark exposure times.",
)
@click.option(
    "-f",
    "--filter-exposures",
    type=float,
    multiple=True,
    help="The flat exposure times for each filter.",
)
@click.option(
    "-i",
    "--filter-brightness",
    type=float,
    multiple=True,
    help="The intensity of the calibrator [if present] for each filter.",
)
@click.option(
    "-r",
    "--readouts",
    type=int,
    multiple=True,
    default=[0],
    show_default=True,
    help="Readout modes to iterate through.",
)
@click.option(
    "-b",
    "--binnings",
    type=str,
    multiple=True,
    default=["1x1"],
    show_default=True,
    help="Binnings to iterate through.",
)
@click.option(
    "-n",
    "--repeat",
    type=int,
    default=10,
    show_default=True,
    help="Number of times to repeat each exposure.",
)
@click.option(
    "-s",
    "--save-path",
    type=click.Path(exists=True),
    default="./images/",
    show_default=True,
    help="Path to save calibration set.",
)
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["0", "1"]),
    default=0,
    show_default=True,
    help="Mode to use for averaging FITS files (0 = median, 1 = mean).",
)
@click.option(
    "-M",
    "--master",
    type=bool,
    default=True,
    show_default=True,
    help="Create master calibration files.",
)

@click.version_option()
def collect_calibration_set_cli(
    observatory,
    camera,
    dark_exposures,
    filter_exposures,
    filter_brightness,
    readouts=[0],
    binnings=["1x1"],
    repeat=10,
    save_path="./",
    mode=0,
    master=True,
):
    """
    Collects a calibration set for the observatory.

    .. warning::
        The dark_exposures, filter_exposures, and filter_brightnesses must be of equal length.

    Parameters
    ----------
    observatory : str

    camera : str, default="ccd"

    dark_exposures : list, default=[0.1, 1, 10, 100]

    filter_exposures : list

    filter_brightness : list

    redouts : list, default=[0]

    binnings : list, default=["1x1"]

    repeat : int, default=10

    save_path : str, default="./"
        Location to save the file

    mode : int, default=0

    Returns
    -------
    None

    """
    if type(observatory) == str:
        logger.info(f"Collecting calibration set for {observatory}")
        print(f"Collecting calibration set for {observatory}")
        obs = Observatory(observatory)
    elif type(observatory) == Observatory:
        logger.info(f"Collecting calibration set for {observatory.site_name}")
        print(f"Collecting calibration set for {observatory}")
        obs = observatory
    else:
        logger.exception(f"Invalid observatory type: {type(observatory)}")
        print(f"Invalid observatory type: {type(observatory)}")
        return False

    # if type(binnings) == str:
    #     binnings = binnings.split(",")
    # binnings = [tuple(map(int, b.split("x"))) for b in binnings]

    obs.connect_all()

    save_folder = os.path.join(
        save_path,
        "calibration_set_%s" % obs.observatory_time.strftime("%Y-%m-%d_%H-%M-%S"),
    )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    logger.info(f"Saving calibration set to {save_folder}")
    print(f"Saving calibration set to {save_folder}")

    logger.info("Collecting flats")
    print("Collecting flats")
    success = obs.take_flats(
        filter_exposures,
        filter_brightness=filter_brightness,
        readouts=readouts,
        binnings=binnings,
        repeat=repeat,
        save_path=save_folder,
        new_folder="flats",
    )
    if not success:
        logger.error("Failed to collect flats")
        print("failed to collect flats")
        return False

    if camera == "cmos":
        logger.info("Collecting flat-darks")
        print("Collecting flat-darks")
        success = obs.take_darks(
            np.unique(filter_exposures),
            readouts=readouts,
            binnings=binnings,
            repeat=repeat,
            save_path=save_folder,
            new_folder="flat_darks",
        )
        if not success:
            logger.error("Failed to collect flat-darks")
            print("failed to collect flat-darks")
            return False

    logger.info("Collecting darks")
    print("Collecting darks")
    success = obs.take_darks(
        dark_exposures,
        readouts=readouts,
        binnings=binnings,
        repeat=repeat,
        save_path=save_folder,
        new_folder="darks",
    )
    if not success:
        logger.error("Failed to collect darks")
        print("failed to collect darks")
        return False

    if camera == "ccd":
        logger.info("Collecting biases")
        print("Collecting biases")
        success = obs.take_darks(
            [0],
            readouts=readouts,
            binnings=binnings,
            repeat=repeat,
            save_path=save_folder,
            new_folder="biases",
        )
        if not success:
            logger.error("Failed to collect biases")
            print("failed to collect biases")
            return False
    elif camera == "cmos":
        logger.warning("Skipping bias collection for CMOS camera")
        print("Skipping bias collection for CMOS camera")

    logger.info("Collection complete.")
    print("Collection complete.")

    if master == True:
        logger.info("Creating master directory...")
        print("Creating master directory...")
        os.makedirs(os.path.join(save_folder, "masters"))
        for readout in readouts:
            logger.debug("Readout: %s" % readout)
            for binning in binnings:
                logger.debug("Binning: %ix%i" % (int(binning[0]), int(binning[2])))
                print(f"Binning: {int(binning[0])}x{int(binning[2])}")
                logger.info("Creating master darks...")
                print("Creating master darks...")
                for exposure in dark_exposures:
                    dark_paths = []
                    for i in range(repeat):
                        dark_paths.append(
                            # os.path.join(
                            #     os.path.join(save_folder, "darks"),
                            #     (
                            #         # "dark_%s_%ix%i_%4.4gs__%i.fts"
                            #         # % (readout, int(binning[0]), int(binning[2]), exposure, i)
                            #         f"dark_{readout}_{int(binning[0])}x{int(binning[2])}_{exposure}__{i}.fts"
                            #     ),
                            # )
                            
                            save_folder + "/darks/" + f"dark_{int(binning[0])}x{int(binning[2])}_{exposure}s_{readout}__{i}.fts"
                        )

                    reduction.avg_fits(
                        mode=mode,
                        outfile=os.path.join(
                            os.path.join(save_folder, "masters"),
                            (
                                # "master_dark_%s_%ix%i_%4.4gs.fts"
                                # % (readout, int(binning[0]), int(binning[2]), exposure)
                                f"master_dark_{readout}_{int(binning[0])}x{int(binning[2])}_{exposure}s.fts"
                            ),
                        ),
                        fnames=dark_paths,
                    )

                if camera == "ccd":
                    logger.info("Creating master biases...")
                    print("Creating master biases...")
                    bias_paths = []
                    for i in range(repeat):
                        bias_paths.append(
                            # os.path.join(
                            #     os.path.join(save_folder, "biases"),
                            #     (
                            #         "bias_%s_%ix%i__%i.fts"
                            #         % (readout, int(binning[0]), int(binning[2]), i)
                            #     ),
                            # )
                            save_folder + "/biases/" + f"dark_{int(binning[0])}x{int(binning[2])}_0s_{readout}__{i}.fts"
                        )

                    reduction.avg_fits(
                        mode=mode,
                        outfile=os.path.join(
                            os.path.join(save_folder, "masters"),
                            (
                                # "master_bias_%s_%ix%i.fts"
                                # % (readout, int(binning[0]), int(binning[2]))
                                f"master_bias_{int(binning[0])}x{int(binning[2])}_{readout}.fts"
                            ),
                        ),
                        fnames=bias_paths,
                    )

                logger.info("Creating master flats...")
                print("Creating master flats...")
                # for filt, exposure, brightness in zip(obs.filters, filter_exposures, filter_brightness):  
                for filt, exposure in zip(obs.filters, filter_exposures):
                    flat_paths = []
                    for i in range(repeat):
                        flat_paths.append(
                            # os.path.join(
                            #     os.path.join(save_folder, "flats"),
                            #     (
                            #         "flat_%s_%s_%ix%i_%4.4gs__%i.fts"
                            #         % (filt, readout, int(binning[0]), int(binning[2]), exposure, i)
                            #     ),
                            # )
                            save_folder + "/flats/" + f"flat_{filt}_{int(binning[0])}x{int(binning[2])}_{exposure}s_{readout}__{i}.fts"
                        )

                    reduction.avg_fits(
                        mode=mode,
                        outfile=os.path.join(
                            os.path.join(save_folder, "masters"),
                            (
                                # "master_flat_%s_%ix%i_%4.4gs.fts"
                                # % (readout, int(binning[0]), int(binning[2]), exposure)
                                f"master_flat_{filt}_{int(binning[0])}x{int(binning[2])}_{exposure}s.fts"
                            ),
                        ),
                        fnames=flat_paths,
                    )

                    if camera == "cmos":
                        logger.info("CMOS camera selected, creating master flat-dark...")
                        print("CMOS camera selected, creating master flat-dark...")
                        flat_dark_paths = []
                        for i in range(repeat):
                            flat_dark_paths.append(
                                # os.path.join(
                                #     os.path.join(save_folder, "flat_darks"),
                                #     (
                                #         "dark_%s_%ix%i_%4.4gs__%i.fts"
                                #         % (readout, int(binning[0]), int(binning[2]), exposure, i)
                                #     ),
                                # )
                                save_folder + "/flat_darks/" + f"dark_{int(binning[0])}x{int(binning[2])}_{exposure}s_{readout}__{i}.fts"
                            )

                        reduction.avg_fits(
                            mode=mode,
                            outfile=os.path.join(
                                os.path.join(save_folder, "masters"),
                                (
                                    # "master_flat_dark_%s_%ix%i_%4.4gs.fts"
                                    # % (readout, int(binning[0]), int(binning[2]), exposure)
                                    f"master_flat_dark_{int(binning[0])}x{int(binning[2])}_{exposure}s.fts"
                                ),
                            ),
                            fnames=flat_dark_paths,
                        )
    else:
        logger.info("Skipping master creation")
        print("Skipping master creation")
    logger.info("Calibration set complete.")
    print("Calibration set complete.")


collect_calibration_set = collect_calibration_set_cli.callback
