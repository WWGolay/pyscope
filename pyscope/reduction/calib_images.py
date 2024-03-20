import datetime
import glob
import logging
import os
import shutil
import pathlib

# i will be working on this
import click
from astropy.io import fits

from ..analysis import calc_zmag
from ..observatory import AstrometryNetWCS
from .ccd_calib import ccd_calib

logger = logging.getLogger(__name__)


@click.command(
    epilog="""Check out the documentation at
                https://pyscope.readthedocs.io/ for more
                information."""
)
@click.option(
    "-t",
    "--camera-type",
    type=click.Choice(["ccd", "cmos"]),
    default="ccd",
    show_choices=True,
    show_default=True,
    help="Camera type.",
)
@click.option(
    "-i",
    "--image-dir",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default="./images/",
    show_default=True,
    help="""Directory containing images to be calibrated. If passed, then
                    the fnames argument is ignored.""",
)
@click.option(
    "-c",
    "--calib-dir",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default="./images/",
    show_default=True,
    required=True,
    help="Location of all calibration files.",
)
@click.option(
    "-r",
    "--raw-archive-dir",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="""Directory to archive raw images. If none given, no archiving is done,
                however, the '--in-place' option is not allowed.""",
)
@click.option(
    "--in-place",
    is_flag=True,
    default=False,
    show_default=True,
    help="""If given, the raw images are overwritten with the calibrated images.
                If not given, the calibrated images are written to the same directory as the
                raw images, but with the suffix '_cal' added to the filename prior to the extension.""",
)
@click.option(
    "-s",
    "--astro-scrappy",
    nargs=2,
    default=(1, 3),
    show_default=True,
    help="Number of hot pixel removal iterations and estimated camera read noise.",
)
@click.option(
    "-b",
    "--bad-columns",
    default="",
    show_default=True,
    help="Comma-separated list of bad columns to fix.",
)
@click.option(
    "-w",
    "--wcs",
    is_flag=True,
    default=False,
    show_default=True,
    help="""If given, the WCS is solved for each image. If not given, the WCS is
                not solved.""",
)
@click.option(
    "-z",
    "--zmag",
    is_flag=True,
    default=False,
    show_default=True,
    help="""If given, the zero-point magnitude is calculated for each image with
                SDSS filters. If not given, the zero-point magnitude is not calculated.""",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    show_default=True,
    help="Verbose output.",
)
@click.argument(
    "fnames", nargs=-1, type=click.Path(exists=True, file_okay=True, resolve_path=True)
)
@click.version_option()
def calib_images_cli(
    camera_type="ccd",
    image_dir="./images/",
    calib_dir=None,
    raw_archive_dir="./images/raw/",
    in_place=False,
    astro_scrappy=(1, 3),
    bad_columns="",
    wcs=False,
    zmag=False,
    verbose=False,
    fnames=[],
):
    """Calibrate a set of images by recursively selecting the
    appropriate flat, dark, and bias frame and then calling
    ccd_calib to do the actual calibration.

    Notes: Vaving an example set up of this function would be helpful.
    This would allow for me to know what the environment looks like when
    calling this function.

    Args:
        camera_type (_type_): _description_
        image_dir (_type_): _description_
        calib_dir (_type_): _description_
        raw_archive_dir (_type_): _description_
        in_place (_type_): _description_
        astro_scrappy (_type_): _description_
        bad_columns (_type_): _description_
        wcs (_type_): _description_
        zmag (_type_): _description_
        verbose (_type_): _description_
        fnames (_type_): _description_

    Raises:
        click.BadParameter: _description_
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    if raw_archive_dir is None and in_place:
        raise click.BadParameter(
            "The --in-place option is not allowed if no raw archive directory is given."
        )

    if image_dir is not None:
        logger.info(
            "--image-dir passed, ignoring fnames argument and using all images in directory."
        )
        fnames = []
        for ext in (".fts", ".fits", ".fit"):
            fnames.extend(glob.glob(f"{image_dir}/*{ext}"))
        logger.debug(f"fnames = {fnames}")

    logger.info(f"Calibrating {len(fnames)} images.")
    for fname in fnames:
        logger.info(f"Calibrating {fname}")

        if raw_archive_dir is not None:
            raw_archive_dir = os.path.join(
                raw_archive_dir, datetime.datetime.now().strftime("%Y-%m-%d")
            )
            if not os.path.exists(raw_archive_dir):
                logger.info(f"Creating raw archive directory: {raw_archive_dir}")
                os.makedirs(raw_archive_dir)
            logger.info(f"Archiving {fname} to {raw_archive_dir}")
            shutil.copy(fname, raw_archive_dir)

        with fits.open(fname) as hdu:
            hdr = hdu[0].header

        if "CALSTAT" in hdr.keys():
            logger.info(f"{fname} has already been calibrated, skipping.")
            continue

        try:
            filt = hdr["FILTER"]
        except KeyError:
            filt = ""

        try:
            readout = hdr["READOUTM"].replace(" ", "")
        except KeyError:
            readout = hdr["READOUT"].replace(" ", "")


        try:
            exptime = round(hdr["EXPTIME"], 3)
        except KeyError:
            exptime = round(hdr["EXPOSURE"], 3)

        try:
            xbin = hdr["XBINNING"]
        except:
            xbin = hdr["XBIN"]

        try:
            ybin = hdr["YBINNING"]
        except:
            ybin = hdr["YBIN"]


        observatory_home = os.getenv("observatory_home")
        masters_dir = pathlib.Path(observatory_home + "/images/masters")
        flat_frame = None
        dark_frame = None
        bias_frame = None
        flat_dark_frame = None
        if calib_dir is None:
            files = glob.glob(f"{masters_dir}/*", recursive=True)
            for filename in files:
                # find flat_frame, dark_frame, bias_frame, flat_dark_frame
                with fits.open(filename) as h:
                    hdrf = h[0].header
                    # if the corresponding header values match, then set the appropriate frame
                    if (
                        "master_flat" in filename and "master_flat_dark" not in filename
                        and hdrf["FILTER"] == filt
                        and hdrf["READOUTM"] == readout
                        and hdrf["EXPTIME"] == exptime
                        and hdrf["XBINNING"] == xbin
                        and hdrf["YBINNING"] == ybin
                    ):
                        flat_frame = pathlib.Path(filename)
                    elif (
                        "master_dark" in filename
                        and hdrf["READOUTM"] == readout
                        and hdrf["EXPTIME"] == exptime
                        and hdrf["XBINNING"] == xbin
                        and hdrf["YBINNING"] == ybin
                        ):
                        dark_frame = pathlib.Path(filename)
                    elif (
                        "master_bias" in filename
                        and hdrf["READOUTM"] == readout
                        and hdrf["XBINNING"] == xbin
                        and hdrf["YBINNING"] == ybin
                    ):
                        bias_frame = pathlib.Path(filename)
                    elif (
                        "master_flat_dark" in filename
                        and hdrf["READOUTM"] == readout
                        and hdrf["EXPTIME"] == exptime
                        and hdrf["XBINNING"] == xbin
                        and hdrf["YBINNING"] == ybin
                    ):
                        flat_dark_frame = pathlib.Path(filename)
                h.close()

            # given a camera type, set the appropriate frame; bias for CCD, flat dark for CMOS
            if camera_type == "ccd":
                bias_frame = bias_frame
            elif camera_type == "cmos":
                flat_dark_frame = flat_dark_frame

        else:  # use the calib_dir passed by the user
            flat_frame = (
                pathlib.Path(f"{calib_dir}/master_flat_{filt}_{readout}_{exptime}_{xbin}x{ybin}.fts")
            )
            dark_frame = (
                pathlib.Path(f"{calib_dir}/master_dark_{readout}_{exptime}_{xbin}x{ybin}.fts")
            )

            # given a camera type, set the appropriate frame; bias for CCD, flat dark for CMOS
            if camera_type == "ccd":
                bias_frame = pathlib.Path(f"{calib_dir}/master_bias_{readout}_{xbin}x{ybin}.fts")
            elif camera_type == "cmos":
                flat_dark_frame = pathlib.Path(f"{calib_dir}/master_flat_dark_{readout}_{exptime}_{xbin}x{ybin}.fts")

        # for each image, print out the calibration frames being used
        logger.debug("Using calibration frames:")
        logger.debug(f"Flat: {flat_frame}")
        logger.debug(f"Dark: {dark_frame}")
        if camera_type == "ccd":
            logger.debug(f"Bias: {bias_frame}")
        elif camera_type == "cmos":
            logger.debug(f"Flat dark: {flat_dark_frame}")

        # After gethering all the required parameters, run ccd_calib
        logger.debug("Running ccd_calib...")
        ccd_calib(
            camera_type=camera_type,
            flat_frame=flat_frame,
            dark_frame=dark_frame,
            bias_frame=bias_frame,
            flat_dark_frame=flat_dark_frame,
            astro_scrappy=astro_scrappy,
            bad_columns=bad_columns,
            in_place=in_place,
            verbose=verbose,
            fnames=[fname],
        )

        # world coordinate system
        if wcs:
            logger.debug("Running Astrometry.net WCS solver...")
            solver = AstrometryNetWCS()
            solver.solve(fname)

        logger.debug("Done!")

    # outside for loop
    if zmag:
        logger.info("Calculating zero-point magnitudes...")
        calc_zmag(fnames=fnames)

    logger.info("Done!")


calib_images = calib_images_cli.callback
