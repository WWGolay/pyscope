import logging
import time

import astroscrappy
import click
import numpy as np
from astropy.io import fits

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
    "-d",
    "--dark-frame",
    type=click.Path(exists=True, resolve_path=True),
    help="Path to master dark frame.",
)
@click.option(
    "-f",
    "--flat-frame",
    type=click.Path(exists=True, resolve_path=True),
    help="Path to master flat frame.",
)
@click.option(
    "-b",
    "--bias-frame",
    type=click.Path(exists=True, resolve_path=True),
    help="Path to master bias frame. Ignored if camera type is CMOS.",
)
@click.option(
    "-a",
    "--flat-dark-frame",
    type=click.Path(exists=True, resolve_path=True),
    help="Path to master flat-dark frame. Ignored if camera type is CCD.",
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
    "-c",
    "--bad-columns",
    default="",
    show_default=True,
    help="Comma-separated list of bad columns to fix.",
)
@click.option(
    "-i",
    "--in-place",
    is_flag=True,
    default=False,
    show_default=True,
    help="Overwrite input files.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    show_default=True,
    help="Print verbose output.",
)
@click.argument("fnames", nargs=-1, type=click.Path(exists=True, resolve_path=True))
@click.version_option()
def ccd_calib_cli(
    camera_type,
    dark_frame,
    flat_frame,
    bias_frame,
    flat_dark_frame,
    astro_scrappy,
    bad_columns,
    in_place,
    verbose,
    fnames,
):
    if verbose:
        logger.setLevel(logging.DEBUG)

    logger.debug(
        f"""ccd_calib(camera_type={camera_type}, dark_frame={dark_frame},
    flat_frame={flat_frame}, bias_frame={bias_frame}, astro_scrappy={astro_scrappy},
    bad_columns={bad_columns}, in_place={in_place}, fnames={fnames})"""
    )

    if camera_type == "ccd":
        if bias_frame is not None:
            logger.info(f"Loading bias frame: {bias_frame}")
            with fits.open(bias_frame) as hdul:
                bias = hdul[0].data
                hdr = hdul[0].header
            bias = bias.astype(np.float32)

            try:
                bias_readout_mode = hdr["READOUTM"]
            except KeyError:
                bias_readout_mode = hdr["READOUT"]

            try:
                bias_exptime = round(hdr["EXPTIME"], 3)
            except KeyError:
                bias_exptime = round(hdr["EXPOSURE"], 3)

            try:
                bias_xbin = hdr["XBINNING"]
            except:
                bias_xbin = hdr["XBIN"]

            try:
                bias_ybin = hdr["YBINNING"]
            except:
                bias_ybin = hdr["YBIN"]

            logger.debug(f"Bias frame readout mode: {bias_readout_mode}")
            logger.debug(f"Bias frame exposure time: {bias_exptime}")
            logger.debug(f"Bias frame X binning: {bias_xbin}")
            logger.debug(f"Bias frame Y binning: {bias_ybin}")
            # logger.debug(f"Bias frame pedestal: {bias_pedestal}")
    elif camera_type == "cmos":
        if flat_dark_frame is not None:
            logger.info(f"Loading flat-dark frame: {flat_dark_frame}")
            with fits.open(flat_dark_frame) as hdul:
                flat_dark = hdul[0].data
                hdr = hdul[0].header
            flat_dark = flat_dark.astype(np.float32)

            try:
                flat_dark_filter = hdr["FILTER"]
            except KeyError:
                flat_dark_filter = ""

            try:
                flat_dark_readout_mode = hdr["READOUTM"]
            except KeyError:
                flat_dark_readout_mode = hdr["READOUT"]

            try:
                flat_dark_exptime = hdr["EXPTIME"]
            except KeyError:
                flat_dark_exptime = hdr["EXPOSURE"]

            try:
                flat_dark_xbin = hdr["XBINNING"]
            except:
                flat_dark_xbin = hdr["XBIN"]

            try:
                flat_dark_ybin = hdr["YBINNING"]
            except:
                flat_dark_ybin = hdr["YBIN"]

            logger.debug(f"Flat-dark frame readout mode: {flat_dark_readout_mode}")
            logger.debug(f"Flat-dark frame exposure time: {flat_dark_exptime}")
            logger.debug(f"Flat-dark frame X binning: {flat_dark_xbin}")
            logger.debug(f"Flat-dark frame Y binning: {flat_dark_ybin}")

    logger.info("Loading calibration frames...")
    if dark_frame is not None:
        logger.info(f"Loading dark frame: {dark_frame}")
        with fits.open(dark_frame) as hdul:
            dark = hdul[0].data
            hdr = hdul[0].header
        dark = dark.astype(np.float32)

        try:
            dark_readout_mode = hdr["READOUTM"]
        except KeyError:
            dark_readout_mode = hdr["READOUT"]

        try:
            dark_exptime = round(hdr["EXPTIME"], 3)
        except KeyError:
            dark_exptime = round(hdr["EXPOSURE"], 3)

        try:
            dark_xbin = hdr["XBINNING"]
        except:
            dark_xbin = hdr["XBIN"]

        try:
            dark_ybin = hdr["YBINNING"]
        except:
            dark_ybin = hdr["YBIN"]

        logger.debug(f"Dark frame readout mode: {dark_readout_mode}")
        logger.debug(f"Dark frame exposure time: {dark_exptime}")
        logger.debug(f"Dark frame X binning: {dark_xbin}")
        logger.debug(f"Dark frame Y binning: {dark_ybin}")

    if flat_frame is not None:
        logger.info(f"Loading flat frame: {flat_frame}")
        with fits.open(flat_frame) as hdul:
            flat = hdul[0].data
            hdr = hdul[0].header

        flat = flat.astype(np.float32)

        try:
            flat_filter = hdr["FILTER"]
        except KeyError:
            flat_filter = ""

        try:
            flat_readout_mode = hdr["READOUTM"]
        except KeyError:
            flat_readout_mode = hdr["READOUT"]

        try:
            flat_exptime = round(hdr["EXPTIME"], 3)
        except KeyError:
            flat_exptime = round(hdr["EXPOSURE"], 3)

        try:
            flat_xbin = hdr["XBINNING"]
        except:
            flat_xbin = hdr["XBIN"]

        try:
            flat_ybin = hdr["YBINNING"]
        except:
            flat_ybin = hdr["YBIN"]

        logger.debug(f"Flat frame readout mode: {flat_readout_mode}")
        logger.debug(f"Flat frame exposure time: {flat_exptime}")
        logger.debug(f"Flat frame X binning: {flat_xbin}")
        logger.debug(f"Flat frame Y binning: {flat_ybin}")

    logger.info("Looping through images...")
    for fname in fnames:
        with fits.open(fname) as hdul:
            image = hdul[0].data
            hdr = hdul[0].header

        image = image.astype(np.float32)

        try:
            image_filter = hdr["FILTER"]
        except KeyError:
            image_filter = ""

        try:
            image_readout_mode = hdr["READOUTM"].replace(" ", "")
        except KeyError:
            image_readout_mode = hdr["READOUT"].replace(" ", "")

        try:
            image_exptime = round(hdr["EXPTIME"], 3)
        except KeyError:
            image_exptime = round(hdr["EXPOSURE"], 3)

        try:
            image_xbin = hdr["XBINNING"]
        except:
            image_xbin = hdr["XBIN"]

        try:
            image_ybin = hdr["YBINNING"]
        except:
            image_ybin = hdr["YBIN"]

        logger.debug(f"Image readout mode: {image_readout_mode}")
        logger.debug(f"Image exposure time: {image_exptime}")
        logger.debug(f"Image X binning: {image_xbin}")
        logger.debug(f"Image Y binning: {image_ybin}")

        if image_readout_mode != dark_readout_mode:
            logger.warning(
                f"Image readout mode ({image_readout_mode}) does not match dark readout mode ({dark_readout_mode})"
            )

        if image_readout_mode != flat_readout_mode:
            logger.warning(
                f"Image readout mode ({image_readout_mode}) does not match flat readout mode ({flat_readout_mode})"
            )

        if image_xbin != dark_xbin:
            logger.warning(
                f"Image X binning ({image_xbin}) does not match dark X binning ({dark_xbin})"
            )

        if image_xbin != flat_xbin:
            logger.warning(
                f"Image X binning ({image_xbin}) does not match flat X binning ({flat_xbin})"
            )

        if image_ybin != dark_ybin:
            logger.warning(
                f"Image Y binning ({image_ybin}) does not match dark Y binning ({dark_ybin})"
            )

        if image_ybin != flat_ybin:
            logger.warning(
                f"Image Y binning ({image_ybin}) does not match flat Y binning ({flat_ybin})"
            )

        if image_filter != flat_filter:
            logger.warning(
                f"Image filter ({image_filter}) does not match flat filter ({flat_filter})"
            )

        if camera_type == "cmos":
            if image_exptime != dark_exptime:
                logger.warning(
                    f"""Image exposure time ({image_exptime}) does not match dark exposure time ({dark_exptime}),
                    recommended for a CMOS camera"""
                )
            if flat_exptime != flat_dark_exptime:
                logger.warning(
                    f"""Flat exposure time ({flat_exptime}) does not match flat-dark exposure time ({flat_dark_exptime}),
                    recommended for a CMOS camera"""
                )

        hdr.add_comment(f"Calibrated using pyscope")
        hdr.add_comment(f"Calibration mode: {camera_type}")
        hdr.add_comment(f"Calibration dark frame: {dark_frame}")
        hdr.add_comment(f"Calibration flat frame: {flat_frame}")
        hdr.add_comment(f"Calibration bias frame: {bias_frame}")
        hdr.add_comment(f"Calibration flat-dark frame: {flat_dark_frame}")
        hdr.add_comment(f"Calibration astro-scrappy: {astro_scrappy}")
        hdr.add_comment(f"Calibration bad columns: {bad_columns}")

        cal_image = image.copy()

        if camera_type == "ccd":
            if bias_frame is not None:
                logger.info("Applying bias frame (CCD selected)...")
                cal_image -= bias

            if dark_frame is not None:
                logger.info(
                    """Applying the dark frame. CCD selected so a bias-subtracted
                            dark frame is scaled by the ratio of the image exposure time over
                            the dark exposure time then subtracted from the image."""
                )

                cal_image -= (dark - bias) * (image_exptime / dark_exptime)
        elif camera_type == "cmos":
            if flat_dark_frame is not None:
                logger.info(
                    """Applying the flat-dark frame. CMOS selected so a dark
                            is subtracted from the image."""
                )
                cal_image -= dark

        if flat_frame is not None:
            logger.info("""Prepping the flat frame...""")

            if camera_type == "ccd":
                logger.debug(
                    """CCD selected so the flat frame is
                        bias-subtracted then a dark frame scaled in the same way as above
                        is subtracted from the flat frame, if those frames are present."""
                )
                if bias_frame is not None:
                    flat -= bias

                if dark_frame is not None:
                    flat -= (dark - bias) * (flat_exptime / dark_exptime)

            elif camera_type == "cmos":
                logger.debug(
                    """CMOS selected so the flat frame is simply
                        dark-flat subtracted if present."""
                )
                if flat_dark_frame is not None:
                    flat -= flat_dark

            logger.info(
                """Normalizing the flat frame by the mean of the
                        middle quarter of the image."""
            )
            x_size = flat.shape[1]
            y_size = flat.shape[0]
            x_start = int(x_size / 4)
            x_end = int(3 * x_size / 4)
            y_start = int(y_size / 4)
            y_end = int(3 * y_size / 4)

            flat_mean = np.mean(flat[y_start:y_end, x_start:x_end])
            flat /= flat_mean

            logger.debug("Clipping negative values to 1...")
            flat[flat <= 0] = 1

            logger.info("Applying the flat frame...")
            cal_image /= flat

        logger.info("Flooring the calibrated image...")
        cal_image = np.floor(cal_image)

        logger.info("Checking if pedestal is necessary...")
        if np.min(cal_image) < 0:
            logger.info(
                "Minimum value is %i, adding pedestal of equal value"
                % np.min(cal_image)
            )
            cal_image += np.abs(np.min(cal_image))
            hdr["PEDESTAL"] = np.abs(np.min(cal_image))

        if astro_scrappy[0] > 0:
            logger.info("Removing hot pixels...")
            t0 = time.time()
            mask, cal_image = astroscrappy.detect_cosmics(
                cal_image, niter=astro_scrappy[0], readnoise=astro_scrappy[1]
            )
            t = time.time() - t0
            hdr.add_comment(
                f"Removed hot pixels using astroscrappy, {astro_scrappy[0]} iterations"
            )
            hdr.add_comment("Hot pixel removal took %.1f seconds" % t)
            logger.debug(f"Hot pixel removal took {t} seconds")

        if len(bad_columns) > 0:
            logger.info("Fixing bad columns...")
            for badcol in bad_columns:
                cal_image[:, badcol] = (
                    cal_image[:, badcol - 1] + cal_image[:, badcol + 1]
                ) / 2

        logger.info("Clipping to uint16 range...")
        cal_image = np.clip(cal_image, 0, 65535)
        cal_image = cal_image.astype(np.uint16)

        logger.info("Writing calibrated status to header...")
        hdr["CALSTAT"] = True
        if in_place:
            logger.info(f"Overwriting {fname}")
            fits.writeto(fname, cal_image, hdr, overwrite=True)
        else:
            logger.info(f"Writing calibrated image to {fname}")
            fits.writeto(
                fname.split(".")[:-1][0] + "_cal.fts", cal_image, hdr, overwrite=True
            )

        logger.info("Done!")


ccd_calib = ccd_calib_cli.callback
