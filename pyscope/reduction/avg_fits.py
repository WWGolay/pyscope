import logging

import click
import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

"""
TODO: use ccdproc to average FITS files
- look into ccdproc library
- write tests for avg_fits
- make a new file avg_fits_ccdproc.py
"""


@click.command(
    epilog="""Check out the documentation at
                https://pyscope.readthedocs.io/ for more
                information."""
)
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["0", "1"]),
    default="0",
    show_choices=True,
    show_default=True,
    help="Mode to use for averaging FITS files (0 = median, 1 = mean).",
)
@click.option(
    "-o",
    "--outfile",
    type=click.Path(),
    required=True,
    help="Path to save averaged FITS file.",
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
def avg_fits_cli(mode, outfile, fnames, verbose=False):
    """
    Test
    asdf
    """

    if verbose:
        logger.setLevel(logging.DEBUG)

    logger.debug(f"avg_fits(mode={mode}, outfile={outfile}, fnames={fnames})")

    logger.info("Loading FITS files...")
    print(fnames)
    images = np.array([fits.open(fname)[0].data for fname in fnames])
    images = images.astype(np.float32)
    logger.info(f"Loaded {len(images)} FITS files")

    logger.info("Averaging FITS files...")
    print(f"mode: {mode}, {type(mode)}")
    if str(mode) == "0":
        logger.debug("Calculating median...")
        image_avg = np.median(images, axis=0)
    elif str(mode) == "1":
        logger.debug("Calculating mean...")
        image_avg = np.mean(images, axis=0)

    logger.debug(f"Image mean: {np.mean(image_avg)}")
    logger.debug(f"Image median: {np.median(image_avg)}")

    image_avg = image_avg.astype(np.uint16)

    logger.info(f"Saving averaged FITS file to {outfile}")

    with fits.open(fnames[-1]) as hdul:
        hdr = hdul[0].header

    hdr.add_comment(f"Averaged {len(images)} FITS files using pyscope")
    hdr.add_comment(f"Average mode: {mode}")
    fits.writeto(outfile, image_avg, hdr, overwrite=True)

    logger.info("Done!")


avg_fits = avg_fits_cli.callback
