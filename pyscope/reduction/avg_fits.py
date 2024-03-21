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
    "-d",
    "--datatype",
    type=click.Choice(["int", "int_", "intc",
                        "intp", "int8", "int16",
                        "int32", "int64", "int128",
                        "int256",
                        
                        "uint", "uintc", "uintp",
                        "uint8", "uint16", "uint32",
                        "uint64", "uint128", "uint256",
                        
                        "float", "float_", "float32",
                        "float64", "float80", "float96",
                        "float128", "float256",
                        
                        "double", "longdouble", "longfloat",
                        "longlong", "ulonglong"]),
    default="uint16",
    show_choices=True,
    show_default=True,
    help="Data type to use for averaged FITS file.",
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
def avg_fits_cli(mode, outfile, fnames, datatype='uint16', verbose=False):
    """Averages FITS files.

    Parameters
    ----------
    mode : str, default="0"
        Mode to use for averaging FITS files (0 = median, 1 = mean).
    
    outfile : str
        Path to save averaged FITS files.
    
    fnames : list
        List of FITS file paths to average.
    
    verbose : bool, default=False
        Print verbose output.

    Returns
    -------
    None
    """
    
    if verbose:
        logger.setLevel(logging.DEBUG)

    logger.debug(f"avg_fits(mode={mode}, outfile={outfile}, fnames={fnames})")

    logger.info("Loading FITS files...")
    print(fnames)
    images = np.array([fits.open(fname)[0].data for fname in fnames])
    
    match datatype:
        case "int" | "int_":
            dt = np.int_
        case "intc":
            dt = np.intc
        case "intp":
            dt = np.intp
        case "int8":
            dt = np.int8
        case "int16":
            dt = np.int16
        case "int32":
            dt = np.int32
        case "int64":
            dt = np.int64
        case "int128":
            dt = np.int128
        case "int256":
            dt = np.int256

        case "uint":
            dt = np.uint
        case "uintc":
            dt = np.uintc
        case "uintp":
            dt = np.uintp
        case "uint8":
            dt = np.uint8
        case "uint16":
            dt = np.uint16
        case "uint32":
            dt = np.uint32
        case "uint64":
            dt = np.uint64
        case "uint128":
            dt = np.uint128
        case "uint256":
            dt = np.uint256

        case "float" | "float_":
            dt = np.float_
        case "float32":
            dt = np.float32
        case "float64":
            dt = np.float64
        case "float80":
            dt = np.float80
        case "float96":
            dt = np.float96
        case "float128":
            dt = np.float128
        case "float256":
            dt = np.float256
    
        case "double":
            dt = np.double
        case "longdouble":
            dt = np.longdouble
        case "longfloat":
            dt = np.longfloat
        case "longlong":
            dt = np.longlong
        case "ulonglong":
            dt = np.ulonglong
        
        case _:
            raise ValueError(f"Invalid datatype: {datatype}")
    
    images = images.astype(dt)
    
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

    image_avg = image_avg.astype(dt)

    print(datatype)


    logger.info(f"Saving averaged FITS file to {outfile}")

    with fits.open(fnames[-1]) as hdul:
        hdr = hdul[0].header

    hdr.add_comment(f"Averaged {len(images)} FITS files using pyscope")
    hdr.add_comment(f"Average mode: {mode}")
    fits.writeto(outfile, image_avg, hdr, overwrite=True)

    logger.info("Done!")


avg_fits = avg_fits_cli.callback
