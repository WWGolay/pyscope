import logging

import click
import markdown
import numpy as np
from astropy import coordinates as coord
from astropy import table
from astropy import time as astrotime
from astropy.io import fits

from ..reduction import fitslist

logger = logging.getLogger(__name__)


@click.command(
    epilog="""Check out the documentation at
               https://pyscope.readthedocs.io/en/latest/
               for more information."""
)
@click.option(
    "-s",
    "--schedule",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="""Path to the schedule file. If none provided, the default is in
    the ./schedules/ directory, find the latest schedule.""",
)
@click.option(
    "-i",
    "--images-dir",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default="./images/",
    show_default=True,
    help="Path to the directory containing the images.",
)
@click.option(
    "-p",
    "--preamble",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="""Path to the preamble file. Must be a .md file, which will be displayed
    prior to the summary information.""",
)
@click.option(
    "-c",
    "--code",
    type=str,
    help="""The observing code to generate the report for. If none provided,
    a report including all observing codes will be generated.""",
)
@click.option(
    "-o",
    "--output",
    default="",
    type=click.Path(exists=False, dir_okay=False, resolve_path=True),
    help="""Path to the output file. If none provided, the default is report_%Y-%m-%d
    where the date is from the first observation in the schedules directory.""",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    type=click.IntRange(0, 1),  # Range can be changed
    default=0,
    help="Increase verbosity",
)
@click.version_option()
def summary_report_cli(
    schedule, images_dir="./images/", preamble=None, code=None, verbose=-1
):
    """
    Generates an HTML summary report on the requested schedule.\b

    Using information written to the schedule file by TelrunOperator and
    the corresponding image files from that schedule, a summary report
    detailing the observing conditions and the specific outcomes of each
    image is generated. The report is written to a .html file for easy
    viewing in a web browser and compatability with various summary
    delivery methods (e.g. email, webpage, etc.).

    .. hint::
        A unique report could be generated for each observing code by
        using the ``--code`` option. For example, each observer could
        receive a custom report detailing only their observations.

    Parameters
    ----------
    schedule : `str`, optional
        Path to the schedule file. If none provided, the default behavior
        is to check the ./schedules/ directory for the latest schedule.

    images_dir : `str`, default='./images/'
        Path to the directory containing the images.

    preamble : `str`, optional
        Path to the preamble file. Must be a .md file, which will be
        displayed prior to the summary information.

    code : `str`, optional
        The observing code to generate the report for. If none provided,
        a report including all observing codes will be generated.

    output : `str`, optional
        Path to the output file. If none provided from the CLI, the default is in the
        ./reports/ directory, report_%Y-%m-%d where the date is from the
        first observation's UTC time in the schedules directory. If none
        provided from the API, the output HTML is returned as a string.

    verbose : `int`, {-1, 0, 1}, default=-1
        Verbosity level. 0: no debug messages, 1: debug messages. Default set to -1
        for API use and 0 for CLI use.

    Returns
    -------
    `str`, optional
        If no output file is provided, the HTML report is returned as a string.

    Raises
    ------
    `click.BadOptionUsage`

    See Also
    --------


    Notes
    -----


    Examples
    --------

    """

    if verbose > -1:
        logger.setLevel(int(10 * (2 - verbose)))  # Change range via 2
        logger.addHandler(logging.StreamHandler())  # Redirect to stdout
    logger.debug(f"Verbosity level set to {verbose}")
    logger.debug(f"""summary_report_cli()""")

    if code is not None:
        if len(code) != 3:
            raise click.BadOptionUsage("Observing codes must be 3 characters long.")

    # Get schedule
    if schedule is None:
        last_mtime = 0
        for fname in glob.glob("./schedules/*.ecsv"):
            if os.path.getmtime(schedule) > last_mtime:
                last_mtime = os.path.getmtime(schedule)
                schedule = fname
    schedule = table.Table.read(schedule, format="ascii.ecsv")

    # Check images all exist
    fnames = []
    headers = []
    for row in schedule:
        for fname in row["configuration"]["filename"].split(","):
            if not os.path.exists(os.path.join(images_dir, fname)):
                raise click.BadOptionUsage(f"Image {fname} does not exist.")
            else:
                fnames.append(os.path.join(images_dir, fname))
                headers.append(fits.getheader(fnames[-1]))

    tbl = table.Table(
        [
            [hdr["OBSCODE"] for hdr in headers],
            [hdr["OBSERVER"] for hdr in headers],
            fnames,
            [hdr["TARGET"] for hdr in headers],
            [
                astrotime.Time(hdr["DATE-OBS"], format="fits", scale="utc")
                for hdr in headers
            ],
            [
                astrotime.Time(row["start time (UTC)"], format="iso", scale="utc")
                for row in schedule
            ],
            [hdr.get("FILTER", "None") for hdr in headers],
            [hdr["READOUT"] for hdr in headers],
            [str(hdr["XBINNING"]) + "x" + str(hdr["YBINNING"]) for hdr in headers],
            [hdr["EXPTIME"] for hdr in headers],
            [
                coord.SkyCoord(
                    hdr["SCHEDRA"], hdr["SCHEDDEC"], unit=("hourangle", "deg")
                )
                for hdr in headers
            ],
            [hdr["AIRMASS"] for hdr in headers],
            [hdr.get("ZMAG", np.nan) for hdr in headers],
            [hdr.get("ZMAGERR", np.nan) for hdr in headers],
            [hdr.get("FWHMH", np.nan) for hdr in headers],
            [hdr.get("FWHMHS", np.nan) for hdr in headers],
            [hdr.get("FWHMV", np.nan) for hdr in headers],
            [hdr.get("FWHMVS", np.nan) for hdr in headers],
            [hdr["XPIXSCAL"] for hdr in headers],
            [hdr["YPIXSCAL"] for hdr in headers],
            [hdr["MOONANGL"] for hdr in headers],
            [hdr["MOONPHAS"] for hdr in headers],
            [
                (
                    coord.SkyCoord(
                        hdr["OBJCTRA"], hdr["OBJCTDEC"], unit=("hourangle", "deg")
                    )
                    if None not in (hdr.get("OBJCTRA", None), hdr.get("OBJCTDEC", None))
                    else None
                )
                for hdr in headers
            ],
        ],
        names=[
            "code",
            "observer",
            "filename",
            "target",
            "obs time",
            "start time",
            "filter",
            "readout",
            "binning",
            "exptime",
            "sched coords",
            "airmass",
            "Zmag",
            "Zmag_err",
            "FWHM_H",
            "FWHM_H_err",
            "FWHM_V",
            "FWHM_V_err",
            "xpixscale",
            "ypixscale",
            "moon angle",
            "moon phase",
            "obj coords",
        ],
    )

    tbl.sort("start time")

    # Get summary information
    n_codes = len(np.unique(tbl["code"]))
    n_images = len(tbl)
    exp_tot = np.sum(tbl["exptime"])
    moon_phs_range = (np.min(tbl["moon phase"]), np.max(tbl["moon phase"]))

    # Seeing
    seeing_arr = np.mean(
        [tbl["xpixscale"] * tbl["FWHM_H"], tbl["ypixscale"] * tbl["FWHM_V"]], axis=0
    )
    seeing_median = np.median(seeing_arr)
    seeing_std = np.std(seeing_arr)

    # Zmag
    zmags = {}
    zmag_errs = {}
    for filt in ["u", "b", "g", "r", "i", "z"]:
        reads = {}
        read_errs = {}
        for readout in np.unique(tbl["readout"]):
            idx = np.where(
                (tbl["filter"].lower() == filt) & (tbl["readout"] == readout)
            )[0]
            if len(idx) > 0:
                reads[readout] = np.median(tbl["Zmag"][idx])
                read_errs[readout] = np.std(tbl["Zmag"][idx])
        if len(reads) > 0:
            zmags[filt] = reads
            zmag_errs[filt] = read_errs

    # Pointing errors in arcsec
    dra = np.array(
        [
            (
                (tbl["sched coords"][i].ra.deg - tbl["obj coords"][i].ra.deg) / u.arcsec
                if tbl["obj coords"][i] is not None
                else np.nan
            )
            for i in range(len(tbl))
        ]
    )
    ddec = np.array(
        [
            (
                (tbl["sched coords"][i].dec.deg - tbl["obj coords"][i].dec.deg)
                / u.arcsec
                if tbl["obj coords"][i] is not None
                else np.nan
            )
            for i in range(len(tbl))
        ]
    )
    dra_median = np.median(dra)
    dra_std = np.std(dra)
    ddec_median = np.median(ddec)
    ddec_std = np.std(ddec)

    # Median late time in minutes
    late_median = np.median((tbl["obs time"] - tbl["start time"]).sec) / 60
    late_std = np.std((tbl["obs time"] - tbl["start time"]).sec) / 60

    # Summary table
    summary_table = table.Table(
        [
            np.unique(tbl["code"]),
            [tbl["observer"][tbl["code"] == code] for code in all_codes],
            [len(tbl[tbl["code"] == code]) for code in all_codes],
            [
                np.sum(tbl["exptime"][tbl["code"] == code]) / 60 for code in all_codes
            ],  # Minutes
            [np.unique(tbl["target"][tbl["code"] == code]) for code in all_codes],
        ],
        names=["code", "observer", "n images", "exp time [min]", "targets"],
    )

    if output == "":
        output = "report_" + tbl["start time"][0].iso.split("T")[0] + ".html"

    # Generate code specific table
    if code is not None:
        tbl = tbl[tbl["code"] == code]

    if output is not None:
        with open(output, "w") as f:
            f.write(
                f"""<!DOCTYPE html>
            <html>
            <head>
            <style>
            table {{
              font-family: arial, sans-serif;
              border-collapse: collapse;
              width: 100%;
            }}

            td, th {{
              border: 1px solid #dddddd;
              text-align: left;
              padding: 8px;
            }}

            tr:nth-child(even) {{
              background-color:
            }}
            </style>
            </head>
            <body>
            """
            )

            if preamble is not None:
                with open(preamble, "r") as f:
                    preamble = f.read()
                f.write(markdown.markdown(preamble))

            f.write(
                f"""
            <h1>Observing Summary</h1>
            <p>Number of observing projects: {n_codes}</p>
            <p>Number of images: {n_images}</p>
            <p>Total exposure time: {exp_tot:.2f} minutes</p>
            <p>Moon phase range: {100*moon_phs_range[0]:.2f}% -- {100*moon_phs_range[1]:.2f}%</p>
            <p>Median seeing: {seeing_median:.2f} +/- {seeing_std:.2f} arcsec</p>
            <p>Median pointing error: {dra_median:.2f} +/- {dra_std:.2f} arcsec (RA), {ddec_median:.2f} +/- {ddec_std:.2f} arcsec (Dec)</p>
            <p>Median late time: {late_median:.2f} +/- {late_std:.2f} minutes</p>
            """
            )

            f.write(
                f"""
            <h2>Summary Table</h2>
            {summary_table.pformat_all(html=True)}
            """
            )

            f.write(
                f"""
            <h2>Image Table</h2>
            {tbl.pformat_all(html=True)}
            """
            )

            f.write(
                f"""
            </body>
            </html>
            """
            )

        logger.info(f"Summary report written to {output}")

    if verbose == -1:
        return (
            n_codes,
            n_images,
            exp_tot,
            moon_phs_range,
            seeing_median,
            seeing_std,
            zmags,
            zmag_errs,
            dra_median,
            dra_std,
            ddec_median,
            ddec_std,
            late_median,
            late_std,
            summary_table,
            tbl,
        )


@click.command()
def schedule_report_cli():
    pass


summary_report = summary_report_cli.callback
schedule_report = schedule_report_cli.callback
