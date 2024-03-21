import datetime
import logging
import re
import shlex

import astroplan
import numpy as np
from astropy import coordinates as coord
from astropy import time as astrotime
from astropy import units as u
from astroquery import mpc

from pyscope import __version__

logger = logging.getLogger(__name__)


def read(
    filename,
    location=None,
    t0=None,
    default_title="Pyscope Observation",
    default_observers="default",
    default_code="aaa",
    default_nonsidereal=False,
    default_priority=1,
    default_repositioning=(0, 0),
    default_shutter_state=True,
    default_readout=0,
    default_binning=(1, 1),
    default_frame_position=(0, 0),
    default_frame_size=(0, 0),
    default_nexp=1,
    default_do_not_interrupt=False,
):
    possible_keys = {
        "tit": "title",
        "obs": "observer",
        "cod": "code",
        "bl": "block",
        "ty": "type",
        "ba": "backend",
        "dates": "datestart",
        "datee": "dateend",
        "so": "source",
        "ta": "source",
        "obj": "source",
        "na": "source",
        "ra": "ra",
        "ri": "ra",
        "de": "dec",
        "no": "nonsidereal",
        "non_": "nonsidereal",
        "non-": "nonsidereal",
        "pm_r": "pm_ra_cosdec",
        "pm_d": "pm_dec",
        "file": "filename",
        "prio": "priority",
        "repo": "repositioning",
        "sh": "shutter_state",
        "read": "readout",
        "bi": "binning",
        "frame_p": "frame_position",
        "frame_s": "frame_size",
        "u": "utstart",
        "st": "utstart",
        "ca": "cadence",
        "sc": "schederr",
        "n_": "nexp",
        "ne": "nexp",
        "repe": "nexp",
        "do": "do_not_interrupt",
        "filt": "filter",
        "exp": "exposures",
        "du": "exposures",
        "dw": "exposures",
        "tim": "exposures",
        "l": "exposures",
        "com": "comment",
    }

    with open(filename, "r") as f:
        raw_lines = f.readlines()

    # Remove equal signs, quotes, blank lines, etc
    lines = []
    for line in raw_lines:
        logger.debug(f"Parsing line: {line}")
        line = line.replace("=", " ")
        line = line.replace(": ", " ")
        line = line.replace(";", ",")
        line = line.replace(", ", ",")
        line = line.replace("(", '"')
        line = line.replace(")", '"')
        line = line.replace("[", '"')
        line = line.replace("]", '"')
        line = line.replace("{", '"')
        line = line.replace("}", '"')
        line = line.replace("<", '"')
        line = line.replace(">", '"')
        line = line.replace("'", '"')
        line = line.replace("`", '"')
        line = line.replace("‘", '"')
        line = line.replace("’", '"')
        lines.append(line)

    # From: https://stackoverflow.com/questions/28401547/how-to-remove-comments-from-a-string
    lines = [re.sub(r"(?m)^ *#.*\n?", "", line) for line in lines]
    lines = [re.sub(r"(?m)^ *!.*\n?", "", line) for line in lines]
    lines = [re.sub(r"(?m)^ *%.*\n?", "", line) for line in lines]
    lines = [line.split("#")[0] for line in lines]  # Remove comments
    lines = [line.split("!")[0] for line in lines]  # Remove comments
    lines = [line.split("%")[0] for line in lines]  # Remove comments

    lines = [line.replace("\n", "") for line in lines]  # Remove line breaks
    lines = [
        " ".join(line.split()) for line in lines
    ]  # Remove multiple, trailing, leading whitespace
    lines = [line for line in lines if line != ""]  # Remove empty lines

    # From: https://stackoverflow.com/questions/35544325/python-convert-entire-string-to-lowercase-except-for-substrings-in-quotes
    lines = [
        re.sub(r'\b(?<!")(\w+)(?!")\b', lambda match: match.group(1).lower(), line)
        for line in lines
    ]  # Make entire line lowercase except substrings in quotes

    # Turn each line into a dictionary, parse keywords
    lines = [
        dict(
            (key, value)
            for key, value in zip(shlex.split(line)[::2], shlex.split(line)[1::2])
        )
        for line in lines
    ]
    for line_number, line in enumerate(lines):
        new_line = dict()
        for key in line.keys():
            key_matches = []
            value_matches = []
            for possible_key in possible_keys.keys():
                if key.startswith(possible_key):
                    key_matches.append(possible_key)
                    value_matches.append(possible_keys[possible_key])
            value_matches = list(set(value_matches))
            if len(value_matches) > 1:
                logger.error(
                    f"Keyword {key} matches multiple possible keywords: {value_matches}, removing line {line_number}: {line}"
                )
                lines.remove(line)
                continue
            elif len(value_matches) == 0:
                logger.error(
                    f"Keyword {key} does not match any possible keywords: {possible_keys.values()}, removing line {line_number}: {line}"
                )
                lines.remove(line)
                continue
            new_line.update({value_matches[0]: line[key]})
        lines[line_number] = new_line

    # Look for title, observers, code, datestart, and dateend keywords
    title_matches = []
    line_matches = []
    for line_number, line in enumerate(lines):
        if "title" in line.keys():
            title_matches.append(line["title"])
            line_matches.append(line_number)
            if len(line.keys()) > 1:
                logger.warning(
                    f"Multiple keywords found on title line {line_number}, ignoring all but title: {line}"
                )
    lines = [
        line
        for line_number, line in enumerate(lines)
        if line_number not in line_matches
    ]

    if len(title_matches) > 1:
        logger.warning(f"Multiple titles found: {title_matches}, using first")
        title = title_matches[0]
    elif len(title_matches) == 1:
        title = title_matches[0]
    else:
        logger.warning("No title found, using parsing function default")
        title = default_title

    observers = []
    line_matches = []
    for line_number, line in enumerate(lines):
        if "observer" in line.keys():
            observers.append(line["observer"])
            line_matches.append(line_number)
            if len(line.keys()) > 1:
                logger.warning(
                    f"Multiple keywords found on observer line {line_number}, ignoring all but observer: {line}"
                )
    lines = [
        line
        for line_number, line in enumerate(lines)
        if line_number not in line_matches
    ]
    if len(observers) == 0:
        logger.warning("No observers found, using parsing function default")
        observers = default_observers

    code_matches = []
    line_matches = []
    for line_number, line in enumerate(lines):
        if "code" in line.keys():
            code_matches.append(line["code"])
            line_matches.append(line_number)
            if len(line.keys()) > 1:
                logger.warning(
                    f"Multiple keywords found on code line {line_number}, ignoring all but code: {line}"
                )
    lines = [
        line
        for line_number, line in enumerate(lines)
        if line_number not in line_matches
    ]
    if len(code_matches) > 1:
        logger.warning(f"Multiple codes found: {code_matches}, using first")
        code = code_matches[0]
    elif len(code_matches) == 1:
        code = code_matches[0]
    else:
        logger.warning("No code found, using parsing function default")
        code = default_code

    datestart_matches = []
    line_matches = []
    for line_number, line in enumerate(lines):
        if "datestart" in line.keys():
            datestart_matches.append(line["datestart"])
            line_matches.append(line_number)
            if len(line.keys()) > 1:
                logger.warning(
                    f"Multiple keywords found on datestart line {line_number}, ignoring all but datestart: {line}"
                )
    lines = [
        line
        for line_number, line in enumerate(lines)
        if line_number not in line_matches
    ]
    if len(datestart_matches) > 1:
        logger.warning(f"Multiple datestarts found: {datestart_matches}, using first")
        datestart = astrotime.Time(
            datestart_matches[0],
            format="isot",
        )
    elif len(datestart_matches) == 1:
        datestart = astrotime.Time(
            datestart_matches[0],
            format="isot",
        )
    else:
        logger.info("No datestart found")
        datestart = None

    dateend_matches = []
    line_matches = []
    for line_number, line in enumerate(lines):
        if "dateend" in line.keys():
            dateend_matches.append(line["dateend"])
            line_matches.append(line_number)
            if len(line.keys()) > 1:
                logger.warning(
                    f"Multiple keywords found on dateend line {line_number}, ignoring all but dateend: {line}"
                )
    lines = [
        line
        for line_number, line in enumerate(lines)
        if line_number not in line_matches
    ]
    if len(dateend_matches) > 1:
        logger.warning(f"Multiple dateends found: {dateend_matches}, using first")
        dateend = astrotime.Time(
            dateend_matches[0],
            format="isot",
        )
    elif len(dateend_matches) == 1:
        dateend = astrotime.Time(
            dateend_matches[0],
            format="isot",
        )
    else:
        logger.info("No dateend found")
        dateend = None

    if datestart is not None:
        if dateend is not None:
            if datestart.jd > dateend.jd:
                logger.error(f"datestart must be before dateend, ignoring")
                datestart = None
                dateend = None
        else:
            if datestart.jd > astrotime.Time.now().jd:
                logger.exception(f"datestart must be before now, exiting")
                return
    else:
        if dateend is not None:
            if dateend.jd < astrotime.Time.now().jd:
                logger.exception(
                    f"dateend must be after now, renaming file to .sch.old and exiting"
                )
                os.rename(filename, filename.replace(".sch", ".sch.old"))
                return

    # Look for block keywords and collapse into single line
    new_lines = []
    i = 0
    while i < len(lines):
        if "block" in lines[i].keys():
            if len(lines[i].keys()) > 1:
                logger.warning(
                    f"Multiple keywords found on block line {line_number}, ignoring all but block: {line}"
                )
            if lines[i]["block"].startswith("s"):
                if i >= len(lines) - 2:
                    logger.error(
                        f"Block cannot start with s on last line, ignoring: {line}"
                    )
                    continue
                new_line = dict()
                j = i + 1
                while j < len(lines):
                    if "block" not in lines[j].keys():
                        new_line.update(lines[j])
                        j += 1
                    elif "block" in lines[j].keys():
                        if lines[j]["block"].startswith("e"):
                            break
                        else:
                            logger.error(
                                f"Block on line {line_number} not ended properly, ignoring: {line}"
                            )
                            j = len(lines)
                else:
                    logger.error(
                        f"Block end never found for block starting on line {line_number}, ignoring: {line}"
                    )
                    i = j + 1
                    continue
                new_lines.append(new_line)
                i = j + 1
                continue
            elif lines[i]["block"].startswith("e"):
                logger.error(
                    f"Block cannot start with e on line {line_number}, ignoring: {line}"
                )
                continue
            else:
                logger.error(
                    f"Block must start with s or e on line {line_number}, ignoring: {line}"
                )
                continue
        else:
            new_lines.append(lines[i])
            i += 1
    lines = new_lines

    # If no t0 or location, remove nonsidereal lines
    if t0 is None or location is None:
        lines = [
            line for line in lines if "nonsidereal" not in line.keys()
        ]  # Remove nonsidereal lines

    prior_filters = None
    prior_exposures = None

    # Parse each line and place into ObservingBlock
    blocks = []

    for line_number, line in enumerate(lines):
        # Parse source
        source_name = ""
        if "source" in line.keys():
            source_name = line["source"]

        # Parse ra and dec
        ra = None
        dec = None
        if "ra" in line.keys() and "dec" in line.keys():
            ra = line["ra"]
            dec = line["dec"]
            if len(ra.split(":")) == 3:
                split_ra = ra.split(":")
                ra = f"{split_ra[0]}h{split_ra[1]}m{split_ra[2]}s"

        # Parse nonsidereal
        nonsidereal = default_nonsidereal
        if "nonsidereal" in line.keys():
            if (
                line["nonsidereal"].startswith("t")
                or line["nonsidereal"].startswith("1")
                or line["nonsidereal"].startswith("y")
            ):
                nonsidereal = True
            elif (
                line["nonsidereal"].startswith("f")
                or line["nonsidereal"].startswith("0")
                or line["nonsidereal"].startswith("n")
            ):
                nonsidereal = False
            else:
                logger.warning(
                    f"nonsidereal flag must be true or false on line {line_number}, using parsing function default ({default_nonsidereal}): {line}"
                )

        # Parse pm_ra_cosdec and pm_dec
        pm_ra_cosdec = 0 * u.arcsec / u.hour
        pm_dec = 0 * u.arcsec / u.hour
        if "pm_ra_cosdec" in line.keys() and "pm_dec" in line.keys() and nonsidereal:
            pm_ra_cosdec = float(line["pm_ra_cosdec"]) * u.arcsec / u.hour
            pm_dec = float(line["pm_dec"]) * u.arcsec / u.hour
        elif (
            "pm_ra_cosdec" in line.keys()
            and "pm_dec" in line.keys()
            and not nonsidereal
        ):
            logger.warning("Proper motions found on non-nonsidereal line, ignoring")
        elif (
            "pm_ra_cosdec" not in line.keys()
            and "pm_dec" not in line.keys()
            and nonsidereal
            and source_name is not None
        ):
            logger.info(
                f"No proper motions found on line {line_number}, will attempt MPC lookup: {source_name}"
            )
            try:
                ephemerides = mpc.MPC.get_ephemeris(
                    target=source_name,
                    location=location,
                    start=t0.isot,
                    number=1,
                    step=1 * u.second,
                    proper_motion="sky",
                )
                ra = ephemerides["RA"][0]
                dec = ephemerides["Dec"][0]
                pm_ra_cosdec = ephemerides["dRA cos(Dec)"][0] * u.arcsec / u.hour
                pm_dec = ephemerides["dDec"][0] * u.arcsec / u.hour
            except Exception as e1:
                try:
                    logger.warning(
                        f"Failed to find proper motions for {source_name} on line {line_number}, trying to find proper motions using astropy.coordinates.get_body: {e1}"
                    )
                    pos_l = coord.get_body(
                        source_name, t0 - 10 * u.minute, location=location
                    )
                    pos_m = coord.get_body(source_name, t0, location=location)
                    pos_h = coord.get_body(
                        source_name, t0 + 10 * u.minute, location=location
                    )
                    ra = pos_m.ra.to_string("hourangle", sep="hms", precision=3)
                    dec = pos_m.dec.to_string("deg", sep="dms", precision=2)
                    pm_ra_cosdec = (
                        (
                            pos_h.ra * np.cos(pos_h.dec.rad)
                            - pos_l.ra * np.cos(pos_l.dec.rad)
                        )
                        / (pos_h.obstime - pos_l.obstime)
                    ).to(u.arcsec / u.hour)
                    pm_dec = (
                        (pos_h.dec - pos_l.dec) / (pos_h.obstime - pos_l.obstime)
                    ).to(u.arcsec / u.hour)
                except Exception as e2:
                    logger.warning(
                        f"Failed to find proper motions for {source_name} on line {line_number}, skipping: {e2}"
                    )
                    continue
        elif (
            "pm_ra_cosdec" not in line.keys()
            and "pm_dec" not in line.keys()
            and nonsidereal
            and source_name is None
        ):
            logger.warning(
                f"No proper motions found on line {line_number} and no source name found for MPC lookup, ignoring nonsidereal: {line}"
            )
        elif (
            "pm_ra_cosdec" not in line.keys()
            and "pm_dec" in line.keys()
            and nonsidereal
        ):
            logger.warning(
                f"Missing proper motion pm_ra_cosdec on line {line_number}, assuming 0: {line}"
            )
        elif (
            "pm_ra_cosdec" in line.keys()
            and "pm_dec" not in line.keys()
            and nonsidereal
        ):
            logger.warning(
                f"Missing proper motion pm_dec on line {line_number}, assuming 0: {line}"
            )

        # Parse source if not already parsed by pm lookup
        if source_name is None and ra is None and dec is None:
            logger.warning(
                f"No source or coordinates found on line {line_number}, skipping: {line}"
            )
            continue
        elif None not in (ra, dec):
            obj = coord.SkyCoord(
                ra,
                dec,
                unit=(u.hourangle, u.deg),
                pm_ra_cosdec=pm_ra_cosdec,
                pm_dec=pm_dec,
            )
            if source_name is None:
                source_name = obj.to_string("hmsdms")
        elif source_name is not None:
            try:
                obj = coord.SkyCoord.from_name(source_name)
            except Exception as e:
                logger.warning(
                    f"Failed to parse source name on line {line_number}, skipping: {e}"
                )
                continue
        else:
            logger.warning(
                f"Failed to parse source on line {line_number}, skipping: {line}"
            )
            continue

        # Parse filename
        fname = ""
        if "filename" in line.keys():
            fname = line["filename"]
        # Parse priority
        priority = default_priority
        if "priority" in line.keys():
            priority = int(line["priority"])

        # Parse repositioning
        repositioning = default_repositioning
        if "repositioning" in line.keys():
            if (
                line["repositioning"].startswith("t")
                or line["repositioning"].startswith("1")
                or line["repositioning"].startswith("y")
            ):
                repositioning = (-1, -1)
            elif (
                line["repositioning"].startswith("f")
                or line["repositioning"].startswith("0")
                or line["repositioning"].startswith("n")
            ):
                repositioning = (0, 0)
            elif (
                line["repositioning"].split("x")[0].isnumeric()
                and line["repositioning"].split("x")[1].isnumeric()
            ):
                repositioning = (
                    int(line["repositioning"].split("x")[0]),
                    int(line["repositioning"].split("x")[1]),
                )
            elif (
                line["repositioning"].split(",")[0].isnumeric()
                and line["repositioning"].split(",")[1].isnumeric()
            ):
                repositioning = (
                    int(line["repositioning"].split(",")[0]),
                    int(line["repositioning"].split(",")[1]),
                )
            else:
                logger.warning(
                    f"repositioning flag must be true, false, or integers split with an x on line {line_number}, setting to parsing function default ({default_repositioning}): {line}"
                )

        # Parse shutter state
        shutter_state = default_shutter_state
        if "shutter_state" in line.keys():
            if (
                line["shutter_state"].startswith("o")
                or line["shutter_state"].startswith("t")
                or line["shutter_state"].startswith("1")
                or line["shutter_state"].startswith("y")
            ):
                shutter_state = True
            elif (
                line["shutter_state"].startswith("c")
                or line["shutter_state"].startswith("f")
                or line["shutter_state"].startswith("0")
                or line["shutter_state"].startswith("n")
            ):
                shutter_state = False
            else:
                logger.warning(
                    f"shutter_state flag must be open or closed on line {line_number}, setting to parsing function default ({default_shutter_state}): {line}"
                )

        # Parse readout
        readout = default_readout
        if "readout" in line.keys():
            readout = int(line["readout"])

        # Parse binning
        binning = default_binning
        if "binning" in line.keys():
            if (
                line["binning"].split("x")[0].isnumeric()
                and line["binning"].split("x")[1].isnumeric()
            ):
                binning = (
                    int(line["binning"].split("x")[0]),
                    int(line["binning"].split("x")[1]),
                )
            elif (
                line["binning"].split(",")[0].isnumeric()
                and line["binning"].split(",")[1].isnumeric()
            ):
                binning = (
                    int(line["binning"].split(",")[0]),
                    int(line["binning"].split(",")[1]),
                )
            else:
                logger.warning(
                    f"binning must be integers split with an x on line {line_number}, setting to parsing function default ({default_binning}): {line}"
                )

        # Parse frame position
        frame_position = default_frame_position
        if "frame_position" in line.keys():
            if (
                line["frame_position"].split("x")[0].isnumeric()
                and line["frame_position"].split("x")[1].isnumeric()
            ):
                frame_position = (
                    int(line["frame_position"].split("x")[0]),
                    int(line["frame_position"].split("x")[1]),
                )
            elif (
                line["frame_position"].split(",")[0].isnumeric()
                and line["frame_position"].split(",")[1].isnumeric()
            ):
                frame_position = (
                    int(line["frame_position"].split(",")[0]),
                    int(line["frame_position"].split(",")[1]),
                )
            else:
                logger.warning(
                    f"frame_position must be integers split with a comma on line {line_number}, setting to parsing function default ({default_frame_position}): {line}"
                )

        # Parse frame size
        frame_size = default_frame_size
        if "frame_size" in line.keys():
            if (
                line["frame_size"].split("x")[0].isnumeric()
                and line["frame_size"].split("x")[1].isnumeric()
            ):
                frame_size = (
                    int(line["frame_size"].split("x")[0]),
                    int(line["frame_size"].split("x")[1]),
                )
            elif (
                line["frame_size"].split(",")[0].isnumeric()
                and line["frame_size"].split(",")[1].isnumeric()
            ):
                frame_size = (
                    int(line["frame_size"].split(",")[0]),
                    int(line["frame_size"].split(",")[1]),
                )
            else:
                logger.warning(
                    f"frame_size must be integers split with a comma or x on line {line_number}, setting to parsing function default ({default_frame_size}): {line}"
                )

        # Get utstart, cadence, schederr
        utstart = None
        if "utstart" in line.keys():
            check_day = False
            if "t" not in line["utstart"]:
                line["utstart"] = f"{t0.isot.split('T')[0]}T{line['utstart']}"
                check_day = True
            utstart = astrotime.Time(
                line["utstart"].upper(), format="isot", scale="utc"
            )
            if check_day:
                if utstart.jd < t0.jd:
                    utstart += 1 * u.day

        cadence = None
        if "cadence" in line.keys():
            h, m, s = line["cadence"].split(":")
            cadence = astrotime.TimeDelta(
                datetime.timedelta(hours=int(h), minutes=int(m), seconds=float(s)),
                format="datetime",
            )

            if utstart is None:
                logger.error(
                    f"Must specify utstart if cadence is specified on line {line_number}, skipping: {line}"
                )
                continue

        schederr = None
        if "schederr" in line.keys():
            h, m, s = line["schederr"].split(":")
            schederr = astrotime.TimeDelta(
                datetime.timedelta(hours=int(h), minutes=int(m), seconds=float(s)),
                format="datetime",
            )

        if utstart is None and schederr is not None:
            logger.error(
                f"Must specify utstart if schederr is specified on line {line_number}, skipping: {line}"
            )
            continue
        elif utstart is not None and schederr is None:
            logger.info(
                f"Assuming schederr is 60 seconds on line {line_number}: {line}"
            )
            schederr = 60 * u.second

        # Get exposure behavior keywords
        nexp = default_nexp
        if "nexp" in line.keys():
            nexp = int(line["nexp"])

        do_not_interrupt = default_do_not_interrupt
        if "do_not_interrupt" in line.keys():
            if (
                line["do_not_interrupt"].startswith("t")
                or line["do_not_interrupt"].startswith("1")
                or line["do_not_interrupt"].startswith("y")
            ):
                do_not_interrupt = True
            elif (
                line["do_not_interrupt"].startswith("f")
                or line["do_not_interrupt"].startswith("0")
                or line["do_not_interrupt"].startswith("n")
            ):
                do_not_interrupt = False
            else:
                logger.warning(
                    f"do_not_interrupt flag must be true or false on line {line_number}, setting to parsing function default ({default_do_not_interrupt}): {line}"
                )

        # Get comment
        comment = ""
        if "comment" in line.keys():
            comment = line["comment"]

        # Get filters
        filters = []
        if "filter" in line.keys():
            filters = line["filter"].split(",")
            prior_filters = filters
        elif prior_filters is not None:
            filters = prior_filters
        else:
            filters = []
            prior_filters = None

        # Get exposures
        exposures = []
        if "exposures" in line.keys():
            exposures = [float(e) for e in line["exposures"].split(",")]
            prior_exposures = exposures
        elif prior_exposures is not None:
            exposures = prior_exposures
        else:
            exposures = []
            prior_exposures = None

        # Expand exposures or filters to match length of the other if either length is one
        if len(exposures) == 1 and len(filters) > 1:
            exposures = exposures * len(filters)
        elif len(filters) == 1 and len(exposures) > 1:
            filters = filters * len(exposures)

        # Sanity Check 1: matching number of filters and exposures
        if len(filters) != len(exposures) and len(filters) != 0:
            logger.error(
                f"Number of filters ({len(filters)}) does not match number of exposures ({len(exposures)}) on line {line_number}, skipping: {line}"
            )
            continue

        # Sanity Check 2: do_not_interrupt and cadence don't both appear:
        if do_not_interrupt and cadence is not None:
            logger.error(
                f"do_not_interrupt and cadence cannot both be specified on line {line_number}, skipping: {line}"
            )
            continue

        # Sanity Check 3: if cadence is specified, verify it exceeds exposure time
        # times number of exposures times number of filters
        if cadence is not None:
            if cadence.to(u.second).value < np.sum(exposures) * nexp * len(filters):
                logger.warning(
                    f"Cadence ({cadence}) is less than total exposure time ({np.sum(exposures) * nexp * len(filters)}) on line {line_number}, setting cadence to total exposure time: {line}"
                )
                cadence = np.sum(exposures) * nexp * len(filters)

        for i in range(len(filters)):
            filt = filters[i]
            exp = exposures[i]
            constraints = None

            if do_not_interrupt:
                loop_max = 1
                temp_dur = exp * nexp * u.second
                temp_nexp = nexp
            else:
                loop_max = nexp
                temp_dur = exp * u.second
                temp_nexp = 1

            if utstart is not None:
                if cadence is not None:
                    constraint_cadence = cadence
                else:
                    constraint_cadence = temp_dur

                constraints = [
                    [
                        astroplan.constraints.TimeConstraint(
                            min=(
                                utstart
                                + (i + j * len(filters)) * constraint_cadence
                                - schederr
                            ),
                            max=(
                                utstart
                                + (i + j * len(filters)) * constraint_cadence
                                + schederr
                            ),
                        )
                    ]
                    for j in range(loop_max)
                ]
            else:
                constraints = [None for j in range(loop_max)]

            for j in range(loop_max):
                if loop_max > 1 and fname != "":
                    final_fname = f"{fname}_{j}"
                else:
                    final_fname = f"{fname}"
                temp_id = astrotime.Time.now()
                temp_id.format = "mjd"
                blocks.append(
                    astroplan.ObservingBlock(
                        target=obj,
                        duration=temp_dur,
                        priority=priority,
                        name=source_name,
                        configuration={
                            "observer": observers,
                            "code": code,
                            "title": title,
                            "filename": final_fname,
                            "type": "light",
                            "backend": 0,
                            "filter": filt,
                            "exposure": exp,
                            "nexp": temp_nexp,
                            "repositioning": repositioning,
                            "shutter_state": shutter_state,
                            "readout": readout,
                            "binning": binning,
                            "frame_position": frame_position,
                            "frame_size": frame_size,
                            "pm_ra_cosdec": pm_ra_cosdec,
                            "pm_dec": pm_dec,
                            "comment": comment,
                            "sch": filename.split("/")[-1].split(".")[0],
                            "ID": temp_id,
                            "status": "U",
                            "message": "Unscheduled",
                            "sched_time": None,
                        },
                        constraints=constraints[j],
                    )
                )
                logger.debug(
                    f"""Created ObservingBlock: {blocks[-1].target},
                                {blocks[-1].duration}, {blocks[-1].priority},
                                {blocks[-1].name}, {blocks[-1].constraints},
                                {blocks[-1].configuration}"""
                )

    return blocks


def write(observing_blocks, filename=None):
    if type(observing_blocks) is not list:
        observing_blocks = [observing_blocks]

    codes = []
    for block in observing_blocks:
        if type(block) is not astroplan.scheduling.ObservingBlock:
            logger.exception("observing_blocks must be a list of ObservingBlocks")
            return
        codes.append(block.configuration["code"])

    unique_codes = list(set(codes))

    time_now = astrotime.Time.now().strftime("%Y-%m-%d_%H:%M:%S")

    for unique_code in unique_codes:
        blocks = [
            block
            for block in observing_blocks
            if block.configuration["code"] == unique_code
        ]

        if [block.configuration["title"] for block in blocks].count(
            blocks[0].configuration["title"]
        ) != len(blocks):
            logger.warning(
                f"Title must be the same for all blocks with the same code {unique_code}, setting all titles to first title ({blocks[0].configuration['title']})"
            )
            blocks = [
                block.configuration.update("title", blocks[0].configuration["title"])
                for block in blocks
            ]

        if [block.configuration["observer"] for block in blocks].count(
            blocks[0].configuration["observer"]
        ) != len(blocks):
            logger.warning(
                f"Observer must be the same for all blocks with the same code {unique_code}, setting all observers to first observer ({blocks[0].configuration['observer']})"
            )
            blocks = [
                block.configuration.update(
                    "observer", blocks[0].configuration["observer"]
                )
                for block in blocks
            ]

        if filename is None:
            filename = f"{unique_code}_{time_now}.sch"
        elif len(unique_codes) > 1:
            filename = filename.replace(".sch", f"_{unique_code}_{time_now}.sch")
        else:
            filename = filename

        with open(filename, "w") as f:
            f.write(f"# {len(blocks)} Blocks\n")
            f.write(f"# Written {time_now}\n")
            f.write(f"# By pyscope version {__version__}\n")
            f.write("\n")

            f.write('title "{0}"\n'.format(blocks[0].configuration["title"]))
            if type(blocks[0].configuration["observer"]) is not list:
                observers = [blocks[0].configuration["observer"]]
            else:
                observers = blocks[0].configuration["observer"]
            for observer in observers:
                f.write(f'observer "{observer}"\n')
            f.write(f"code {unique_code}\n")
            f.write("\n")

            for block in blocks:
                write_string = "block start\n"
                try:
                    if block.name != "":
                        write_string += f'source "{block.name}"\n'
                except:
                    pass
                write_string += f"ra {block.target.ra.to_string('hourangle', sep='hms', precision=4)}\n"
                write_string += (
                    f"dec {block.target.dec.to_string('deg', sep='dms', precision=3)}\n"
                )
                try:
                    write_string += f"priority {block.priority}\n"
                except:
                    pass
                try:
                    if block.configuration["filename"] != "":
                        write_string += 'filename "{0}"\n'.format(
                            block.configuration["filename"]
                        )
                except:
                    pass
                try:
                    if (
                        block.configuration["pm_ra_cosdec"].value != 0
                        or block.configuration["pm_dec"].value != 0
                    ):
                        write_string += f"nonsidereal true\n"
                        write_string += f"pm_ra_cosdec {block.configuration['pm_ra_cosdec'].to(u.arcsec/u.hour).value}\n"
                        write_string += f"pm_dec {block.configuration['pm_dec'].to(u.arcsec/u.hour).value}\n"
                    else:
                        write_string += f"nonsidereal false\n"
                except:
                    pass
                try:
                    if block.configuration["shutter_state"]:
                        write_string += f"shutter_state open\n"
                    else:
                        write_string += f"shutter_state closed\n"
                except:
                    pass
                write_string += f"exposure {block.configuration['exposure']}\n"
                write_string += f"nexp {block.configuration['nexp']}\n"
                try:
                    if block.configuration["do_not_interrupt"]:
                        write_string += f"do_not_interrupt true\n"
                    else:
                        write_string += f"do_not_interrupt false\n"
                except:
                    pass
                try:
                    write_string += f"readout {block.configuration['readout']}\n"
                except:
                    pass
                try:
                    write_string += f"binning {block.configuration['binning'][0]}x{block.configuration['binning'][1]}\n"
                except:
                    pass
                write_string += f"filter {block.configuration['filter']}\n"
                try:
                    if block.configuration["repositioning"] is True:
                        write_string += f"repositioning true\n"
                    elif type(block.configuration["repositioning"]) is tuple:
                        write_string += f"repositioning {block.configuration['repositioning'][0]}x{block.configuration['repositioning'][1]}\n"
                except:
                    pass
                try:
                    write_string += f"frame_position {block.configuration['frame_position'][0]}x{block.configuration['frame_position'][1]}\n"
                except:
                    pass
                try:
                    write_string += f"frame_size {block.configuration['frame_size'][0]}x{block.configuration['frame_size'][1]}\n"
                except:
                    pass
                try:
                    write_string += f"utstart {block.start_time.isot}\n"
                except:
                    try:
                        if block.constraints is not None:
                            if type(block.constraints) is not list:
                                block.constraints = [block.constraints]
                            for (
                                constraint
                            ) in (
                                block.constraints
                            ):  # TODO: Add in support for all constraints
                                possible_min_times = []
                                possible_max_times = []
                                if type(constraint) is astroplan.TimeConstraint:
                                    possible_min_times.append(constraint.min)
                                    possible_max_times.append(constraint.max)
                            min_time_idx = np.argmax(
                                [time.jd for time in possible_min_times]
                            )
                            max_time_idx = np.argmin(
                                [time.jd for time in possible_max_times]
                            )
                            min_time = possible_min_times[min_time_idx]
                            max_time = possible_max_times[max_time_idx]
                            mid_time = astrotime.Time(
                                (min_time.jd + max_time.jd) / 2, format="jd"
                            )
                            error_time = round(
                                astrotime.TimeDelta(
                                    (max_time.jd - min_time.jd) / 2, format="jd"
                                ).sec,
                                3,
                            )
                            err_hours = int(error_time / 3600)
                            err_minutes = int((error_time - err_hours * 3600) / 60)
                            err_seconds = (
                                error_time - err_hours * 3600 - err_minutes * 60
                            )
                            write_string += f"utstart {mid_time.isot}\n"
                            write_string += f"schederr {err_hours:02.0f}:{err_minutes:02.0f}:{err_seconds:02.3f}\n"
                    except:
                        pass
                try:
                    if block.configuration["comment"] != "":
                        write_string += 'comment "{comment} -- written by pyscope v{version}"\n'.format(
                            comment=block.configuration["comment"], version=__version__
                        )
                    else:
                        write_string += f'comment "written by pyscope v{__version__}"\n'
                except:
                    write_string += f'comment "written by pyscope v{__version__}"\n'

                f.write(write_string + "block end\n\n")
            f.write("\n")
