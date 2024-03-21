import configparser
import datetime
import json
import logging
import os
import zoneinfo

import astroplan
import click
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import timezonefinder
import tqdm
from astroplan import plots as astroplan_plots
from astropy import coordinates as coord
from astropy import table
from astropy import time as astrotime
from astropy import units as u
from astroquery import mpc
from cmcrameri import cm as ccm
from matplotlib import ticker

from .. import utils
from ..observatory import Observatory
from . import sch, schedtab

logger = logging.getLogger(__name__)


"""
Codes:

U = unscheduled
X = expired
I = invalid
S = scheduled
1,2,3... = failed on this attempt number but can be rescheduled
F = failed, cannot be rescheduled
C = completed
"""


@click.command(
    epilog="""Check out the documentation at
               https://pyscope.readthedocs.io/en/latest/
               for more information."""
)
@click.option(
    "-c",
    "--catalog",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False, readable=True),
    help="""The catalog of .sch files to be scheduled. The catalog can be a
    single .sch file or a .cat file containing a list of .sch files. If no
    catalog is provided, then the function searches for a schedule.cat file
    in the $OBSERVATORY_HOME/schedules/ directory, then searches
    in the current working directory.""",
)
@click.option(
    "-q",
    "--queue",
    type=click.Path(
        exists=True, resolve_path=True, dir_okay=False, readable=True, writable=True
    ),
    help="""A queue table of requested observations (.ecsv). If a catalog is provided
    and -t is set, then the catalog is parsed and the queue is updated with those entries.
    If no catalog is provided, the queue is scheduled. WARNING: If a catalog is provided,
    then existing scheduled entries in the queue will be overwritten. If no catalog is provided,
    then all entries in the queue (including previously scheduled entries) will be re-scheduled.""",
)
@click.option(
    "-ao",
    "--add-only",
    "add_only",
    is_flag=True,
    default=False,
    show_default=True,
    help="""If a catalog and a queue are provided, then only add the catalog
    entries to the queue without scheduling them. By default, the catalog is
    parsed, scheduled, and the queue is updated with the scheduled entries.""",
)
# TODO: Add option to update an existing schedule table
# @click.option(
#     "-es",
#     "--existing-schedule",
#     "existing_schedule",
#     type=click.Path(exists=True, resolve_path=True, dir_okay=False, readable=True,
#         writable=True),
#     help="""An existing schedule table (.ecsv) to be updated."""
# )
@click.option(
    "-i",
    "--ignore-order",
    "ignore_order",
    is_flag=True,
    default=False,
    show_default=True,
    help="""Ignore the order of the .sch files in the catalog,
    acting as if there is only one .sch file. By default, the
    .sch files are scheduled one at a time.""",
)
@click.option(
    "-d",
    "--date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    show_default=False,
    help="""The local date at the observatory of the night to be scheduled.
    By default, the current date at the observatory location is used.""",
)
@click.option(
    "-l",
    "--length",
    type=click.IntRange(min=1, clamp=True),
    default=1,
    show_default=True,
    help="""The length of the schedule [days].""",
)
@click.option(
    "-o",
    "--observatory",
    "observatory",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False, readable=True),
    help="""The observatory configuration file. If no observatory configuration
    file is provided, then the function searches for an observatory.cfg file
    in the $OBSERVATORY_HOME/config/ directory, then searches in the current working
    directory.""",
)
@click.option(
    "-m",
    "--max-altitude",
    "max_altitude",
    type=click.FloatRange(min=-90, max=90, clamp=True),
    default=-12,
    show_default=True,
    help="""The maximum altitude of the Sun for the night [degrees].
    Civil twilight is -6, nautical twilight is -12, and astronomical
    twilight is -18.""",
)
@click.option(
    "-e",
    "--elevation",
    type=click.FloatRange(min=0, max=90, clamp=True),
    default=30,
    show_default=True,
    help="""The minimum elevation of all targets [degrees]. This is a
    'boolean' constraint; that is, there is no understanding of where
    it is closer to ideal to observe the target. To implement a preference
    for higher elevation targets, the airmass constraint should be used.""",
)
@click.option(
    "-a",
    "--airmass",
    type=click.FloatRange(min=1, clamp=True),
    default=3,
    show_default=True,
    help="""The maximum airmass of all targets.""",
)
@click.option(
    "-ms",
    "--moon-separation",
    "moon_separation",
    type=click.FloatRange(min=0, clamp=True),
    default=30,
    show_default=True,
    help="The minimum angular separation between the Moon and all targets [degrees].",
)
@click.option(
    "-s",
    "--scheduler",
    nargs=2,
    type=(
        click.Path(exists=True, resolve_path=True, dir_okay=False, executable=True),
        str,
    ),
    default=("", ""),
    show_default=False,
    help="""The filepath to and name of an astroplan.Scheduler
    custom sub-class. By default, the astroplan.PriorityScheduler is used.""",
)
@click.option(
    "-gt",
    "--gap-time",
    "gap_time",
    type=click.FloatRange(min=0, clamp=True),
    default=60,
    show_default=True,
    help="""The maximum length of time a transition between
    ObservingBlocks could take [seconds].""",
)
@click.option(
    "-r",
    "--resolution",
    type=click.FloatRange(min=1, clamp=True),
    default=5,
    show_default=True,
    help="""The time resolution of the schedule [seconds].""",
)
@click.option(
    "-nf",
    "--name-format",
    "name_format",
    type=str,
    default="{code}_{sch}_{ra}_{dec}_{start_time}",
    show_default=True,
    help="""The format of the scheduled image name. The format
    is a string that can include any column from the schedule table or
    the configuration dictionary.
    """,
)
@click.option(
    "-f",
    "--filename",
    type=click.Path(resolve_path=True, dir_okay=False, writable=True),
    default=None,
    show_default=False,
    help="""The output file name. The file name is formatted with the
    UTC date of the first observation in the schedule. By default,
    it is placed in the current working directory, but if a path is specified,
    the file will be placed there. WARNING: If the file already exists,
    it will be overwritten.""",
)
@click.option(
    "-t",
    "--telrun-execute",
    "telrun_execute",
    is_flag=True,
    default=False,
    show_default=True,
    help="""Places the output file in specified by the $TELRUN_EXECUTE environment
    variable. If not defined, then the $OBSERVATORY_HOME/schedules/ directory is used.
    If neither are defined, then ./schedules/ is used. WARNING: If the file already exists,
    it will be overwritten.""",
)
@click.option(
    "-p",
    "--plot",
    type=click.IntRange(1, 3, clamp=True),
    default=None,
    show_default=False,
    help="""Plots the schedule. The argument specifies the type of plot:
    1: Gantt chart.
    2: target with airmass.
    3: sky chart.""",
)
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    default=False,
    show_default=True,
    help="""Automatically answer yes to any questions. This will write the schedule
    to file even if there are unscheduled or invalid blocks.""",
)
@click.option(
    "-q", "--quiet", is_flag=True, default=False, show_default=True, help="Quiet output"
)
@click.option(
    "-v", "--verbose", default=0, show_default=True, count=True, help="Verbose output"
)
@click.version_option()
def schedtel_cli(
    catalog=None,
    queue=None,
    add_only=False,
    #     existing_schedule=None, # TODO
    ignore_order=False,
    date=None,
    length=1,
    observatory=None,
    max_altitude=-12,
    elevation=30,
    airmass=3,
    moon_separation=30,
    scheduler=("", ""),
    gap_time=60,
    resolution=5,
    name_format="{code}_{sch}_{ra}_{dec}_{start_time}",
    filename=None,
    telrun=False,
    plot=None,
    yes=False,
    quiet=False,
    verbose=0,
):
    # Set up logging
    if quiet:
        level = logging.ERROR
    elif verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    # logging.basicConfig(level=level)

    logger.info("Starting schedtel")
    logger.debug(f"catalog: {catalog}")
    logger.debug(f"ignore_order: {ignore_order}")
    logger.debug(f"date: {date}")
    logger.debug(f"observatory: {observatory}")
    logger.debug(f"max_altitude: {max_altitude}")
    logger.debug(f"elevation: {elevation}")
    logger.debug(f"airmass: {airmass}")
    logger.debug(f"moon_separation: {moon_separation}")
    logger.debug(f"scheduler: {scheduler}")
    logger.debug(f"gap_time: {gap_time}")
    logger.debug(f"resolution: {resolution}")
    logger.debug(f"filename: {filename}")
    logger.debug(f"telrun: {telrun}")
    logger.debug(f"plot: {plot}")
    logger.debug(f"quiet: {quiet}")
    logger.debug(f"verbose: {verbose}")

    # Set the schedule time
    sched_time = astrotime.Time.now()
    sched_time.format = "mjd"

    # Define the observatory
    if observatory is None:
        try:
            observatory = os.environ.get("OBSERVATORY_HOME") + "/config/observatory.cfg"
            logger.info(
                "No observatory provided, using observatory.cfg from $OBSERVATORY_HOME environment variable"
            )
        except:
            observatory = os.getcwd() + "/observatory.cfg"
            logger.info(
                "No observatory provided, using observatory.cfg from current working directory"
            )

    logger.info("Parsing the observatory config")
    if type(observatory) is str:
        obs_cfg = configparser.ConfigParser()
        obs_cfg.read(observatory)
        slew_rate = obs_cfg.getfloat("scheduling", "slew_rate") * u.deg / u.second
        instrument_reconfig_times = json.loads(
            obs_cfg.get("scheduling", "instrument_reconfig_times")
        )
        observatory = astroplan.Observer(
            location=coord.EarthLocation(
                lon=obs_cfg.get("site", "longitude"),
                lat=obs_cfg.get("site", "latitude"),
            )
        )
        obs_lon = observatory.location.lon
        obs_lat = observatory.location.lat
    elif type(observatory) is Observatory:
        obs_lon = observatory.observatory_location.lon
        obs_lat = observatory.observatory_location.lat
        slew_rate = observatory.slew_rate * u.deg / u.second
        instrument_reconfig_times = observatory.instrument_reconfiguration_times
        observatory = astroplan.Observer(
            location=coord.EarthLocation(lon=obs_lon, lat=obs_lat)
        )
    elif type(observatory) is astroplan.Observer:
        obs_lon = observatory.location.lon
        obs_lat = observatory.location.lat
        slew_rate = observatory.slew_rate * u.deg / u.second
        instrument_reconfig_times = observatory.instrument_reconfig_times
    else:
        logger.error(
            "Observatory must be, a string, Observatory object, or astroplan.Observer object."
        )
        return

    # Schedule
    tz = timezonefinder.TimezoneFinder().timezone_at(lng=obs_lon.deg, lat=obs_lat.deg)
    tz = zoneinfo.ZoneInfo(tz)
    logger.debug(f"tz = {tz}")

    if date is None:
        logger.debug("Using current date at observatory location")
        date = datetime.datetime.now()
    else:
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
    date = datetime.datetime(date.year, date.month, date.day, 12, 0, 0, tzinfo=tz)

    t0 = astrotime.Time(
        datetime.datetime(date.year, date.month, date.day, 12, 0, 0, tzinfo=tz),
        format="datetime",
    )
    t1 = t0 + length * u.day
    logger.info("Schedule time range: %s to %s (UTC)" % (t0.iso, t1.iso))

    # TODO: Add option to update an existing schedule table
    schedule = astroplan.Schedule(t0, t1)

    block_groups = []

    if catalog is None and queue is None:
        try:
            catalog = os.environ.get("OBSERVATORY_HOME") + "/schedules/schedule.cat"
            logger.info(
                "No catalog provided, using schedule.cat from $OBSERVATORY_HOME environment variable"
            )
        except:
            catalog = os.getcwd() + "/schedule.cat"
            logger.info(
                "No catalog provided, using schedule.cat from current working directory"
            )

    if type(catalog) is list:
        logger.debug(f"catalog is a list")
        for block in catalog:
            if type(block) is list:
                for b in block:
                    if type(b) is not astroplan.ObservingBlock:
                        logger.error(
                            f"Object {b} in catalog {catalog} is not an astroplan.ObservingBlock, skipping."
                        )
                        continue
                block_groups.append(block)
            elif type(block) is astroplan.ObservingBlock:
                block_groups.append([block])
            else:
                logger.error(
                    f"Object {block} in catalog {catalog} is not an astroplan.ObservingBlock, skipping."
                )
                continue
    elif os.path.isfile(catalog):
        logger.debug(f"catalog is a file")
        if catalog.endswith(".cat"):
            with open(catalog, "r") as f:
                sch_files = f.read().splitlines()
            sch_files = ["/".join(catalog.split("/")[:-1]) + "/" + f for f in sch_files]
            for f in sch_files:
                if not os.path.isfile(f):
                    logger.error(
                        f"File {f} in catalog {catalog} does not exist, skipping."
                    )
                    continue
                try:
                    block_groups.append(
                        sch.read(
                            f,
                            location=coord.EarthLocation(
                                lon=obs_lon,
                                lat=obs_lat,
                            ),
                            t0=t0,
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"File {f} in catalog {catalog} is not a valid .sch file, skipping: {e}"
                    )
                    continue
        elif catalog.endswith(".sch"):
            try:
                block_groups.append(sch.read(catalog))
            except Exception as e:
                logger.error(f"File {catalog} is not a valid .sch file: {e}")
    else:
        logger.error(
            f"Catalog {catalog} is not a valid .cat file or list of astroplan.ObservingBlocks."
        )

    if add_only and not ignore_order:
        logger.error(
            "Option -ao/--add-only requires option -i/--ignore-order, setting to True"
        )
        ignore_order = True

    if ignore_order:
        logger.info("Ignoring order of .sch files in catalog")
        block_groups = [
            [
                block_groups[i][j]
                for i in range(len(block_groups))
                for j in range(len(block_groups[i]))
            ]
        ]

    # Add IDs to ObservingBlocks without them
    logger.info("Adding IDs to ObservingBlocks")
    for i in range(len(block_groups)):
        for j in range(len(block_groups[i])):
            try:
                block_groups[i][j].configuration["ID"]
            except:
                block_groups[i][j].configuration["ID"] = astrotime.Time.now()
                block_groups[i][j].configuration["ID"].format = "mjd"

    previously_queued_blocks = None
    if queue is not None and len(block_groups) > 0:
        if add_only:
            logger.info("Adding catalog to queue without scheduling")

            # Only add the catalog to the queue without scheduling

            return
        else:
            logger.info("Reading queue")
            # Load all blocks from the queue
            queue_blocks = schedtab.table_to_blocks(queue)

            # Mark scheduled as X expired
            for block in queue_blocks:
                if block.configuration["status"] != "U":
                    block.configuration["status"] = "X"
                    block.configuration["message"] = "Expired"
            previously_queued_blocks = queue_blocks

    elif queue is not None and len(block_groups) == 0:
        # Load all blocks from the queue
        queue_blocks = schedtab.table_to_blocks(queue)
        # Scheduled and unscheduled blocks get scheduled
        for block in queue_blocks:
            if block.configuration["status"] in ("S", "U"):
                block.configuration["status"] = "S"
                block.configuration["message"] = "Scheduled"
            else:
                block.configuration["status"] = "X"
                block.configuration["message"] = "Expired"
        block_groups = [queue_blocks]

    elif queue is None and len(block_groups) > 0:
        # Schedule the blocks in the catalog, but don't write to the queue
        logger.info("No queue provided")
    elif add_only:
        logger.error("Missing catalog or queue for option -ao/--add-only, exiting")
        return
    else:
        logger.error("No catalog or queue provided")
        return

    # Add sched_time to all blocks
    for i in range(len(block_groups)):
        for j in range(len(block_groups[i])):
            block_groups[i][j].configuration["sched_time"] = sched_time

    # Constraints
    logger.info("Defining global constraints")
    global_constraints = [
        astroplan.AtNightConstraint(max_solar_altitude=max_altitude * u.deg),
        astroplan.AltitudeConstraint(min=elevation * u.deg),
        astroplan.AirmassConstraint(max=airmass, boolean_constraint=False),
        astroplan.MoonSeparationConstraint(min=moon_separation * u.deg),
    ]

    # Transitioner
    logger.info("Defining transitioner")
    if instrument_reconfig_times == {}:
        instrument_reconfig_times = None
    transitioner = astroplan.Transitioner(
        slew_rate, instrument_reconfig_times=instrument_reconfig_times
    )

    # Scheduler
    if scheduler == ("", ""):
        logger.info("Using default scheduler: astroplan.PriorityScheduler")
        schedule_handler = astroplan.PriorityScheduler(
            constraints=global_constraints,
            observer=observatory,
            transitioner=transitioner,
            gap_time=gap_time * u.second,
            time_resolution=resolution * u.second,
        )
    else:
        logger.info(f"Using custom scheduler: {scheduler[0]}")
        spec = importlib.util.spec_from_file_location(
            scheduler[0].split("/")[-1].split(".")[0], scheduler[0]
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        schedule_handler_class = getattr(module, scheduler[1])
        if not astroplan.Scheduler in schedule_handler_class.__bases__:
            logger.error(
                f"Scheduler {scheduler[0]} does not inherit from astroplan.Scheduler."
            )
            return
        schedule_handler = schedule_handler_class(
            constraints=global_constraints,
            observer=observatory,
            transitioner=transitioner,
            gap_time=gap_time * u.second,
            time_resolution=resolution * u.second,
        )

    logger.info("Scheduling ObservingBlocks")
    for i in tqdm.tqdm(range(len(block_groups))):
        logger.debug("Block group %i of %i" % (i + 1, len(block_groups)))
        schedule_handler(block_groups[i], schedule)

    # Flatten block_groups for comparison with scheduled ObservingBlocks
    all_blocks = [block for block_group in block_groups for block in block_group]

    # Get scheduled ObservingBlocks
    scheduled_blocks = [
        slot.block for slot in schedule.slots if hasattr(slot.block, "target")
    ]
    transition_blocks = [
        slot.block
        for slot in schedule.slots
        if slot.block is not None and not hasattr(slot.block, "target")
    ]

    unscheduled_slots = [slot for slot in schedule.slots if slot.block is None]

    # Update ephem for non-sidereal targets, update object types, set filenames
    for block_number, block in enumerate(scheduled_blocks):
        block.configuration["status"] = "S"
        block.configuration["message"] = "Scheduled"

        if (
            block.configuration["pm_ra_cosdec"].value != 0
            or block.configuration["pm_dec"].value != 0
        ) and block.name != "":
            logger.info("Updating ephemeris for '%s' at scheduled time" % block.name)
            try:
                ephemerides = mpc.MPC.get_ephemeris(
                    target=block.name,
                    location=observatory.location,
                    start=block.start_time,
                    number=1,
                    proper_motion="sky",
                )
                new_ra = ephemerides["RA"][0]
                new_dec = ephemerides["Dec"][0]
                block.target = astroplan.FixedTarget(
                    coord.SkyCoord(ra=new_ra, dec=new_dec)
                )
                block.configuration["pm_ra_cosdec"] = (
                    ephemerides["dRA cos(Dec)"][0] * u.arcsec / u.hour
                )
                block.configuration["pm_dec"] = (
                    ephemerides["dDec"][0] * u.arcsec / u.hour
                )
            except Exception as e1:
                try:
                    logger.warning(
                        f"Failed to find proper motions for {block.name}, trying to find proper motions using astropy.coordinates.get_body"
                    )
                    pos_l = coord.get_body(
                        block.name,
                        block.start_time - 10 * u.minute,
                        location=observatory.location,
                    )
                    pos_m = coord.get_body(
                        block.name,
                        (block.start_time + block.end_time) / 2,
                        location=location,
                    )
                    pos_h = coord.get_body(
                        block.name,
                        block.end_time + 10 * u.minute,
                        location=observatory.location,
                    )
                    new_ra = pos_m.ra
                    new_dec = pos_m.dec
                    block.target = astroplan.FixedTarget(
                        coord.SkyCoord(ra=new_ra, dec=new_dec)
                    )
                    block.configuration["pm_ra_cosdec"] = (
                        (
                            pos_h.ra * np.cos(pos_h.dec.rad)
                            - pos_l.ra * np.cos(pos_l.dec.rad)
                        )
                        / (pos_h.obstime - pos_l.obstime)
                    ).to(u.arcsec / u.hour)
                    block.configuration["pm_dec"] = (
                        (pos_h.dec - pos_l.dec) / (pos_h.obstime - pos_l.obstime)
                    ).to(u.arcsec / u.hour)
                except Exception as e2:
                    logger.warning(
                        f"Failed to find proper motions for {block.name}, keeping old ephemerides"
                    )

        if block.configuration["filename"] == "":
            block.configuration["filename"] = name_format.format(
                index=block_number,
                target=block.target.to_string("hmsdms")
                .replace(" ", "_")
                .replace(".", "-"),
                start_time=block.start_time.isot.replace(":", "-").split(".")[0],
                end_time=block.end_time.isot.replace(":", "-").split(".")[0],
                duration="%i" % block.duration.to(u.second).value,
                ra=block.target.ra.to_string(
                    sep="hms", unit="hourangle", precision=2, pad=True
                )
                .replace(" ", "_")
                .replace(".", "-"),
                dec=block.target.dec.to_string(
                    sep="dms", precision=3, alwayssign=True, pad=True
                )
                .replace(" ", "_")
                .replace(".", "-"),
                observer=block.configuration["observer"],
                code=block.configuration["code"],
                title=block.configuration["title"],
                type=block.configuration["type"],
                backend=block.configuration["backend"],
                exposure=block.configuration["exposure"],
                nexp=block.configuration["nexp"],
                repositioning=block.configuration["repositioning"],
                shutter_state=block.configuration["shutter_state"],
                readout=block.configuration["readout"],
                binning=block.configuration["binning"],
                frame_position=block.configuration["frame_position"],
                frame_size=block.configuration["frame_size"],
                pm_ra_cosdec=block.configuration["pm_ra_cosdec"],
                pm_dec=block.configuration["pm_dec"],
                comment=block.configuration["comment"],
                sch=block.configuration["sch"],
                ID=block.configuration["ID"],
                status=block.configuration["status"],
                message=block.configuration["message"],
                sched_time=block.configuration["sched_time"],
            )

    # Report unscheduled or invalid blocks, and report sch files that were
    # complete, partially scheduled, or not scheduled at all

    # First find unscheduled blocks
    # TODO: report reason for unscheduled blocks, use is_observable
    unscheduled_blocks = [
        block
        for block in all_blocks
        if block.configuration["ID"]
        not in [b.configuration["ID"] for b in scheduled_blocks]
    ]

    # TODO: Fix this condition
    if type(observatory) is Observatory:
        validated_blocks = schedtab.validate(scheduled_blocks, observatory=observatory)
    else:
        validated_blocks = schedtab.validate(scheduled_blocks)

    # Then find invalid blocks
    invalid_blocks = [
        block for block in validated_blocks if block.configuration["status"] == "I"
    ]

    if len(unscheduled_blocks) > 0 or len(invalid_blocks) > 0:
        logger.warning("There are unscheduled or invalid blocks")
        logger.warning("Unscheduled blocks:")
        for block in unscheduled_blocks:
            logger.warning(
                f"""
            ID = {block.configuration["ID"]}
            Title = {block.configuration["title"]}
            Target = {block.target.to_string("hmsdms")}
            sch = {block.configuration["sch"]}
            Status = {block.configuration["status"]}

                """
            )
        logger.warning("Invalid blocks:")
        for block in invalid_blocks:
            logger.warning(
                f"""
            ID = {block.configuration["ID"]}
            Title = {block.configuration["title"]}
            Target = {block.target.to_string("hmsdms")}
            sch = {block.configuration["sch"]}
            Status = {block.configuration["status"]}
            Message = {block.configuration["message"]}

                """
            )
        if not yes:
            if click.confirm("Continue?"):
                pass
            else:
                return

    # Scheduled, unscheduled, invalid, transition blocks and unscheduled slots

    # Blocks to be placed in an execution schedule
    exec_blocks = scheduled_blocks + transition_blocks + invalid_blocks
    if queue is None:
        logger.info(
            "No queue provided, including unscheduled blocks in execution schedule. Note that these blocks will not actually be executed."
        )
        exec_blocks += unscheduled_blocks
    elif queue is not None:
        exec_table = schedtab.blocks_to_table(exec_blocks)

        # Blocks to be placed back in the queue
        queue_blocks = scheduled_blocks + unscheduled_blocks

        # If queue and catalog are not none, this carries unscheduled blocks
        # and all other blocks marked as expired "X"
        if previously_queued_blocks is not None:
            queue_blocks += previously_queued_blocks

        queue_table = schedtab.blocks_to_table(queue_blocks)

    exec_blocks = exec_blocks + unscheduled_slots
    exec_table = schedtab.blocks_to_table(exec_blocks)

    # Write the schedule to file
    logger.info("Writing schedule to file")
    if filename is None or telrun:
        first_time = np.min(exec_table["start_time"]).strftime("%Y-%m-%dT%H-%M-%S")
        filename = "telrun_" + first_time + ".ecsv"

    write_queue = False
    if telrun:
        write_queue = True
        try:
            path = os.environ.get("TELRUN_EXECUTE")
            logger.info(
                "-t/--telrun flag set, writing schedule to %s from $TELRUN_EXECUTE environment variable"
                % path
            )
        except:
            try:
                path = os.environ.get("TELHOME") + "/schedules/execute/"
                logger.info(
                    "-t/--telrun flag set, writing schedule to %s from $TELHOME environment variable"
                    % path
                )
            except:
                path = os.getcwd() + "/schedules/"
                logger.info(
                    "-t/--telrun flag set, writing schedule to %s from current working directory"
                    % path
                )
        if not os.path.isdir(path):
            path = os.getcwd() + "/"
            logger.warning(
                f"Path {path} does not exist, writing to current working directory instead: {path}"
            )
    else:
        write_queue = False
        path = os.getcwd() + "/"
        logger.info("-t/--telrun flag not set")
        logger.info("If queue was provided, it will not be written to file")

    if not telrun:
        write_fname = filename
    else:
        write_fname = path + filename
    exec_table.write(write_fname, overwrite=True, format="ascii.ecsv")

    # If a queue was passed, update the queue if --telrun is set
    # and the file was written to the expected location
    if queue is not None:
        if write_queue:
            logger.info("Writing queue to file")
            queue_table.write(queue, overwrite=True, format="ascii.ecsv")
        else:
            logger.info("Not writing queue to file")
    else:
        logger.info("No queue provided")

    # TODO: Report observing statistics (time used, transition, unscheduled, etc.)
    # reports.pre_exec_report(exec_table)

    # Plot the schedule
    match plot:
        case 1:  # Gantt chart
            logger.info("Plotting schedule as a Gantt chart")
            fig, ax = plot_schedule_gantt(schedule, observatory)
            return exec_table, fig, ax
        case 2:  # Plot by target w/ altitude
            pass
            # logger.info("Plotting schedule by target with airmass")
            # ax = astroplan.plots.plot_schedule_airmass(schedule)
            # plt.legend()
            # return exec_table, fig, ax
        case 3:  # Sky chart
            logger.info("Plotting schedule on a sky chart")
            ax = plot_schedule_sky(schedule, observatory)
            return exec_table, fig, ax
        case _:
            logger.info("No plot requested")

    return exec_table


@click.command(
    epilog="""Check out the documentation at
               https://pyscope.readthedocs.io/en/latest/
               for more information."""
)
@click.argument(
    "schedule_table",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False, readable=True),
)
@click.argument(
    "observatory",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False, readable=True),
)
@click.version_option()
def plot_schedule_gantt_cli(schedule_table, observatory):
    if type(schedule_table) is not table.Table:
        schedule_table = table.Table.read(schedule_table, format="ascii.ecsv")

    if type(observatory) is str:
        obs_cfg = configparser.ConfigParser()
        obs_cfg.read(observatory)
        observatory = astroplan.Observer(
            location=coord.EarthLocation(
                lon=obs_cfg.get("site", "longitude"),
                lat=obs_cfg.get("site", "latitude"),
            )
        )
        obs_lon = observatory.location.lon
        obs_lat = observatory.location.lat
    elif type(observatory) is Observatory:
        obs_lon = observatory.observatory_location.lon
        obs_lat = observatory.observatory_location.lat
    elif type(observatory) is astroplan.Observer:
        obs_lon = observatory.location.lon
        obs_lat = observatory.location.lat
    else:
        logger.error(
            "Observatory must be, a string, Observatory object, or astroplan.Observer object."
        )
        return
    location = coord.EarthLocation(lon=obs_lon, lat=obs_lat)

    tz = timezonefinder.TimezoneFinder().timezone_at(lng=obs_lon.deg, lat=obs_lat.deg)
    tz = zoneinfo.ZoneInfo(tz)
    date = np.min(schedule_table["start_time"]).datetime
    t0 = astrotime.Time(
        datetime.datetime(date.year, date.month, date.day, 12, 0, 0, tzinfo=tz)
    )
    t1 = t0 + 1 * u.day

    obscodes = list(np.unique(schedule_table["code"]))

    fig, ax = plt.subplots(1, 1, figsize=(12, len(obscodes) * 0.75))
    mdates.set_epoch(t0.strftime("%Y-%m-%dT%H:%M:%S"))

    for i in range(len(obscodes)):
        plot_blocks = [
            block for block in schedule_table if block["code"] == obscodes[i]
        ]

        for block in plot_blocks:
            start_time = block["start_time"]
            end_time = block["end_time"]
            length_minutes = int(np.ceil((end_time - start_time).sec / 60))
            times = (
                start_time
                + np.linspace(0, length_minutes, length_minutes, endpoint=True)
                * u.minute
            )
            airmass = []
            for t in times:
                obj = coord.SkyCoord(
                    block["target"], obstime=t, location=location
                ).transform_to("altaz")
                airmass.append(utils.airmass(obj.alt.rad))

            ax.scatter(
                times.datetime,
                i * np.ones(len(times)),
                lw=0,
                marker="s",
                c=airmass,
                cmap=ccm.batlow,
                vmin=1,
                vmax=3,
            )

            scatter = ax.scatter(
                times.datetime,
                len(obscodes) * np.ones(len(times)),
                lw=0,
                marker="s",
                c=airmass,
                cmap=ccm.batlow,
                vmin=1,
                vmax=3,
            )

    obscodes.append("All")

    twilight_times = [
        t0,
        astrotime.Time(observatory.sun_set_time(t0, which="next"), scale="utc"),
        astrotime.Time(
            observatory.twilight_evening_civil(t0, which="next"), scale="utc"
        ),
        astrotime.Time(
            observatory.twilight_evening_nautical(t0, which="next"), scale="utc"
        ),
        astrotime.Time(
            observatory.twilight_evening_astronomical(t0, which="next"), scale="utc"
        ),
        astrotime.Time(
            observatory.twilight_morning_astronomical(t0, which="next"), scale="utc"
        ),
        astrotime.Time(
            observatory.twilight_morning_nautical(t0, which="next"), scale="utc"
        ),
        astrotime.Time(
            observatory.twilight_morning_civil(t0, which="next"), scale="utc"
        ),
        astrotime.Time(observatory.sun_rise_time(t0, which="next"), scale="utc"),
        t1,
    ]
    opacities = [0.8, 0.6, 0.4, 0.2, 0, 0.2, 0.4, 0.6, 0.8]
    for i in range(len(twilight_times) - 1):
        ax.axvspan(
            twilight_times[i].datetime,
            twilight_times[i + 1].datetime,
            color="grey",
            alpha=opacities[i],
        )

    ax.set_xlabel(
        "Time beginning %s [UTC]"
        % (twilight_times[1] - 0.5 * u.hour).strftime("%Y-%m-%d")
    )
    ax.set_xlim(
        [
            (twilight_times[1] - 0.5 * u.hour).datetime,
            (twilight_times[-2] + 0.5 * u.hour).datetime,
        ]
    )
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.xaxis.set_tick_params(rotation=45)

    ax1 = ax.twiny()
    ax1.set_xlim(ax.get_xlim())
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=tz))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=1, tz=tz))
    ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax1.xaxis.set_tick_params(rotation=45)
    ax1.set_xlabel("Observatory Local Time (%s)" % tz)

    ax.set_ylabel("Observer Code")
    ax.set_ylim([len(obscodes) - 0.5, 0.5])
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(len(obscodes))))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(obscodes))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_ticks([1, 1.5, 2, 2.5, 3])
    cbar.set_label("Airmass", rotation=270, labelpad=20)

    """ax.set_title(
        "Observing Schedule: %s"
        % t0.strftime("%Y-%m-%d"), fontsize=14
    )"""
    ax.grid(linestyle=":", color="black")

    fig.set_facecolor("white")
    fig.set_dpi(300)

    return fig, ax


@click.command(
    epilog="""Check out the documentation at
               https://pyscope.readthedocs.io/en/latest/
               for more information."""
)
@click.argument(
    "schedule_table",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False, readable=True),
)
@click.argument(
    "observatory",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False, readable=True),
)
@click.version_option()
def plot_schedule_sky_cli(schedule_table, observatory):
    if type(schedule_table) is not table.Table:
        schedule_table = table.Table.read(schedule_table, format="ascii.ecsv")

    if type(observatory) is str:
        obs_cfg = configparser.ConfigParser()
        obs_cfg.read(observatory)
        observatory = astroplan.Observer(
            location=coord.EarthLocation(
                lon=obs_cfg.get("site", "longitude"),
                lat=obs_cfg.get("site", "latitude"),
            )
        )
        obs_lon = observatory.location.lon
        obs_lat = observatory.location.lat
    elif type(observatory) is Observatory:
        obs_lon = observatory.observatory_location.lon
        obs_lat = observatory.observatory_location.lat
    elif type(observatory) is astroplan.Observer:
        obs_lon = observatory.location.lon
        obs_lat = observatory.location.lat
    else:
        logger.error(
            "Observatory must be, a string, Observatory object, or astroplan.Observer object."
        )
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={"projection": "polar"})
    for i, row in enumerate(schedule_table):
        if row["name"] == "TransitionBlock" or row["name"] == "EmptyBlock":
            continue
        ax = astroplan_plots.plot_sky(
            astroplan.FixedTarget(row["target"]),
            observatory,
            row["start_time"],
            ax=ax,
            style_kwargs={
                "label": row["target"].to_string("hmsdms"),
            },
        )
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    ax.legend(*zip(*unique), loc=(1.1, 0))

    fig.set_facecolor("white")
    fig.set_dpi(300)

    return fig, ax


schedtel = schedtel_cli.callback
plot_schedule_gantt = plot_schedule_gantt_cli.callback
plot_schedule_sky = plot_schedule_sky_cli.callback
