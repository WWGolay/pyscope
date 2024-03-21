import logging
import os
import platform
import shutil
from pathlib import Path

import click

from . import init_queue

logger = logging.getLogger(__name__)


@click.command(
    epilog="""Check out the documentation at
               https://pyscope.readthedocs.io/en/latest/
               for more information."""
)
@click.argument(
    "path",
    type=click.Path(resolve_path=True),
    default="./telhome/",
    required=False,
)
@click.version_option()
def init_telrun_dir_cli(path="./telhome/"):
    """
    Initialize a telrun home directory at PATH.

    PATH defaults to ./telhome/ if not specified. If PATH already exists,
    the command will abort. Otherwise, the following directory structure
    will be created::

        PATH/
        |---start_telrun            # A shortcut script to start a TelrunOperator mainloop
        |---config/
        |   |---logging.cfg         # Optional
        |   |---notifications.cfg   # Optional
        |   |---observatory.cfg
        |   |---sync.cfg            # Optional
        |   |---telrun.cfg
        |
        |---images/                 # Images captured by TelrunOperator are saved here
        |   |---im1.fts
        |   |---autofocus/
        |   |---calibrations/
        |   |   |---masters/        # Master calibration images, created by cal scripts
        |   |   |   |---YYYY-MM-DD/     # Timestamped calibration sets
        |   |   |---YYYY-MM-DD/     # Individual calibration images
        |   |---raw_archive/        # Optional, but recommended. Backup of raw images
        |   |---recenter/
        |   |---reduced/            # Optional. Where reduced images are saved if not done in-place
        |
        |---logs/                   # TelrunOperator logs and auto-generated reports are saved here if requested
        |   |---telrun_status.json  # A JSON file containing the current status of TelrunOperator
        |
        |---schedules/
        |   |---aaa000.sch          # A schedule file for schedtel
        |   |---schedules.cat       # A catalog of sch files to be scheduled
        |   |---queue.ecsv          # A queue of unscheduled blocks
        |   |---completed/          # sch files that have been parsed and scheduled
        |   |---execute/            # Where schedtel puts schedules to be executed
        |
        |---tmp/                    # Temporary files are saved here, not synced by default

    Parameters
    ----------
    path : `str`, optional, default="./telhome/"
        The path to the telrun home directory to be created

    Returns
    -------
    `None`

    See Also
    --------
    pyscope.telrun.TelrunOperator : The main class for running a telescope
    pyscope.telrun.schedtel : Schedule a sch file
    pyscope.telrun.synctools.sync : Sync a local telrun home directory with a remote one

    """

    logger.debug(f"init_telrun_dir(path={path})")

    path = Path(path).resolve()

    logger.info("Initializing a telrun home directory at %s" % path)

    if not path.exists():
        logger.info(f"Creating directory {path}")
        path.mkdir()
    elif not path.is_dir():
        raise ValueError(f"{path} is not a directory")

    if not (path / "config").exists():
        logger.info("Creating config directory")
        (path / "config").mkdir()

    cfg_templates = Path(os.path.dirname(__file__)).resolve() / "../bin/cfg_templates/"
    bin_scripts = Path(os.path.dirname(__file__)).resolve() / "../bin/scripts/"

    logger.info("Creating empty config files")

    if not (path / "config/telrun.cfg").exists():
        shutil.copyfile(
            cfg_templates / "telrun-empty.cfg",
            path / "config/telrun.cfg",
        )
    else:
        logger.warning("telrun.cfg already exists, skipping")

    if not (path / "config/observatory.cfg").exists():
        shutil.copyfile(
            cfg_templates / "observatory-empty.cfg",
            path / "config/observatory.cfg",
        )
    else:
        logger.warning("observatory.cfg already exists, skipping")

    if not (path / "config/logging.cfg").exists():
        shutil.copyfile(
            cfg_templates / "logging-empty.cfg",
            path / "config/logging.cfg",
        )
    else:
        logger.warning("logging.cfg already exists, skipping")

    if not (path / "config/notifications.cfg").exists():
        shutil.copyfile(
            cfg_templates / "notifications-empty.cfg",
            path / "config/notifications.cfg",
        )
    else:
        logger.warning("notifications.cfg already exists, skipping")

    if not (path / "config/sync.cfg").exists():
        shutil.copyfile(
            cfg_templates / "sync-empty.cfg",
            path / "config/sync.cfg",
        )
    else:
        logger.warning("sync.cfg already exists, skipping")

    logger.info("Creating images directory")

    if not (path / "images").exists():
        (path / "images").mkdir()
    else:
        logger.warning("%s already exists, skipping" % (path / "images"))

    if not (path / "images" / "autofocus").exists():
        (path / "images" / "autofocus").mkdir()
    else:
        logger.warning("%s already exists, skipping" % (path / "images" / "autofocus"))

    if not (path / "images" / "calibrations").exists():
        (path / "images" / "calibrations").mkdir()
    else:
        logger.warning(
            "%s already exists, skipping" % (path / "images" / "calibrations")
        )

    if not (path / "images" / "calibrations" / "masters").exists():
        (path / "images" / "calibrations" / "masters").mkdir()
    else:
        logger.warning(
            "%s already exists, skipping"
            % (path / "images" / "calibrations" / "masters")
        )

    if not (path / "images" / "raw_archive").exists():
        (path / "images" / "raw_archive").mkdir()
    else:
        logger.warning(
            "%s already exists, skipping" % (path / "images" / "raw_archive")
        )

    if not (path / "images" / "recenter").exists():
        (path / "images" / "recenter").mkdir()
    else:
        logger.warning("%s already exists, skipping" % (path / "images" / "recenter"))

    if not (path / "images" / "reduced").exists():
        (path / "images" / "reduced").mkdir()
    else:
        logger.warning("%s already exists, skipping" % (path / "images" / "reduced"))

    logger.info("Creating logs directory")
    if not (path / "logs").exists():
        (path / "logs").mkdir()
    else:
        logger.warning("%s already exists, skipping" % (path / "logs"))

    logger.info("Creating schedules directory")
    if not (path / "schedules").exists():
        (path / "schedules").mkdir()
    else:
        logger.warning("%s already exists, skipping" % (path / "schedules"))

    if not (path / "schedules" / "completed").exists():
        (path / "schedules" / "completed").mkdir()
    else:
        logger.warning(
            "%s already exists, skipping" % (path / "schedules" / "completed")
        )

    if not (path / "schedules" / "execute").exists():
        (path / "schedules" / "execute").mkdir()
    else:
        logger.warning("%s already exists, skipping" % (path / "schedules" / "execute"))

    if not (path / "schedules" / "schedules.cat").exists():
        (path / "schedules" / "schedules.cat").touch()
    else:
        logger.warning(
            "%s already exists, skipping" % (path / "schedules" / "schedules.cat")
        )

    # TODO: call init_queue here
    # init_queue(path / "schedules" / "queue.ecsv")

    logger.info("Creating tmp directory")
    if not (path / "tmp").exists():
        (path / "tmp").mkdir()
    else:
        logger.warning("%s already exists, skipping" % (path / "tmp"))

    logger.info("Copying default shortcut startup script")
    if platform.system() == "Windows":
        shutil.copyfile(
            bin_scripts / "start_telrun.bat",
            path / "start_telrun.bat",
        )
    else:
        shutil.copyfile(
            bin_scripts / "start_telrun",
            path / "start_telrun",
        )

    logger.info("Done")


init_telrun_dir = init_telrun_dir_cli.callback
