import atexit
import configparser
import glob
import json
import logging
import os
import shutil
import threading
import time
import tkinter as tk
import tkinter.ttk as ttk
from io import StringIO
from pathlib import Path
from tkinter import font

import astroplan
import tksheet
from astropy import coordinates as coord
from astropy import table
from astropy import time as astrotime
from astropy import units as u
from astroquery import mpc

from ..observatory import Observatory
from . import TelrunException, init_telrun_dir, schedtab

logger = logging.getLogger(__name__)


class TelrunOperator:
    def __init__(self, telhome="./", gui=True, **kwargs):
        """
        The main class for robotic telescope operation.

        TelrunOperator is responsible for connecting to the observatory hardware
        and executing observing schedules. It has various parameters that modify
        its behavior for testing and debugging purposes. It also has a GUI that
        can be used to monitor the status of the observatory hardware and the
        execution of observing schedules. In addition to the GUI, it will write
        a number of status variables to a JSON file that can be parsed by other
        programs, such as a web server that displays the status of the observatory
        hardware and the execution of schedules.

        Parameters
        ----------
        telhome : str, optional
            The path to the TelrunOperator home directory. The default is the current working directory.
            Telhome must have a specific directory structure, which is created if it does not exist. The
            directory structure and some relevant files are as follows::

                telhome/
                |---start_telrun            # A shortcut script to start a TelrunOperator mainloop
                |---config/
                |   |---logging.cfg         # Optional
                |   |---notifications.cfg   # Optional
                |   |---observatory.cfg
                |   |---syncfiles.cfg       # Optional
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

            .. note::
                For more info on the directory structure, see the `~pyscope.telrun.init_telrun_dir` command line tool,
                which can be used to create a TelrunOperator home directory with the correct structure. The `schedules/`
                directory structure is dictated by the `~pyscope.telrun.schedtel` function, which is used to schedule
                observing blocks. The `images/` directory structure is dictated by the `~pyscope.reduction` scripts,
                which are used for rapid image calibration and reduction.

        gui : bool, optional
            Whether to start the GUI. Default is True.

        **kwargs
            Keyword arguments to pass to the TelrunOperator constructor. More details in Other Parameters
            section below.

        Other Parameters
        ----------------
        TBD

        Raises
        ------
        TelrunException

        See Also
        --------
        TBD

        Notes
        -----
        TBD

        References
        ----------
        TBD

        Examples
        --------
        TBD


        """
        # Private attributes
        self._config = configparser.ConfigParser(allow_no_value=True)
        self._gui = gui
        self._execution_thread = None
        self._status_log_thread = None
        self._wcs_threads = []
        self._execution_event = threading.Event()
        self._status_event = threading.Event()
        self._status_log_update_event = threading.Event()

        # Read-only attributes with constructor arguments
        self._telhome = Path(telhome).resolve()
        self._queue_fname = None
        self._dome_type = None  # None, 'dome' or 'safety-monitor' or 'both'

        # More private attributes
        self._config_path = self._telhome / "config"
        self._schedules_path = self._telhome / "schedules"
        self._images_path = self._telhome / "images"
        self._logs_path = self._telhome / "logs"

        save_at_end = False
        if not self._config_path.exists():
            save_at_end = True

        # Read-only attributes with no constructor arguments
        self._schedule_fname = None
        self._schedule = None
        self._best_focus_result = None
        self._do_periodic_autofocus = True
        self._last_autofocus_time = astrotime.Time("2000-01-01T00:00:00", format="isot")
        self._skipped_block_count = 0
        self._current_block_index = None
        self._current_block = None
        self._previous_block = None
        self._next_block = None
        self._autofocus_status = ""
        self._camera_status = ""
        self._cover_calibrator_status = ""
        self._dome_status = ""
        self._filter_wheel_status = ""
        self._focuser_status = ""
        self._observing_conditions_status = ""
        self._rotator_status = ""
        self._safety_monitor_status = ""
        self._switch_status = ""
        self._telescope_status = ""
        self._wcs_status = ""

        # Public attributes with constructor arguments
        self._initial_home = True
        self._wait_for_sun = True
        self._max_solar_elev = -12
        self._check_safety_monitors = True
        self._wait_for_cooldown = True
        self._default_readout = 0
        self._check_block_status = True
        self._update_block_status = True
        self._write_to_status_log = True
        self._status_log_update_interval = 5  # seconds
        self._wait_for_block_start_time = True
        self._max_block_late_time = 60
        self._preslew_time = 60
        self._hardware_timeout = 120
        self._autofocus_filters = None
        self._autofocus_interval = 3600
        self._autofocus_initial = True
        self._autofocus_exposure = 5
        self._autofocus_midpoint = 0
        self._autofocus_nsteps = 5
        self._autofocus_step_size = 500
        self._autofocus_use_current_pointing = False
        self._autofocus_timeout = 180
        self._recenter_filters = None
        self._recenter_initial_offset_dec = 0
        self._recenter_check_and_refine = True
        self._recenter_max_attempts = 5
        self._recenter_tolerance = 3
        self._recenter_exposure = 10
        self._recenter_save_images = False
        self._recenter_save_path = self._images_path / "recenter"
        self._recenter_sync_mount = False
        self._recenter_timeout = 180
        self._wcs_filters = None
        self._wcs_timeout = 30

        # Load config file if there
        if (self._config_path / "telrun.cfg").exists():
            logger.info(
                "Using config file to initialize telrun: %s"
                % (self._config_path / "telrun.cfg")
            )
            try:
                self._config.read(self._config_path / "telrun.cfg")
            except:
                raise TelrunException(
                    "Could not read config file: %s" % self._config_path
                )
            self._queue_fname = self._config.get(
                "telrun",
                "queue_fname",
                fallback=self._queue_fname,
            )
            self._dome_type = self._config.get(
                "telrun", "dome_type", fallback=self._dome_type
            )
            self._initial_home = self._config.getboolean(
                "telrun", "initial_home", fallback=self._initial_home
            )
            self._wait_for_sun = self._config.getboolean(
                "telrun", "wait_for_sun", fallback=self._wait_for_sun
            )
            self._max_solar_elev = self._config.getfloat(
                "telrun", "max_solar_elev", fallback=self._max_solar_elev
            )
            self._check_safety_monitors = self._config.getboolean(
                "telrun", "check_safety_monitors", fallback=self._check_safety_monitors
            )
            self._wait_for_cooldown = self._config.getboolean(
                "telrun", "wait_for_cooldown", fallback=self._wait_for_cooldown
            )
            self._default_readout = self._config.getint(
                "telrun", "default_readout", fallback=self._default_readout
            )
            self._check_block_status = self._config.getboolean(
                "telrun", "check_block_status", fallback=self._check_block_status
            )
            self._update_block_status = self._config.getboolean(
                "telrun", "update_block_status", fallback=self._update_block_status
            )
            self._write_to_status_log = self._config.getboolean(
                "telrun", "write_to_status_log", fallback=self._write_to_status_log
            )
            self._status_log_update_interval = self._config.getfloat(
                "telrun",
                "status_log_update_interval",
                fallback=self._status_log_update_interval,
            )
            self._wait_for_block_start_time = self._config.getboolean(
                "telrun",
                "wait_for_block_start_time",
                fallback=self._wait_for_block_start_time,
            )
            self._max_block_late_time = self._config.getfloat(
                "telrun", "max_block_late_time", fallback=self._max_block_late_time
            )
            self._preslew_time = self._config.getfloat(
                "telrun", "preslew_time", fallback=self._preslew_time
            )
            self._hardware_timeout = self._config.getfloat(
                "telrun", "hardware_timeout", fallback=self._hardware_timeout
            )
            self._autofocus_filters = [
                f.strip()
                for f in self._config.get("autofocus", "autofocus_filters").split(",")
            ]
            self._autofocus_interval = self._config.getfloat(
                "autofocus", "autofocus_interval", fallback=self._autofocus_interval
            )
            self._autofocus_initial = self._config.getboolean(
                "autofocus", "autofocus_initial", fallback=self._autofocus_initial
            )
            self._autofocus_exposure = self._config.getfloat(
                "autofocus", "autofocus_exposure", fallback=self._autofocus_exposure
            )
            self._autofocus_midpoint = self._config.getfloat(
                "autofocus", "autofocus_midpoint", fallback=self._autofocus_midpoint
            )
            self._autofocus_nsteps = self._config.getint(
                "autofocus", "autofocus_nsteps", fallback=self._autofocus_nsteps
            )
            self._autofocus_step_size = self._config.getfloat(
                "autofocus", "autofocus_step_size", fallback=self._autofocus_step_size
            )
            self._autofocus_use_current_pointing = self._config.getboolean(
                "autofocus",
                "autofocus_use_current_pointing",
                fallback=self._autofocus_use_current_pointing,
            )
            self._autofocus_timeout = self._config.getfloat(
                "autofocus", "autofocus_timeout", fallback=self._autofocus_timeout
            )
            self._recenter_filters = [
                f.strip()
                for f in self._config.get("recenter", "recenter_filters").split(",")
            ]
            self._recenter_initial_offset_dec = self._config.getfloat(
                "recenter",
                "recenter_initial_offset_dec",
                fallback=self._recenter_initial_offset_dec,
            )
            self._recenter_check_and_refine = self._config.getboolean(
                "recenter",
                "recenter_check_and_refine",
                fallback=self._recenter_check_and_refine,
            )
            self._recenter_max_attempts = self._config.getint(
                "recenter",
                "recenter_max_attempts",
                fallback=self._recenter_max_attempts,
            )
            self._recenter_tolerance = self._config.getfloat(
                "recenter", "recenter_tolerance", fallback=self._recenter_tolerance
            )
            self._recenter_exposure = self._config.getfloat(
                "recenter", "recenter_exposure", fallback=self._recenter_exposure
            )
            self._recenter_save_images = self._config.getboolean(
                "recenter", "recenter_save_images", fallback=self._recenter_save_images
            )
            self._recenter_save_path = self._config.get(
                "recenter", "recenter_save_path", fallback=self._recenter_save_path
            )
            self._recenter_sync_mount = self._config.getboolean(
                "recenter", "recenter_sync_mount", fallback=self._recenter_sync_mount
            )
            self._recenter_timeout = self._config.getfloat(
                "recenter", "recenter_timeout", fallback=self._recenter_timeout
            )
            self._wcs_filters = [
                f.strip() for f in self._config.get("wcs", "wcs_filters").split(",")
            ]

            self._wcs_timeout = self._config.getfloat(
                "wcs", "wcs_timeout", fallback=self._wcs_timeout
            )
        else:
            self._config["telrun"] = {}
            self._config["autofocus"] = {}
            self._config["recenter"] = {}
            self._config["wcs"] = {}

        init_telrun_dir(self.telhome)

        # Parse observatory
        self._observatory = self._config_path / "observatory.cfg"
        self._observatory = kwargs.get("observatory", self._observatory)
        if type(self._observatory) is Path:
            logger.info(
                "Observatory is string, loading from config file and saving to config path"
            )
            self._observatory = Observatory(config_path=self._observatory)
            self.observatory.save_config(
                self._config_path / "observatory.cfg", overwrite=True
            )
        elif type(self._observatory) is Observatory:
            logger.info("Observatory is Observatory object, saving to config path")
            self.observatory.save_config(
                self._config_path / "observatory.cfg", overwrite=True
            )
        else:
            raise TelrunException(
                "observatory must be a string representing an observatory config file path or an Observatory object"
            )

        # Load kwargs
        qf = kwargs.get("queue_fname", self._queue_fname)
        if qf is not None and qf != "None" and qf != "none" and qf != "":
            self._queue_fname = self._schedules_path / kwargs.get(
                "queue_fname", self._queue_fname
            )
            if not self._queue_fname.exists():
                logger.warning("Queue file %s does not exist, setting to None" % qf)
                self._queue_fname = None
        else:
            self._queue_fname = None
        self._config["telrun"]["queue_fname"] = str(self._queue_fname)

        # Parse dome_type
        self._dome_type = kwargs.get("dome_type", self._dome_type)
        match self._dome_type:
            case None | "None" | "none":
                logger.info("dome_type is None, setting to None")
                self._dome_type = "None"
            case (
                "dome"
                | "safety-monitor"
                | "safety_monitor"
                | "safetymonitor"
                | "safety monitor"
                | "both"
            ):
                pass
            case _:
                raise TelrunException(
                    'dome_type must be None, "dome", "safety-monitor", "both", or "None"'
                )
        self._config["telrun"]["dome_type"] = str(self._dome_type)

        # Parse other kwargs
        self.initial_home = kwargs.get("initial_home", self._initial_home)
        self.wait_for_sun = kwargs.get("wait_for_sun", self._wait_for_sun)
        self.max_solar_elev = kwargs.get("max_solar_elev", self._max_solar_elev)
        self.check_safety_monitors = kwargs.get(
            "check_safety_monitors", self._check_safety_monitors
        )
        self.wait_for_cooldown = kwargs.get(
            "wait_for_cooldown", self._wait_for_cooldown
        )
        self.default_readout = kwargs.get("default_readout", self._default_readout)
        self.check_block_status = kwargs.get(
            "check_block_status", self._check_block_status
        )
        self.update_block_status = kwargs.get(
            "update_block_status", self._update_block_status
        )
        self.write_to_status_log = kwargs.get(
            "write_to_status_log", self._write_to_status_log
        )
        self.status_log_update_interval = kwargs.get(
            "status_log_update_interval", self._status_log_update_interval
        )
        self.wait_for_block_start_time = kwargs.get(
            "wait_for_block_start_time", self._wait_for_block_start_time
        )
        self.max_block_late_time = kwargs.get(
            "max_block_late_time", self._max_block_late_time
        )
        self.preslew_time = kwargs.get("preslew_time", self._preslew_time)
        self.hardware_timeout = kwargs.get("hardware_timeout", self._hardware_timeout)
        self.autofocus_interval = kwargs.get(
            "autofocus_interval", self._autofocus_interval
        )
        self.autofocus_initial = kwargs.get(
            "autofocus_initial", self._autofocus_initial
        )
        self.autofocus_filters = kwargs.get(
            "autofocus_filters", self._autofocus_filters
        )
        self.autofocus_exposure = kwargs.get(
            "autofocus_exposure", self._autofocus_exposure
        )
        self.autofocus_midpoint = kwargs.get(
            "autofocus_midpoint", self._autofocus_midpoint
        )
        self.autofocus_nsteps = kwargs.get("autofocus_nsteps", self._autofocus_nsteps)
        self.autofocus_step_size = kwargs.get(
            "autofocus_step_size", self._autofocus_step_size
        )
        self.autofocus_use_current_pointing = kwargs.get(
            "autofocus_use_current_pointing", self._autofocus_use_current_pointing
        )
        self.autofocus_timeout = kwargs.get(
            "autofocus_timeout", self._autofocus_timeout
        )
        self.recenter_filters = kwargs.get("recenter_filters", self._recenter_filters)
        self.recenter_initial_offset_dec = kwargs.get(
            "recenter_initial_offset_dec", self._recenter_initial_offset_dec
        )
        self.recenter_check_and_refine = kwargs.get(
            "recenter_check_and_refine", self._recenter_check_and_refine
        )
        self.recenter_max_attempts = kwargs.get(
            "recenter_max_attempts", self._recenter_max_attempts
        )
        self.recenter_tolerance = kwargs.get(
            "recenter_tolerance", self._recenter_tolerance
        )
        self.recenter_exposure = kwargs.get(
            "recenter_exposure", self._recenter_exposure
        )
        self.recenter_save_images = kwargs.get(
            "recenter_save_images", self._recenter_save_images
        )
        self.recenter_save_path = kwargs.get(
            "recenter_save_path", self._recenter_save_path
        )
        self.recenter_sync_mount = kwargs.get(
            "recenter_sync_mount", self._recenter_sync_mount
        )
        self.recenter_timeout = kwargs.get("recenter_timeout", self._recenter_timeout)
        self.wcs_filters = kwargs.get("wcs_filters", self._wcs_filters)
        self.wcs_timeout = kwargs.get("wcs_timeout", self._wcs_timeout)

        # Set filters up if None
        if self.autofocus_filters is None:
            self.autofocus_filters = self.observatory.filters
        if self.recenter_filters is None:
            self.recenter_filters = self.observatory.filters
        if self.wcs_filters is None:
            self.wcs_filters = self.observatory.filters

        # Verify filter restrictions appear in filter wheel
        if self.observatory.filter_wheel is not None:
            for filt in self.autofocus_filters:
                if filt in self.observatory.filters:
                    break
            else:
                raise TelrunException(
                    "At least one autofocus filter must be in filter wheel"
                )

            for filt in self.recenter_filters:
                if filt in self.observatory.filters:
                    break
            else:
                raise TelrunException(
                    "At least one recenter filter must be in filter wheel"
                )

            for filt in self.wcs_filters:
                if filt in self.observatory.filters:
                    break
            else:
                raise TelrunException("At least one WCS filter must be in filter wheel")

        # Register shutdown with atexit
        logger.debug("Registering observatory shutdown with atexit")
        atexit.register(self._terminate)
        logger.debug("Registered")

        # Open GUI if requested
        if self._gui:
            logger.info("Starting GUI")
            root = tk.Tk()
            root.tk.call("source", "../gui/themeSetup.tcl")
            root.tk.call("set_theme", "dark")
            # icon_photo = tk.PhotoImage(file='images/UILogo.png')
            # root.iconphoto(False, icon_photo)
            self._gui = _TelrunGUI(root, self)
            self._gui.mainloop()
            logger.info("GUI started")

        # Connect to observatory hardware
        logger.info("Attempting to connect to observatory hardware")
        self.observatory.connect_all()
        logger.info("Connected")
        self._autofocus_status = "Idle"
        self._camera_status = "Idle"
        if self.observatory.cover_calibrator is not None:
            self._cover_calibrator_status = "Idle"
        if self.observatory.dome is not None:
            self._dome_status = "Idle"
        if self.observatory.filter_wheel is not None:
            self._filter_wheel_status = "Idle"
        if self.observatory.focuser is not None:
            self._focuser_status = "Idle"
        if self.observatory.observing_conditions is not None:
            self._observing_conditions_status = "Idle"
        if self.observatory.rotator is not None:
            self._rotator_status = "Idle"
        if self.observatory.safety_monitor is not None:
            self._safety_monitor_status = "Idle"
        if self.observatory.switch is not None:
            self._switch_status = "Idle"
        self._telescope_status = "Idle"
        if self.observatory._wcs is not None:
            self._wcs_status = "Idle"

        if self.write_to_status_log:
            self.start_status_log_thread(interval=self.status_log_update_interval)

        if save_at_end:
            self.save_config("telrun.cfg")

    def save_config(self, filename):
        logger.debug("Saving config to %s" % filename)
        self.observatory.save_config(
            self._config_path / "observatory.cfg", overwrite=True
        )
        with open(self._config_path / filename, "w") as config_file:
            self._config.write(config_file)

    def mainloop(self):
        if self.observatory.observing_conditions is not None:
            logger.info("Starting the observing_conditions update thread...")
            self._observing_conditions_status = "Update thread running"
            self.observatory.start_observing_conditions_thread()
            logger.info("Started.")

        logger.info("Starting main operation loop...")
        init_load = True
        while True:
            # Check for new schedule
            logger.debug("Checking for new schedule...")
            potential_schedules = glob.glob(
                self.schedules_path + "telrun_????-??-??T??-??-??.ecsv"
            )
            if len(potential_schedules) < 1:
                logger.debug(
                    "No schedules in schedule directory. Waiting for new schedule..."
                )
                time.sleep(1)
                continue
            potential_times = astrotime.Time(
                [
                    f.split("/")[-1]
                    .split("telrun_")[1]
                    .split(".")[0]
                    .strptime("%Y-%m-%dT%H-%M-%S")
                    for f in potential_schedules
                ],
                format="datetime",
            )
            potential_times = potential_times[potential_times < astrotime.Time.now()]
            potential_times.sort()
            if len(potential_times) < 1:
                logger.debug(
                    "No schedules with start times before current time. Waiting for new schedule..."
                )
                time.sleep(1)
                continue
            fname = (
                "telrun_" + potential_times[-1].strftime("%Y-%m-%d_%H-%M-%S") + ".ecsv"
            )
            logger.debug("Newest schedule: %s" % fname)

            # Compare to current schedule filename
            if not os.path.exists(self.schedules_path + fname):
                logger.exception("Schedule file error, waiting for new schedule...")
                time.sleep(1)
                continue
            schedule = table.Table.read(
                self.schedules_path + fname, format="ascii.ecsv"
            )
            if len(schedule) < 1:
                logger.exception("Schedule file empty, waiting for new schedule...")
                time.sleep(1)
                continue
            if self.schedules_path / fname == self._schedule_fname:
                logger.debug("Schedule already loaded, waiting for new schedule...")
                time.sleep(1)
                continue
            else:
                init_load = False
            logger.info("New schedule detected!")

            if self._execution_thread is not None:
                logger.info("Terminating current schedule execution thread...")
                self._execution_event.set()
                self._execution_thread.join()

            logger.info("Clearing execution event...")
            self._execution_event.clear()

            logger.info("Starting new schedule execution thread...")
            self._execution_thread = threading.Thread(
                target=self.execute_schedule,
                args=(self.schedules_path / fname,),
                daemon=True,
                name="Telrun Schedule Execution Thread",
            )
            self._execution_thread.start()
            logger.info("Started.")

    def execute_schedule(self, schedule):
        logger.info("Executing schedule: %s" % schedule)
        sched_fname = None
        try:
            sched_fname = schedule
            schedule = table.Table.read(schedule, format="ascii.ecsv")
        except:
            if type(schedule) is not table.Table:
                logger.exception(
                    "schedule must be a path to an ECSV file or an astropy Table"
                )
                return

        # Schedule validation
        logger.info("Validating schedule...")
        try:
            schedtab.validate(schedule, self.observatory)
        except Exception as e:
            logger.exception(e)
            logger.exception("Schedule failed validation, exiting...")
            return

        if sched_fname is None:
            first_time = np.min(schedule["start_time"]).strftime("%Y-%m-%dT%H-%M-%S")
            sched_fname = self.schedules_path + "telrun_" + first_time + ".ecsv"
        self._schedule_fname = sched_fname
        self._schedule = schedule

        # Save back to file for any invalid blocks
        self._schedule.write(self._schedule_fname, format="ascii.ecsv", overwrite=True)

        # Sort schedule by start time
        self._schedule.sort("start_time")

        # Initial home?
        if self._initial_home and self.observatory.telescope.CanFindHome:
            logger.info("Finding telescope home...")
            self._telescope_status = "Homing"
            self.observatory.telescope.FindHome()
            self._telescope_status = "Idle"
            logger.info("Found.")

        # Wait for sunset?
        while (
            self.observatory.sun_altaz()[0] > self.max_solar_elev and self.wait_for_sun
        ):
            logger.info(
                "Sun altitude: %.3f degs (above limit of %s), waiting 60 seconds"
                % (self.observatory.sun_altaz()[0], self.max_solar_elev)
            )
            time.sleep(60)
        logger.info(
            "Sun altitude: %.3f degs (below limit of %s), continuing..."
            % (self.observatory.sun_altaz()[0], self.max_solar_elev)
        )

        # Either open dome or check if open
        match self._dome_type:
            case "dome":
                if self.observatory.dome is not None:
                    if self.observatory.dome.CanSetShutter:
                        logger.info("Opening the dome shutter...")
                        self._dome_status = "Opening shutter"
                        self.observatory.dome.OpenShutter()
                        self._dome_status = "Idle"
                        logger.info("Opened.")
                    if self.observatory.dome.CanFindHome:
                        self._dome_status = "Homing"
                        logger.info("Finding the dome home...")
                        self.observatory.dome.FindHome()
                        self._dome_status = "Idle"
                        logger.info("Found.")
            case (
                "safety-monitor" | "safety_monitor" | "safetymonitor" | "safety monitor"
            ):
                logger.info("Designating first safety monitor state as dome...")
                if self.observatory.safety_monitor is not None:
                    status = False
                    while not status:
                        if self.observatory.safety_monitor is not (iter, tuple, list):
                            status = self.observatory.safety_monitor.IsSafe
                        else:
                            status = self.observatory.safety_monitor[0].IsSafe
                        logger.info("Safety monitor status: %s" % status)
                        logger.info("Waiting for safety monitor to be safe...")
                        time.sleep(10)
                    logger.info("Safety monitor indicates safe, continuing...")
            case "both":
                logger.info("Checking first safety monitor status...")
                if self.observatory.safety_monitor is not None:
                    status = False
                    while not status:
                        if self.observatory.safety_monitor is not (iter, tuple, list):
                            status = self.observatory.safety_monitor.IsSafe
                        else:
                            status = self.observatory.safety_monitor[0].IsSafe
                        logger.info("Safety monitor status: %s" % status)
                        logger.info("Waiting for safety monitor to be safe...")
                        time.sleep(10)
                    logger.info("Safety monitor indicates safe, continuing...")
                else:
                    logger.info("Safety monitor not found, continuing...")

                logger.info("Checking dome status...")
                if self.observatory.dome is not None:
                    if self.observatory.dome.CanSetShutter:
                        logger.info("Opening the dome shutter...")
                        self._dome_status = "Opening shutter"
                        self.observatory.dome.OpenShutter()
                        self._dome_status = "Idle"
                        logger.info("Opened.")
                    if self.observatory.dome.CanFindHome:
                        logger.info("Finding the dome home...")
                        self._dome_status = "Homing"
                        self.observatory.dome.FindHome()
                        self._dome_status = "Idle"
                        logger.info("Found.")

        # Wait for cooler?
        if self.wait_for_cooldown:
            while (
                self.observatory.camera.CCDTemperature
                > self.observatory.cooler_setpoint + self.observatory.cooler_tolerance
                and self.wait_for_cooldown
            ):
                logger.info(
                    "CCD temperature: %.3f degs (above limit of %.3f with %.3f tolerance), waiting for 10 seconds"
                    % (
                        self.observatory.camera.CCDTemperature,
                        self.observatory.cooler_setpoint,
                        self.observatory.cooler_tolerance,
                    )
                )
                self._camera_status = "Cooling"
                time.sleep(10)
            logger.info(
                "CCD temperature: %.3f degs (below limit of %.3f with %.3f tolerance), continuing..."
                % (
                    self.observatory.camera.CCDTemperature,
                    self.observatory.cooler_setpoint,
                    self.observatory.cooler_tolerance,
                )
            )
        self._camera_status = "Idle"

        # Initial autofocus?
        if self.autofocus_interval < 0:
            self._do_periodic_autofocus = False

        if self.autofocus_initial and self.do_periodic_autofocus:
            self._last_autofocus_time = (
                astrotime.Time.now() - self.autofocus_interval * u.second - 1 * u.second
            )

        # Process blocks
        for block_index, block in enumerate(self._schedule):
            if not self._execution_event.is_set():
                logger.info(
                    "Processing block %i of %i" % (block_index + 1, len(self._schedule))
                )
                logger.info(block)

                if block_index != 0:
                    self._previous_block_index = block_index - 1
                    self._previous_block = self._schedule[self._previous_block_index]
                else:
                    self._previous_block_index = None
                    self._previous_block = None

                if block_index != len(self._schedule) - 1:
                    self._next_block_index = block_index + 1
                    self._next_block = self._schedule[self._next_block_index]
                else:
                    self._next_block_index = None
                    self._next_block = None

                self._current_block_index = block_index
                status, message, block = self.execute_block(block)

                if status != "C":
                    self._skipped_block_count += 1

                # Update block status
                if self.update_block_status:
                    self._schedule[block_index] = block
                    self._schedule.write(
                        self._schedule_fname, format="ascii.ecsv", overwrite=True
                    )

                    # Update queue if it exists
                    if self.queue_fname is not None:
                        queue = table.Table.read(self.queue_fname, format="ascii.ecsv")
                        queue_idx = np.where(queue["id"] == block["id"])[0]
                        if len(queue_idx) > 0:
                            queue[queue_idx] = block
                        else:
                            logger.info(
                                "Block %s not found in queue, appending to end..."
                                % block["id"]
                            )
                            queue = table.vstack([queue, block])
                        queue.write(
                            self.queue_fname, format="ascii.ecsv", overwrite=True
                        )

            else:
                logger.info(
                    "Execution event has been set, exiting this schedule execution session..."
                )
                # TODO: Handle leaving for an interruption b/c of alert and returning to the same schedule
                break

        logger.info("Schedule execution complete.")

        """
        # Generate summary report if schedule completed
        logger.info("Generating summary report")
        summary_report(self.telpath+'/schedules/telrun.sls', self.telpath+'/logs/'+
            self._schedule[0].start_time.datetime.strftime('%m-%d-%Y')+'_telrun-report.txt')"""

        logger.info("Resetting status variables...")
        self._skipped_block_count = 0
        self._previous_block = None
        self._next_block = None
        self._schedule_fname = None
        self._schedule = None

        logger.info("Shutting observatory down...")
        self.observatory.shutdown()
        logger.info("Done.")

        return

    def execute_block(self, *args, **kwargs):
        # Check for block
        if len(args) > 1:
            logger.exception(
                "execute_block() takes 1 positional argument but %i were given"
                % len(args)
            )
            return
        elif len(args) == 1:
            block = args[0]
        elif len(args) == 0:
            block = kwargs.get("block", None)

        # Turn into table/row if necessary
        if type(block) is list:
            block = blocks_to_table(block)
        elif type(block) is astroplan.ObservingBlock:
            block = blocks_to_table([block])[0]
        elif type(block) is None:
            try:
                # TODO: Add kwargs to create block
                raise NotImplementedError
            except:
                logger.exception(
                    "Passed block is None and no kwargs were given, skipping this block"
                )
                return
        elif type(block) is not table.Row:
            logger.exception(
                "Passed block is not a list, ObservingBlock, or None, skipping this block. Returning from this function..."
            )
            return
        else:
            pass

        # Logging setup for writing to FITS headers
        # From: https://stackoverflow.com/questions/31999627/storing-logger-messages-in-a-string
        # TODO: Make this capture output from all pyscope loggers, not just the logger in this file
        str_output = StringIO()
        str_handler = logging.StreamHandler(str_output)
        str_handler.setLevel(logging.INFO)
        str_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        str_handler.setFormatter(str_formatter)
        logger.addHandler(str_handler)

        try:
            val_block = schedtab.validate(block, self.observatory)[0]
        except Exception as e:
            logger.exception("Block failed validation, skipping...")
            logger.exception(e)
            if self.update_block_status:
                block["status"] = "F"
                block["message"] = str(e)
            logger.removeHandler(str_handler)
            return ("F", f"Block failed validation: {e}", block)

            logger.info("Block passed validation, continuing...")
            block = val_block

        self._current_block = block

        # Check 1: Block name and status
        prev_status = "S"
        if self.check_block_status:
            if block["status"] != "S":
                try:
                    prev_status = int(block["status"])
                except:
                    logger.info(
                        "Block status is not S or an attempt number, skipping..."
                    )
                    self._current_block = None
                    block["message"] = (
                        "Block status was %s and check_block_status is True, skipping without updating status"
                        % block["status"]
                    )
                    logger.removeHandler(str_handler)
                    return (
                        "F",
                        "Block status was %s and check_block_status is True, skipping without updating status"
                        % block["status"],
                        block,
                    )
        if block["name"] == "TransitionBlock" or block["name"] == "EmptyBlock":
            logger.info("Block is a TransitionBlock or EmptyBlock, skipping...")
            self._current_block = None
            logger.removeHandler(str_handler)
            return ("", "", block)

        # Check 2: Is the previous target different?
        slew = True
        if self.previous_block is not None:
            if (
                self.previous_block["target"].ra.deg == block["target"].ra.deg
                and self.previous_block["target"].dec.deg == block["target"].dec.deg
                and self.previous_block["status"] == "C"
                and self.autofocus_use_current_pointing
            ):
                logger.info(
                    "Previous target is same ra and dec, skipping initial slew..."
                )
                slew = False
            else:
                logger.info(
                    "Previous target is different ra and dec, slewing is requested..."
                )

                logger.info("Stopping tracking...")
                self.observatory.telescope.DeclinationRate = 0
                self.observatory.telescope.RightAscensionRate = 0
                self.observatory.telescope.Tracking = False
                self._telescope_status = "Idle"

                if self.observatory.rotator is not None:
                    logger.info("Turning off derotation...")
                    self.observatory.stop_derotation_thread()

        # Check 3: Wait for block start time?
        if block["start_time"] is None:
            logger.info("No block start time, starting now...")
            block["start_time"] = astrotime.Time.now()

        seconds_until_start_time = (
            block["start_time"] - self.observatory.observatory_time
        ).sec
        if (
            not self.wait_for_block_start_time
            and seconds_until_start_time < self.max_block_late_time
        ):
            logger.info("Ignoring block start time, continuing...")
        elif (
            not self.wait_for_block_start_time
            and seconds_until_start_time > self.max_block_late_time
        ):
            logger.info(
                "Ignoring block start time, however \
                block start time exceeded max_block_late_time of %i seconds, skipping..."
                % self.max_block_late_time
            )
            self._current_block = None
            if type(prev_status) is int:
                new_status = prev_status + 1
            else:
                new_status = "1"
            if self.update_block_status:
                block["status"] = new_status
                block["message"] = "Exceeded max_block_late_time, non-fatal skip"
            logger.removeHandler(str_handler)
            return (new_status, "Exceeded max_block_late_time, non-fatal skip", block)
        elif (
            self.wait_for_block_start_time
            and seconds_until_start_time > self.max_block_late_time
        ):
            logger.info(
                "Block start time exceeded max_block_late_time of %i seconds, skipping..."
                % self.max_block_late_time
            )
            self._current_block = None
            if type(prev_status) is int:
                new_status = prev_status + 1
            else:
                new_status = "1"
            if self.update_block_status:
                block["status"] = new_status
                block["message"] = "Exceeded max_block_late_time"
            logger.removeHandler(str_handler)
            return (new_status, "Exceeded max_block_late_time", block)
        else:
            logger.info(
                "Waiting %.1f seconds (%.2f hours) for block start time..."
                % (seconds_until_start_time, seconds_until_start_time / 3600)
            )

        while (
            self.wait_for_block_start_time
            and seconds_until_start_time > self.preslew_time
        ):
            time.sleep(0.1)
            seconds_until_start_time = (
                block["start_time"] - self.observatory.observatory_time
            ).sec
        else:
            if seconds_until_start_time > 0:
                logger.info(
                    "Block start time in %.1f seconds" % seconds_until_start_time
                )

        # Check 4: Dome status?
        match self.dome_type:
            case "dome":
                if (
                    self.observatory.dome is not None
                    and not self.observatory.dome.CanSetShutter
                ):
                    if self.observatory.dome.ShutterStatus != 0:
                        logger.info("Dome shutter is not open, skipping...")
                        self._current_block = None
                        if type(prev_status) is int:
                            new_status = prev_status + 1
                        else:
                            new_status = "1"
                        if self.update_block_status:
                            block["status"] = new_status
                            block["message"] = "Dome shutter is not open"
                        logger.removeHandler(str_handler)
                        return (new_status, "Dome shutter is not open", block)

            case (
                "safety-monitor" | "safety_monitor" | "safetymonitor" | "safety monitor"
            ):
                if self.observatory.safety_monitor is not None:
                    status = False
                    if self.observatory.safety_monitor is not (iter, tuple, list):
                        status = self.observatory.safety_monitor.IsSafe
                    else:
                        status = self.observatory.safety_monitor[0].IsSafe
                    if not status:
                        logger.info("Safety monitor indicates unsafe, skipping...")
                        self._current_block = None
                        if type(prev_status) is int:
                            new_status = prev_status + 1
                        else:
                            new_status = "1"
                        if self.update_block_status:
                            block["status"] = new_status
                            block["message"] = "Safety monitor indicates unsafe"
                        logger.removeHandler(str_handler)
                        return (
                            new_status,
                            "Dome safety monitor indicates unsafe",
                            block,
                        )

            case "both":
                if self.observatory.safety_monitor is not None:
                    status = False
                    if self.observatory.safety_monitor is not (iter, tuple, list):
                        status = self.observatory.safety_monitor.IsSafe
                    else:
                        status = self.observatory.safety_monitor[0].IsSafe
                    if not status:
                        logger.info("Safety monitor indicates unsafe, skipping...")
                        self._current_block = None
                        if type(prev_status) is int:
                            new_status = prev_status + 1
                        else:
                            new_status = "1"
                        if self.update_block_status:
                            block["status"] = new_status
                            block["message"] = "Safety monitor indicates unsafe"
                        logger.removeHandler(str_handler)
                        return (
                            new_status,
                            "Dome safety monitor indicates unsafe",
                            block,
                        )

                if (
                    self.observatory.dome is not None
                    and self.observatory.dome.CanSetShutter
                ):
                    if self.observatory.dome.ShutterStatus != 0:
                        logger.info("Dome shutter is not open, skipping...")
                        self._current_block = None
                        if type(prev_status) is int:
                            new_status = prev_status + 1
                        else:
                            new_status = "1"
                        if self.update_block_status:
                            block["status"] = new_status
                            block["message"] = "Dome shutter is not open"
                        logger.removeHandler(str_handler)
                        return (new_status, "Dome shutter is not open", block)

        # Check 5: Check safety monitors?
        if self.check_safety_monitors:
            logger.info("Checking safety monitor statuses")

            status = True
            if type(self.observatory.safety_monitor) not in (iter, list, tuple):
                status = self.observatory.safety_monitor.IsSafe
            else:
                for monitor in self.observatory.safety_monitor:
                    status = status and monitor.IsSafe

            if not status:
                logger.info("Safety monitor indicates unsafe, skipping...")
                self._current_block = None
                if type(prev_status) is int:
                    new_status = prev_status + 1
                else:
                    new_status = "1"
                if self.update_block_status:
                    block["status"] = new_status
                    block["message"] = "Safety monitor indicates unsafe"
                logger.removeHandler(str_handler)
                return (new_status, "Safety monitor indicates unsafe", block)

        # Check 6: Wait for sun?
        sun_alt_degs = self.observatory.sun_altaz()[0]
        if self.wait_for_sun and sun_alt_degs > self.max_solar_elev:
            logger.info(
                "Sun altitude: %.3f degs (above limit of %s), skipping..."
                % (sun_alt_degs, self.max_solar_elev)
            )
            self._current_block = None
            if type(prev_status) is int:
                new_status = prev_status + 1
            else:
                new_status = "1"
            if self.update_block_status:
                block["status"] = new_status
                block["message"] = "Sun altitude above limit"
            logger.removeHandler(str_handler)
            return (new_status, "Sun altitude above limit", block)

        # Check 7: Is autofocus needed?
        self._best_focus_result = None
        if (
            self.observatory.focuser is not None
            and self.do_periodic_autofocus
            and (astrotime.Time.now() - self.last_autofocus_time).to(u.s).value
            > self.autofocus_interval
            # TODO: and not block["do_not_interrupt"]
        ):
            logger.info(
                "Autofocus interval of %.2f hours exceeded, performing autofocus..."
                % (self.autofocus_interval / 3600)
            )

            if (
                self.observatory.filter_wheel is not None
                and self.autofocus_filters is not None
            ):
                if (
                    self.observatory.filters[self.observatory.filter_wheel.Position]
                    not in self.autofocus_filters
                ):
                    logger.info(
                        "Current filter not in autofocus filters, switching to the next filter..."
                    )

                    for i in range(
                        self.observatory.filter_wheel.Position + 1,
                        len(self.observatory.filters),
                    ):
                        if self.observatory.filters[i] in self.autofocus_filters:
                            self._filter_wheel_status = "Changing filter"
                            self._focuser_status = (
                                "Offsetting for filter selection"
                                if self.observatory.focuser is not None
                                else ""
                            )
                            self.observatory.set_filter_offset_focuser(
                                filter_name=self.observatory.filters[i]
                            )
                            self._filter_wheel_status = "Idle"
                            self._focuser_status = (
                                "Idle" if self.observatory.focuser is not None else ""
                            )
                            break
                    else:
                        for i in range(self.observatory.filter_wheel.Position - 1):
                            if self.observatory.filters[i] in self.autofocus_filters:
                                self._filter_wheel_status = "Changing filter"
                                self._focuser_status = (
                                    "Offsetting for filter selection"
                                    if self.observatory.focuser is not None
                                    else ""
                                )
                                self.observatory.set_filter_offset_focuser(
                                    filter_name=self.observatory.filters[i]
                                )
                                self._filter_wheel_status = "Idle"
                                self._focuser_status = (
                                    "Idle"
                                    if self.observatory.focuser is not None
                                    else ""
                                )
                                break

            logger.info("Setting camera readout mode to %s" % self.default_readout)
            self.observatory.camera.ReadoutMode = self.default_readout

            logger.info("Starting autofocus...")
            t = threading.Thread(
                target=self._is_process_complete,
                args=(self.autofocus_timeout, self._status_event),
                daemon=True,
                name="is_autofocus_done_thread",
            )
            t.start()
            self._autofocus_status = "Running"
            self._best_focus_result = self.observatory.run_autofocus(
                exposure=self.autofocus_exposure,
                midpoint=self.autofocus_midpoint,
                nsteps=self.autofocus_nsteps,
                step_size=self.autofocus_step_size,
                use_current_pointing=self.autofocus_use_current_pointing,
            )
            self._status_event.set()
            t.join()
            self._status_event.clear()
            self._autofocus_status = "Idle"

            if self._best_focus_result is None:
                self._best_focus_result = self.focuser.Position
                logger.warning("Autofocus failed, will try again on next block")
            else:
                self._last_autofocus_time = astrotime.Time.now()
                logger.info(
                    "Autofocus complete, best focus position: %i"
                    % self._best_focus_result
                )

        # Check 8: Camera temperature
        if (
            self.wait_for_cooldown
            and self.observatory.camera.cooler_setpoint is not None
        ):
            while (
                self.observatory.camera.CCDTemperature
                > self.observatory.cooler_setpoint + self.observatory.cooler_tolerance
            ):
                logger.info(
                    "CCD temperature: %.3f degs (above limit of %.3f with %.3f tolerance)"
                    % (
                        self.observatory.camera.CCDTemperature,
                        self.observatory.cooler_setpoint,
                        self.observatory.cooler_tolerance,
                    )
                )
                time.sleep(10)
                self._camera_status = "Cooling"
        logger.info(
            "CCD temperature: '%s' degs (below limit of '%s' with '%s' tolerance), continuing..."
            % (
                self.observatory.camera.CCDTemperature,
                self.observatory.cooler_setpoint,
                self.observatory.cooler_tolerance,
            )
        )
        self._camera_status = "Idle"

        # Update ephem for non-sidereal targets
        if (block["pm_ra_cosdec"] != 0 or block["pm_dec"] != 0) and block["name"] != "":
            logger.info("Updating ephemeris for '%s' at scheduled time" % block["name"])
            try:
                ephemerides = mpc.MPC.get_ephemeris(
                    target=block["name"],
                    location=self.observatory.observatory_location,
                    start=self.observatory.observatory_time,
                    number=1,
                    proper_motion="sky",
                )
                new_ra = ephemerides["RA"][0]
                new_dec = ephemerides["Dec"][0]
                block["target"] = coord.SkyCoord(ra=new_ra, dec=new_dec)
                block["pm_ra_cosdec"] = ephemerides["dRA cos(Dec)"][0]
                block["pm_dec"] = ephemerides["dDec"][0]
            except Exception as e1:
                try:
                    logger.warning(
                        f"Failed to find proper motions for {block['name']}, trying to find proper motions using astropy.coordinates.get_body"
                    )
                    pos_l = coord.get_body(
                        block["name"],
                        self.observatory.observatory_time - 10 * u.minute,
                        location=self.observatory.observatory_location,
                    )
                    pos_m = coord.get_body(
                        block["name"],
                        (self.observatory.observatory_time),
                        location=self.observatory.observatory_location,
                    )
                    pos_h = coord.get_body(
                        block["name"],
                        self.observatory.observatory_time + 10 * u.minute,
                        location=self.observatory.observatory_location,
                    )
                    new_ra = pos_m.ra
                    new_dec = pos_m.dec
                    block["target"] = coord.SkyCoord(ra=new_ra, dec=new_dec)
                    block["pm_ra_cosdec"] = (
                        (
                            (
                                pos_h.ra * np.cos(pos_h.dec.rad)
                                - pos_l.ra * np.cos(pos_l.dec.rad)
                            )
                            / (pos_h.obstime - pos_l.obstime)
                        )
                        .to(u.arcsec / u.hour)
                        .value
                    )
                    block["pm_dec"] = (
                        ((pos_h.dec - pos_l.dec) / (pos_h.obstime - pos_l.obstime))
                        .to(u.arcsec / u.hour)
                        .value
                    )
                except Exception as e2:
                    logger.warning(
                        f"Failed to find proper motions for {block['name']}, keeping old ephemerides"
                    )

        # Start tracking
        logger.info("Starting tracking...")
        self._telescope_status = "Tracking"
        self.observatory.telescope.Tracking = True

        # Perform centering if requested
        centered = None
        if (
            block["repositioning"][0] != 0
            and block["repositioning"][1] != 0
            and block["target"] is not None
        ):
            logger.info("Telescope pointing repositioning requested...")

            if (
                self.observatory.filter_wheel is not None
                and self.recenter_filters is not None
            ):
                if (
                    self.observatory.filters[self.observatory.filter_wheel.Position]
                    not in self.recenter_filters
                ):
                    logger.info(
                        "Current filter not in recenter filters, switching to the next filter..."
                    )

                    for i in range(
                        self.observatory.filter_wheel.Position + 1,
                        len(self.observatory.filters),
                    ):
                        if self.observatory.filters[i] in self.recenter_filters:
                            t = threading.Thread(
                                target=self._is_process_complete,
                                args=(self.hardware_timeout, self._status_event),
                                daemon=True,
                                name="is_filter_change_done_thread",
                            )
                            t.start()
                            self._filter_wheel_status = "Changing filter"
                            self._focuser_status = (
                                "Offsetting for filter selection"
                                if self.observatory.focuser is not None
                                else ""
                            )
                            self.observatory.set_filter_offset_focuser(
                                filter_name=self.observatory.filters[i]
                            )
                            self._status_event.set()
                            t.join()
                            self._status_event.clear()
                            self._filter_wheel_status = "Idle"
                            self._focuser_status = (
                                "Idle" if self.observatory.focuser is not None else ""
                            )
                            break
                    else:
                        for i in range(self.observatory.filter_wheel.Position - 1):
                            if self.observatory.filters[i] in self.recenter_filters:
                                t = threading.Thread(
                                    target=self._is_process_complete,
                                    args=(self.hardware_timeout, self._status_event),
                                    daemon=True,
                                    name="is_filter_change_done_thread",
                                )
                                t.start()
                                self._filter_wheel_status = "Changing filter"
                                self._focuser_status = (
                                    "Offsetting for filter selection"
                                    if self.observatory.focuser is not None
                                    else ""
                                )
                                self.observatory.set_filter_offset_focuser(
                                    filter_name=self.observatory.filters[i]
                                )
                                self._status_event.set()
                                t.join()
                                self._status_event.clear()
                                self._filter_wheel_status = "Idle"
                                self._focuser_status = (
                                    "Idle"
                                    if self.observatory.focuser is not None
                                    else ""
                                )
                                break

            logger.info("Setting camera readout mode to default for recentering...")
            self.observatory.camera.ReadoutMode = self.default_readout

            if not slew:
                add_attempt = 1
            else:
                add_attempt = 0

            t = threading.Thread(
                target=self._is_process_complete,
                args=(self.hardware_timeout, self._status_event),
                daemon=True,
                name="is_recenter_done_thread",
            )
            t.start()
            self._camera_status = "Recentering"
            self._telescope_status = "Recentering"
            self._wcs_status = (
                "Recentering" if self.observatory._wcs is not None else ""
            )
            self._dome_status = (
                "Recentering" if self.observatory.dome is not None else ""
            )
            self._rotator_status = (
                "Recentering" if self.observatory.rotator is not None else ""
            )
            centered = self.observatory.recenter(
                obj=block["target"],
                target_x_pixel=block["respositioning"][0],
                target_y_pixel=block["respositioning"][1],
                initial_offset_dec=self.recenter_initial_offset_dec,
                check_and_refine=self.recenter_check_and_refine,
                max_attempts=self.recenter_max_attempts + add_attempt,
                tolerance=self.recenter_tolerance,
                exposure=self.recenter_exposure,
                save_images=self.recenter_save_images,
                save_path=self.recenter_save_path,
                sync_mount=self.recenter_sync_mount,
                do_initial_slew=slew,
            )
            self._camera_status = "Idle"
            self._telescope_status = "Idle"
            self._wcs_status = "Idle" if self.observatory.wcs is not None else ""
            self._dome_status = "Idle" if self.observatory.dome is not None else ""
            self._rotator_status = (
                "Idle" if self.observatory.rotator is not None else ""
            )

            if not centered:
                logger.warning("Recentering failed, continuing anyway...")
            else:
                logger.info("Recentering succeeded, continuing...")
        # If not requested, just slew to the source
        elif slew and block["target"] is not None:
            logger.info("Slewing to source...")

            t = threading.Thread(
                target=self._is_process_complete,
                args=(self.hardware_timeout, self._status_event),
                daemon=True,
                name="is_slew_done_thread",
            )
            t.start()
            self._telescope_status = "Slewing"
            self._dome_status = "Slewing" if self.observatory.dome is not None else ""
            self._rotator_status = (
                "Slewing" if self.observatory.rotator is not None else ""
            )
            self.observatory.slew_to_coordinates(
                obj=block["target"],
                control_dome=(self.observatory.dome is not None),
                control_rotator=(self.observatory.rotator is not None),
                wait_for_slew=False,
                track=False,
            )
            self._status_event.set()
            t.join()
            self._status_event.clear()

        # Set filter and focus offset
        if self.observatory.filter_wheel is not None:
            logger.info("Setting filter and focus offset...")
            t = threading.Thread(
                target=self._is_process_complete,
                args=(self.hardware_timeout, self._status_event),
                daemon=True,
                name="is_filter_change_done_thread",
            )
            t.start()
            self._filter_wheel_status = "Changing filter"
            self._focuser_status = (
                "Offsetting for filter selection"
                if self.observatory.focuser is not None
                else ""
            )
            self.observatory.set_filter_offset_focuser(filter_name=block["filter"])
            self._status_event.set()
            t.join()
            self._status_event.clear()
            self._filter_wheel_status = "Idle"
            self._focuser_status = (
                "Idle" if self.observatory.focuser is not None else ""
            )

        # Set binning
        if (
            block["binning"][0] >= 1
            and block["binning"][0] <= self.observatory.camera.MaxBinX
        ):
            logger.info("Setting binx to %i" % block["binning"][0])
            self.observatory.camera.BinX = block["binning"][0]
        else:
            logger.warning(
                "Requested binx of %i is not supported, skipping..."
                % block["binning"][0]
            )
            self._current_block = None
            if self.update_block_status:
                block["status"] = "F"
                block["message"] = (
                    "Requested binx of %i is not supported" % block["binning"][0]
                )
            logger.removeHandler(str_handler)
            return (
                "F",
                "Requested binx of %i is not supported" % block["binning"][0],
                block,
            )

        if (
            block["binning"][1] >= 1
            and block["binning"][1] <= self.observatory.camera.MaxBinY
            and (
                self.observatory.camera.CanAsymmetricBin
                or block["binning"][1] == block["binning"][0]
            )
        ):
            logger.info("Setting biny to %i" % block["binning"][1])
            self.observatory.camera.BinY = block["binning"][1]
        else:
            logger.warning(
                "Requested biny of %i is not supported, skipping..."
                % block["binning"][1]
            )
            self._current_block = None
            if self.update_block_status:
                block["status"] = "F"
                block["message"] = (
                    "Requested biny of %i is not supported" % block["binning"][1]
                )
            logger.removeHandler(str_handler)
            return (
                "F",
                "Requested biny of %i is not supported" % block["binning"][1],
                block,
            )

        # Set subframe
        if block["frame_size"][0] == 0:
            block["frame_size"][0] = int(
                self.observatory.camera.CameraXSize / self.observatory.camera.BinX
            )
        if block["frame_size"][1] == 0:
            block["frame_size"][1] = int(
                self.observatory.camera.CameraYSize / self.observatory.camera.BinY
            )

        if block["frame_position"][0] + block["frame_size"][0] <= int(
            self.observatory.camera.CameraXSize / self.observatory.camera.BinX
        ):
            logger.info(
                "Setting startx and numx to %i, %i"
                % (
                    block["frame_position"][0],
                    block["frame_size"][0],
                )
            )
            self.observatory.camera.StartX = block["frame_position"][0]
            self.observatory.camera.NumX = block["frame_size"][0]
        else:
            logger.warning(
                "Requested startx and numx of %i, %i is not supported, skipping..."
                % (
                    block["frame_position"][0],
                    block["frame_size"][0],
                )
            )
            self._current_block = None
            if self.update_block_status:
                block["status"] = "F"
                block["message"] = (
                    "Requested startx and numx of %i, %i is not supported"
                    % (
                        block["frame_position"][0],
                        block["frame_size"][0],
                    )
                )
            logger.removeHandler(str_handler)
            return (
                "F",
                "Requested startx and numx of %i, %i is not supported"
                % (
                    block["frame_position"][0],
                    block["frame_size"][0],
                ),
                block,
            )

        if block["frame_position"][1] + block["frame_size"][1] <= int(
            self.observatory.camera.CameraYSize / self.observatory.camera.BinY
        ):
            logger.info(
                "Setting starty and numy to %i, %i"
                % (
                    block["frame_position"][1],
                    block["frame_size"][1],
                )
            )
            self.observatory.camera.StartY = block["frame_position"][1]
            self.observatory.camera.NumY = block["frame_size"][1]
        else:
            logger.warning(
                "Requested starty and numy of %i, %i is not supported, skipping..."
                % (
                    block["frame_position"][1],
                    block["frame_size"][1],
                )
            )
            self._current_block = None
            if self.update_block_status:
                block["status"] = "F"
                block["message"] = (
                    "Requested starty and numy of %i, %i is not supported"
                    % (
                        block["frame_position"][1],
                        block["frame_size"][1],
                    )
                )
            logger.removeHandler(str_handler)
            return (
                "F",
                "Requested starty and numy of %i, %i is not supported"
                % (
                    block["frame_position"][1],
                    block["frame_size"][1],
                ),
                block,
            )

        # Set readout mode
        try:
            logger.info("Setting readout mode to %i" % block["readout"])
            self.observatory.camera.ReadoutMode = block["readout"]
        except:
            logger.warning(
                "Requested readout mode of %i is not supported, setting to default of %i"
                % (block["readout"], self.default_readout)
            )
            self.observatory.camera.ReadoutMode = self.default_readout

        # Check for pm exceeding two pixels in one hour
        if block["pm_ra_cosdec"] > 2 * self.observatory.pixel_scale[0] / (
            60 * 60
        ) or block["pm_dec"] > 2 * self.observatory.pixel_scale[1] / (60 * 60):
            logger.info("Switching to non-sidereal tracking...")
            self._telescope_status = "Non-sidereal tracking"
            self.observatory.mount.RightAscensionRate = (
                block["pm_ra_cosdec"]
                * 0.997269567
                / 15.041
                * (1 / np.cos(block["dec"].rad))
            )
            self.observatory.mount.DeclinationRate = block["pm_dec"]
            logger.info(
                "RA rate: %.2f sec-angle/sec"
                % self.observatory.mount.RightAscensionRate
            )
            logger.info(
                "Dec rate: %.2f arcsec/sec" % self.observatory.mount.DeclinationRate
            )

        # Derotation
        if self.observatory.rotator is not None:
            logger.info("Waiting for rotator motion to complete...")
            while self.observatory.rotator.IsMoving:
                time.sleep(0.1)
            logger.info("Starting derotation...")
            self._rotator_status = "Derotating"
            self.observatory.start_derotation_thread()

        # Wait for any motion to complete
        logger.info("Waiting for telescope motion to complete...")
        while self.observatory.telescope.Slewing:
            time.sleep(0.1)

        # Settle time
        logger.info(
            "Waiting for settle time of %.1f seconds..." % self.observatory.settle_time
        )
        self._telescope_status = "Settling"
        time.sleep(self.observatory.settle_time)
        self._telescope_status = "Tracking"

        # Wait for focuser, dome motion to complete
        condition = True
        logger.info("Waiting for focuser or dome motion to complete...")
        while condition:
            if self.observatory.focuser is not None:
                condition = self.observatory.focuser.IsMoving
                if not condition:
                    self._focuser_status = "Idle"
            if self.observatory.dome is not None:
                if not self.observatory.dome.Slewing:
                    self._dome_status = "Idle"
                condition = condition or self.observatory.dome.Slewing
            time.sleep(0.1)

        # Get previous, current, next block info and add to custom header
        custom_header = self.block_info(block)
        prev_info = {}
        if self._previous_block is not None:
            prev_info_tmp = self.block_info(self._previous_block)
            for key in prev_info_tmp:
                prev_info["P" + key] = prev_info_tmp[key]
        next_info = {}
        if self._next_block is not None:
            next_info_tmp = self.block_info(self._next_block)
            for key in next_info_tmp:
                next_info["N" + key] = next_info_tmp[key]
        custom_header.update(prev_info)
        custom_header.update(next_info)

        # If still time before block start, wait
        seconds_until_start_time = (
            (block["start_time"] - self.observatory.observatory_time).to(u.s).value
        )
        if seconds_until_start_time > 0 and self.wait_for_block_start_time:
            logger.info(
                "Waiting %.1f seconds until start time" % seconds_until_start_time
            )
            time.sleep(seconds_until_start_time - 0.1)

        # Add telrun status to custom header
        custom_header.update(self.telrun_info)

        # Start exposures
        for i in range(block["nexp"]):
            logger.info("Beginning exposure %i of %i" % (i + 1, block["nexp"]))
            logger.info("Starting %.4g second exposure..." % block["exposure"])
            self._camera_status = "Exposing"
            t0 = time.time()
            self.observatory.camera.StartExposure(
                block["exposure"],
                block["shutter_state"],
            )
            logger.info("Waiting for image...")
            while (
                not self.observatory.camera.ImageReady
                and time.time() < t0 + block["exposure"] + self.hardware_timeout
            ):
                time.sleep(0.1)
            self._camera_status = "Idle"

            # Append integer to filename if multiple exposures
            if block["nexp"] > 1:
                fname = Path(block["filename"] + "_%i" % i)
            else:
                fname = Path(block["filename"])

            # WCS thread cleanup
            logger.info("Cleaning up WCS threads...")
            self._wcs_threads = [t for t in self._wcs_threads if t.is_alive()]

            # Save image, do WCS if filter in wcs_filters
            if self.observatory.filter_wheel is not None:
                if self.observatory.filter_wheel.Position in self.wcs_filters:
                    logger.info(
                        "Current filter in wcs filters, attempting WCS solve..."
                    )
                    hist = str_output.getvalue().split("\n")
                    save_success = self.observatory.save_last_image(
                        self.images_path / fname + ".tmp",
                        frametyp=block["shutter_state"],
                        custom_header=custom_header,
                        history=hist,
                    )
                    self._wcs_threads.append(
                        threading.Thread(
                            target=self._async_wcs_solver,
                            args=(self.images_path + fname + ".tmp",),
                            daemon=True,
                            name="wcs_threads",
                        )
                    )
                    self._wcs_threads[-1].start()
                else:
                    logger.info(
                        "Current filter not in wcs filters, skipping WCS solve..."
                    )
                    hist = str_output.getvalue().split("\n")
                    save_success = self.observatory.save_last_image(
                        self._images_path / fname,
                        frametyp=block["shutter_state"],
                        custom_header=custom_header,
                        history=hist,
                    )
            else:
                logger.info("No filter wheel, attempting WCS solve...")
                hist = str_output.getvalue().split("\n")
                save_success = self.observatory.save_last_image(
                    self.images_path + fname + ".tmp",
                    frametyp=block["shutter_state"],
                    custom_header=custom_header,
                    history=hist,
                )
                self._wcs_threads.append(
                    threading.Thread(
                        target=self._async_wcs_solver,
                        args=(self.images_path + fname + ".tmp",),
                        daemon=True,
                        name="wcs_threads",
                    )
                )
                self._wcs_threads[-1].start()

        # If multiple exposures, update filename as a list
        if block["nexp"] > 1:
            block["filename"] = [
                block["filename"] + "_%i" % i for i in range(block["nexp"])
            ]

        # Set block status to done
        self._current_block = None
        if self.update_block_status:
            block["status"] = "C"
            block["message"] = "Completed"
        logger.removeHandler(str_handler)
        return ("C", "Completed", block)

    def update_status_log(self):
        if not self.update_status_log:
            logger.warning("Status log update is disabled, exiting...")
            return

        current_block_info = {}
        if self.current_block is not None:
            current_block_info = self.block_info(self.current_block)
        previous_block_info = {}
        if self.previous_block is not None:
            previous_block_info = self.block_info(self.previous_block)
            for key in previous_block_info:
                previous_block_info["P" + key] = previous_block_info.pop(key)
        next_block_info = {}
        if self._next_block is not None:
            next_block_info = self.block_info(self.next_block)
            for key in next_block_info:
                next_block_info["N" + key] = next_block_info.pop(key)

        status_log = self.telrun_info
        status_log.update(previous_block_info)
        status_log.update(current_block_info)
        status_log.update(next_block_info)

        json.dump(
            status_log,
            open(self._logs_path / "telrun_status.json", "w"),
        )

        return status_log

    @staticmethod
    def block_info(block):
        """Return a dictionary of information about the block."""

        # Define custom header
        info = {
            "BLKID": (block["ID"].value, "Block ID"),
            "BLKNAME": (block["name"], "Block name"),
            "BLKSTRT": (block["start_time"].fits, "Block scheduled start time"),
            "BLKEND": (block["end_time"].fits, "Block scheduled end time"),
            "BLKRA": (
                block["target"].ra.to_string(sep="hms", unit="hourangle"),
                "Block target RA",
            ),
            "BLKDEC": (block["target"].dec.to_string(sep="dms"), "Block target Dec"),
            "BLKPRI": (block["priority"], "Block priority"),
        }
        for idx, obs in enumerate(block["observer"]):
            info["OBSERV%i" % idx] = (obs, "Observer %i" % idx)
        info.update(
            {
                "BLKCODE": (block["code"], "Observing code"),
                "BLKTITL": (block["title"], "Title if provided"),
                "BLKFN": (block["filename"], "Expected filename"),
                "BLKTYPE": (block["type"], "Block type"),
                "BLKBACK": (block["backend"], "Block backend"),
                "BLKFILT": (block["filter"], "Block filter"),
                "BLKEXP": (block["exposure"], "Block exposure time"),
                "BLKNEXP": (block["nexp"], "Block number of exposures"),
                "BLKREPX": (block["repositioning"][0], "Block repositioning x pixel"),
                "BLKREPY": (block["repositioning"][1], "Block repositioning y pixel"),
                "BLKSHUT": (block["shutter_state"], "Block shutter state"),
                "BLKREAD": (block["readout"], "Block readout mode"),
                "BLKBINX": (block["binning"][0], "Block binx"),
                "BLKBINY": (block["binning"][1], "Block biny"),
                "BLKSTX": (block["frame_position"][0], "Block startx"),
                "BLKSTY": (block["frame_position"][1], "Block starty"),
                "BLKNUMX": (block["frame_size"][0], "Block numx"),
                "BLKNUMY": (block["frame_size"][1], "Block numy"),
                "BLKPMR": (
                    block["pm_ra_cosdec"],
                    "Block proper motion in RAcosDec [arcsec/hr]",
                ),
                "BLKPMD": (
                    block["pm_dec"],
                    "Block proper motion in Dec [arcsec/hr]",
                ),
                "BLKCOM": (block["comment"], "Block comment"),
                "BLKSCH": (block["sch"], "Block sch file"),
                "SCHID": (block["sch"], "Block sch file"),
                "BLKSTAT": (block["status"], "Block status"),
                "BLKMSG": (block["message"], "Block message"),
                "BLKST": (
                    str(block["sched_time"].fits),
                    "Time when block was scheduled",
                ),
            }
        )
        for idx, const in enumerate(block["constraints"]):
            if const is not None:
                for key_idx, key in enumerate(const.keys()):
                    info["BLKK%i_%i" % (idx, key_idx)] = (
                        key,
                        "Constraint %i" % idx,
                    )
                    info["BLKV%i_%i" % (idx, key_idx)] = (
                        const[key],
                        "Constraint %i" % idx,
                    )

        return info

    @property
    def telrun_info(self):
        """Returns a dictionary of information about the current status of TelrunOperator."""

        info = {
            "OPMODE": (
                ("Robotic", "Operation mode")
                if self._execution_thread is not None
                else ("Manual", "Operation mode")
            ),
            "SUNELEV": (self.observatory.sun_altaz()[0], "Sun elevation"),
            "MOONELEV": (self.observatory.moon_altaz()[0], "Moon elevation"),
            "MOONILL": (self.observatory.moon_illumination(), "Moon illumination"),
            "LST": (self.observatory.lst().value, "Local sidereal time"),
            "UT": (self.observatory.observatory_time.fits, "Universal time"),
            "LASTAUTO": (
                self.last_autofocus_time.fits,
                "Last autofocus time",
            ),
            "NEXTAUTO": (
                (
                    self.last_autofocus_time
                    + self.autofocus_interval * u.s
                    - self.observatory.observatory_time
                )
                .to(u.s)
                .value,
                "Seconds till next autofocus",
            ),
            "PREVBLKT": (
                (
                    (
                        self.previous_block["start_time"]
                        - self.observatory.observatory_time
                    )
                    .to(u.s)
                    .value,
                    "Seconds till previous block",
                )
                if self.previous_block is not None
                else (0, "Seconds till previous block")
            ),
            "CURRBLKT": (
                (
                    (
                        self.current_block["start_time"]
                        - self.observatory.observatory_time
                    )
                    .to(u.s)
                    .value,
                    "Seconds till current block",
                )
                if self.current_block is not None
                else (0, "Seconds till current block")
            ),
            "NEXTBLKT": (
                (
                    (self.next_block["start_time"] - self.observatory.observatory_time)
                    .to(u.s)
                    .value,
                    "Seconds till next block",
                )
                if self.next_block is not None
                else (0, "Seconds till next block")
            ),
            "CURRIDX": (
                (self.current_block_index, "Current block index")
                if self.current_block_index is not None
                else (0, "Current block index")
            ),
            "SKIPPED": (self.skipped_block_count, "Number of skipped blocks"),
            "TOTALBLK": (
                len(self._schedule) if self._schedule is not None else 1,
                "Total number of blocks",
            ),
            "SCHEDFN": (str(self._schedule_fname), "Schedule filename"),
            "AUTOSTAT": (self.autofocus_status, "Autofocus status"),
            "CAMSTAT": (self.camera_status, "Camera status"),
            "COVSTAT": (self.cover_calibrator_status, "Cover calibrator status"),
            "DOMSTAT": (self.dome_status, "Dome status"),
            "FILSTAT": (self.filter_wheel_status, "Filter wheel status"),
            "FOCSTAT": (self.focuser_status, "Focuser status"),
            "OBSSTAT": (
                self.observing_conditions_status,
                "Observing conditions status",
            ),
            "ROTSTAT": (self.rotator_status, "Rotator status"),
            "SAFESTAT": (self.safety_monitor_status, "Safety monitor status"),
            "SWITSTAT": (self.switch_status, "Switch status"),
            "TELSTAT": (self.telescope_status, "Telescope status"),
            "WCSSTAT": (self.wcs_status, "WCS status"),
            "DOMETYPE": (self.dome_type, "Dome type"),
            "INITHOME": (self.initial_home, "Initial home"),
            "WAIT4SUN": (self.wait_for_sun, "Wait for sun"),
            "MAXSUNEL": (self.max_solar_elev, "Max solar elevation"),
            "CHKSAFE": (self.check_safety_monitors, "Check safety monitors"),
            "WAITCOOL": (self._wait_for_cooldown, "Wait for cooldown"),
            "DEFREAD": (self.default_readout, "Default readout"),
            "CHKBLKS": (self.check_block_status, "Check block status"),
            "UPDBLKS": (self.update_block_status, "Update block status"),
            "WAITBLK": (self.wait_for_block_start_time, "Wait for block start time"),
            "MAXLATE": (self.max_block_late_time, "Max block late time"),
            "PRESLEW": (self.preslew_time, "Preslew time"),
            "AUTOINT": (self.autofocus_interval, "Autofocus interval"),
            "AUTOEXPO": (self.autofocus_exposure, "Autofocus exposure"),
            "AUTOMID": (self.autofocus_midpoint, "Autofocus midpoint"),
            "AUTONSTP": (self.autofocus_nsteps, "Autofocus number of steps"),
            "AUTOSTPS": (self.autofocus_step_size, "Autofocus step size"),
            "AUTOPNT": (
                self.autofocus_use_current_pointing,
                "Autofocus use current pointing",
            ),
            "AUTOTIME": (self.autofocus_timeout, "Autofocus timeout"),
            "RECFILT": (self.recenter_filters, "Recenter filters"),
            "RECOFF": (self.recenter_initial_offset_dec, "Recenter initial offset dec"),
            "RECCHK": (self.recenter_check_and_refine, "Recenter check and refine"),
            "RECMAX": (self.recenter_max_attempts, "Recenter max attempts"),
            "RECTOL": (self.recenter_tolerance, "Recenter tolerance"),
            "RECEXP": (self.recenter_exposure, "Recenter exposure"),
            "RECSAVE": (self.recenter_save_images, "Recenter save images"),
            "RECPATH": (self.recenter_save_path, "Recenter save path"),
            "RECSYNC": (self.recenter_sync_mount, "Recenter sync mount"),
            "RECTIME": (self.recenter_timeout, "Recenter timeout"),
            "WCSFILT": (self.wcs_filters, "WCS filters"),
            "WCSTIME": (self.wcs_timeout, "WCS timeout"),
        }

        return info

    def _async_wcs_solver(self, image_path):
        logger.info("Attempting a plate solution...")
        self._wcs_status = "Solving"

        if type(self.observatory.wcs) not in (iter, list, tuple):
            logger.info("Using solver %s" % self.wcs_driver)
            solution = self.wcs.Solve(
                filename,
                ra_key="TELRAIC",
                dec_key="TELDECIC",
                ra_dec_units=("hour", "deg"),
                solve_timeout=self.wcs_timeout,
                scale_units="arcsecperpix",
                scale_type="ev",
                scale_est=self.observatory.pixel_scale[0],
                scale_err=self.observatory.pixel_scale[0] * 0.1,
                parity=1,
                crpix_center=True,
            )
        else:
            for wcs, i in enumerate(self.wcs):
                logger.info("Using solver %s" % self.wcs_driver[i])
                solution = wcs.Solve(
                    filename,
                    ra_key="TELRAIC",
                    dec_key="TELDECIC",
                    ra_dec_units=("hour", "deg"),
                    solve_timeout=self.wcs_timeout,
                    scale_units="arcsecperpix",
                    scale_type="ev",
                    scale_est=self.observatory.pixel_scale[0],
                    scale_err=self.observatory.pixel_scale[0] * 0.1,
                    parity=1,
                    crpix_center=True,
                )
                if solution:
                    break

        if not solution:
            logger.warning("WCS solution not found.")
        else:
            logger.info("WCS solution found.")

        logger.info("Removing tmp extension...")
        shutil.move(image_path, image_path.replace(".tmp", ""))
        logger.info("File %s complete" % image_path.replace(".tmp", ""))
        self._wcs_status = "Idle"

    def _is_process_complete(self, timeout, event):
        t0 = time.time()
        while time.time() < t0 + timeout:
            if not event.is_set():
                time.sleep(0.1)
            else:
                break
        else:
            logger.warning("Process timed out after %.1f seconds" % timeout)
            # TODO: Add auto-recovery capability for the affected hardware

    def start_status_log_thread(self, interval=60):
        self._status_log_update_event.clear()
        self._write_to_status_log = True
        if interval is not None:
            self.status_log_update_interval = interval
        interval = self.status_log_update_interval
        self._status_log_thread = threading.Thread(
            target=self._update_status_log_thread,
            args=[
                interval,
            ],
            daemon=True,
            name="status_log_thread",
        )
        self._status_log_thread.start()

    def stop_status_log_thread(self):
        self._status_log_update_event.set()
        self._write_to_status_log = False
        self._status_log_thread.join()
        self._status_log_thread = None

    def _update_status_log_thread(self, interval):
        while not self._status_log_update_event.is_set():
            l = self.update_status_log()
            time.sleep(interval)

    def _terminate(self):
        self.observatory.shutdown()

    @property
    def schedule_fname(self):
        return self._schedule_fname

    @property
    def schedule(self):
        return self._schedule

    @property
    def best_focus_result(self):
        return self._best_focus_result

    @property
    def do_periodic_autofocus(self):
        return self._do_periodic_autofocus

    @property
    def last_autofocus_time(self):
        return self._last_autofocus_time

    @property
    def skipped_block_count(self):
        return self._skipped_block_count

    @property
    def current_block_index(self):
        return self._current_block_index

    @property
    def current_block(self):
        return self._current_block

    @property
    def previous_block(self):
        return self._previous_block

    @property
    def next_block(self):
        return self._next_block

    @property
    def autofocus_status(self):
        return self._autofocus_status

    @property
    def camera_status(self):
        return self._camera_status

    @property
    def cover_calibrator_status(self):
        return self._cover_calibrator_status

    @property
    def dome_status(self):
        return self._dome_status

    @property
    def filter_wheel_status(self):
        return self._filter_wheel_status

    @property
    def focuser_status(self):
        return self._focuser_status

    @property
    def observing_conditions_status(self):
        return self._observing_conditions_status

    @property
    def rotator_status(self):
        return self._rotator_status

    @property
    def safety_monitor_status(self):
        return self._safety_monitor_status

    @property
    def switch_status(self):
        return self._switch_status

    @property
    def telescope_status(self):
        return self._telescope_status

    @property
    def wcs_status(self):
        return self._wcs_status

    @property
    def observatory(self):
        return self._observatory

    @property
    def telhome(self):
        return self._telhome

    @property
    def queue_fname(self):
        return self._queue_fname

    @property
    def dome_type(self):
        return self._dome_type

    @property
    def initial_home(self):
        return self._initial_home

    @initial_home.setter
    def initial_home(self, value):
        self._initial_home = bool(value)
        self._config["telrun"]["initial_home"] = str(self._initial_home)

    @property
    def wait_for_sun(self):
        return self._wait_for_sun

    @wait_for_sun.setter
    def wait_for_sun(self, value):
        self._wait_for_sun = bool(value)
        self._config["telrun"]["wait_for_sun"] = str(self._wait_for_sun)

    @property
    def max_solar_elev(self):
        return self._max_solar_elev

    @max_solar_elev.setter
    def max_solar_elev(self, value):
        self._max_solar_elev = float(value)
        self._config["telrun"]["max_solar_elev"] = str(self._max_solar_elev)

    @property
    def check_safety_monitors(self):
        return self._check_safety_monitors

    @check_safety_monitors.setter
    def check_safety_monitors(self, value):
        self._check_safety_monitors = bool(value)
        self._config["telrun"]["check_safety_monitors"] = str(
            self._check_safety_monitors
        )

    @property
    def wait_for_cooldown(self):
        return self._wait_for_cooldown

    @wait_for_cooldown.setter
    def wait_for_cooldown(self, value):
        self._wait_for_cooldown = bool(value)
        self._config["telrun"]["wait_for_cooldown"] = str(self._wait_for_cooldown)

    @property
    def default_readout(self):
        return self._default_readout

    @default_readout.setter
    def default_readout(self, value):
        self._default_readout = int(value)
        self._config["telrun"]["default_readout"] = str(self._default_readout)

    @property
    def check_block_status(self):
        return self._check_block_status

    @check_block_status.setter
    def check_block_status(self, value):
        self._check_block_status = bool(value)
        self._config["telrun"]["check_block_status"] = str(self._check_block_status)

    @property
    def update_block_status(self):
        return self._update_block_status

    @update_block_status.setter
    def update_block_status(self, value):
        self._update_block_status = bool(value)
        self._config["telrun"]["update_block_status"] = str(self._update_block_status)

    @property
    def write_to_status_log(self):
        return self._write_to_status_log

    @write_to_status_log.setter
    def write_to_status_log(self, value):
        self._write_to_status_log = bool(value)
        self._config["telrun"]["write_to_status_log"] = str(self._write_to_status_log)
        if self._write_to_status_log:
            self.start_status_log_thread()

    @property
    def status_log_update_interval(self):
        return self._status_log_update_interval

    @status_log_update_interval.setter
    def status_log_update_interval(self, value):
        self._status_log_update_interval = float(value)
        self._config["telrun"]["status_log_update_interval"] = str(
            self._status_log_update_interval
        )

    @property
    def wait_for_block_start_time(self):
        return self._wait_for_block_start_time

    @wait_for_block_start_time.setter
    def wait_for_block_start_time(self, value):
        self._wait_for_block_start_time = bool(value)
        self._config["telrun"]["wait_for_block_start_time"] = str(
            self._wait_for_block_start_time
        )

    @property
    def max_block_late_time(self):
        return self._max_block_late_time

    @max_block_late_time.setter
    def max_block_late_time(self, value):
        if value < 0:
            value = 1e99
        self._max_block_late_time = float(value)
        self._config["telrun"]["max_block_late_time"] = str(self._max_block_late_time)

    @property
    def preslew_time(self):
        return self._preslew_time

    @preslew_time.setter
    def preslew_time(self, value):
        self._preslew_time = float(value)
        self._config["telrun"]["preslew_time"] = str(self._preslew_time)

    @property
    def hardware_timeout(self):
        return self._hardware_timeout

    @hardware_timeout.setter
    def hardware_timeout(self, value):
        self._hardware_timeout = float(value)
        self._config["telrun"]["hardware_timeout"] = str(self._hardware_timeout)

    @property
    def autofocus_filters(self):
        return self._autofocus_filters

    @autofocus_filters.setter
    def autofocus_filters(self, value):
        if value is self._autofocus_filters:
            return
        if type(value) is list:
            self._autofocus_filters = value
        else:
            self._autofocus_filters = (
                [v for v in value.replace(" ", "").split(",")]
                if value is not None or value != ""
                else None
            )
        self._config["autofocus"]["autofocus_filters"] = (
            ", ".join(self._autofocus_filters)
            if self._autofocus_filters is not None
            else ""
        )

    @property
    def autofocus_interval(self):
        return self._autofocus_interval

    @autofocus_interval.setter
    def autofocus_interval(self, value):
        self._autofocus_interval = float(value)
        self._config["autofocus"]["autofocus_interval"] = str(self._autofocus_interval)

    @property
    def autofocus_initial(self):
        return self._autofocus_initial

    @autofocus_initial.setter
    def autofocus_initial(self, value):
        self._autofocus_initial = bool(value)
        self._config["autofocus"]["autofocus_initial"] = str(self._autofocus_initial)

    @property
    def autofocus_exposure(self):
        return self._autofocus_exposure

    @autofocus_exposure.setter
    def autofocus_exposure(self, value):
        self._autofocus_exposure = float(value)
        self._config["autofocus"]["autofocus_exposure"] = str(self._autofocus_exposure)

    @property
    def autofocus_midpoint(self):
        return self._autofocus_midpoint

    @autofocus_midpoint.setter
    def autofocus_midpoint(self, value):
        self._autofocus_midpoint = float(value)
        self._config["autofocus"]["autofocus_midpoint"] = str(self._autofocus_midpoint)

    @property
    def autofocus_nsteps(self):
        return self._autofocus_nsteps

    @autofocus_nsteps.setter
    def autofocus_nsteps(self, value):
        self._autofocus_nsteps = int(value)
        self._config["autofocus"]["autofocus_nsteps"] = str(self._autofocus_nsteps)

    @property
    def autofocus_step_size(self):
        return self._autofocus_step_size

    @autofocus_step_size.setter
    def autofocus_step_size(self, value):
        self._autofocus_step_size = int(value)
        self._config["autofocus"]["autofocus_step_size"] = str(
            self._autofocus_step_size
        )

    @property
    def autofocus_use_current_pointing(self):
        return self._autofocus_use_current_pointing

    @autofocus_use_current_pointing.setter
    def autofocus_use_current_pointing(self, value):
        self._autofocus_use_current_pointing = bool(value)
        self._config["autofocus"]["autofocus_use_current_pointing"] = str(
            self._autofocus_use_current_pointing
        )

    @property
    def autofocus_timeout(self):
        return self._autofocus_timeout

    @autofocus_timeout.setter
    def autofocus_timeout(self, value):
        self._autofocus_timeout = float(value)
        self._config["autofocus"]["autofocus_timeout"] = str(self._autofocus_timeout)

    @property
    def recenter_filters(self):
        return self._recenter_filters

    @recenter_filters.setter
    def recenter_filters(self, value):
        if value is self._recenter_filters:
            return
        if type(value) is list:
            self._recenter_filters = value
        else:
            self._recenter_filters = (
                [v for v in value.replace(" ", "").split(",")]
                if value is not None or value != ""
                else None
            )
        self._config["recenter"]["recenter_filters"] = (
            ", ".join(self._recenter_filters)
            if self._recenter_filters is not None
            else ""
        )

    @property
    def recenter_initial_offset_dec(self):
        return self._recenter_initial_offset_dec

    @recenter_initial_offset_dec.setter
    def recenter_initial_offset_dec(self, value):
        self._recenter_initial_offset_dec = float(value)
        self._config["recenter"]["recenter_initial_offset_dec"] = str(
            self._recenter_initial_offset_dec
        )

    @property
    def recenter_check_and_refine(self):
        return self._recenter_check_and_refine

    @recenter_check_and_refine.setter
    def recenter_check_and_refine(self, value):
        self._recenter_check_and_refine = bool(value)
        self._config["recenter"]["recenter_check_and_refine"] = str(
            self._recenter_check_and_refine
        )

    @property
    def recenter_max_attempts(self):
        return self._recenter_max_attempts

    @recenter_max_attempts.setter
    def recenter_max_attempts(self, value):
        self._recenter_max_attempts = int(value)
        self._config["recenter"]["recenter_max_attempts"] = str(
            self._recenter_max_attempts
        )

    @property
    def recenter_tolerance(self):
        return self._recenter_tolerance

    @recenter_tolerance.setter
    def recenter_tolerance(self, value):
        self._recenter_tolerance = float(value)
        self._config["recenter"]["recenter_tolerance"] = str(self._recenter_tolerance)

    @property
    def recenter_exposure(self):
        return self._recenter_exposure

    @recenter_exposure.setter
    def recenter_exposure(self, value):
        self._recenter_exposure = float(value)
        self._config["recenter"]["recenter_exposure"] = str(self._recenter_exposure)

    @property
    def recenter_save_images(self):
        return self._recenter_save_images

    @recenter_save_images.setter
    def recenter_save_images(self, value):
        self._recenter_save_images = bool(value)
        self._config["recenter"]["recenter_save_images"] = str(
            self._recenter_save_images
        )

    @property
    def recenter_save_path(self):
        return self._recenter_save_path

    @recenter_save_path.setter
    def recenter_save_path(self, value):
        self._recenter_save_path = os.path.abspath(value)
        self._config["recenter"]["recenter_save_path"] = str(self._recenter_save_path)

    @property
    def recenter_sync_mount(self):
        return self._recenter_sync_mount

    @recenter_sync_mount.setter
    def recenter_sync_mount(self, value):
        self._recenter_sync_mount = bool(value)
        self._config["recenter"]["recenter_sync_mount"] = str(self._recenter_sync_mount)

    @property
    def recenter_timeout(self):
        return self._recenter_timeout

    @recenter_timeout.setter
    def recenter_timeout(self, value):
        self._recenter_timeout = float(value)
        self._config["recenter"]["recenter_timeout"] = str(self._recenter_timeout)

    @property
    def wcs_filters(self):
        return self._wcs_filters

    @wcs_filters.setter
    def wcs_filters(self, value):
        if value is self._wcs_filters:
            return
        if type(value) is list:
            self._wcs_filters = value
        else:
            self._wcs_filters = (
                [v for v in value.replace(" ", "").split(",")]
                if value is not None or value != ""
                else None
            )
        self._config["wcs"]["wcs_filters"] = (
            ", ".join(self._wcs_filters) if self._wcs_filters is not None else ""
        )

    @property
    def wcs_timeout(self):
        return self._wcs_timeout

    @wcs_timeout.setter
    def wcs_timeout(self, value):
        self._wcs_timeout = float(value)
        self._config["wcs"]["wcs_timeout"] = str(self._wcs_timeout)


class _TelrunGUI(ttk.Frame):
    def __init__(self, parent, TelrunOperator):
        ttk.Frame.__init__(self, parent)
        self._parent = parent
        self._telrun = TelrunOperator

        self._gui_font = tk.font.Font(family="Segoe UI", size=10)

        self._build_gui()
        self._update()

    def _build_gui(self):
        ttk.Label(self, text="System Status", font=self._gui_font).grid(
            row=0, column=0, columnspan=3, sticky="new"
        )
        self.system_status_widget = _SystemStatusWidget(self)
        self.system_status_widget.grid(row=1, column=0, columnspan=3, sticky="sew")

        ttk.Label(self, text="Previous Block", font=self._gui_font).grid(
            row=2, column=0, columnspan=1, sticky="sew"
        )
        self.previous_block_widget = _BlockWidget(self)
        self.previous_block_widget.grid(row=3, column=0, columnspan=1, sticky="new")

        ttk.Label(self, text="Current Block", font=self._gui_font).grid(
            row=2, column=1, columnspan=1, sticky="sew"
        )
        self.current_block_widget = _BlockWidget(self)
        self.current_block_widget.grid(row=3, column=1, columnspan=1, sticky="new")

        ttk.Label(self, text="Next Block", font=self._gui_font).grid(
            row=2, column=2, columnspan=1, sticky="sew"
        )
        self.next_block_widget = _BlockWidget(self)
        self.next_block_widget.grid(row=3, column=2, columnspan=1, sticky="new")

        self.schedule = tksheet.Sheet(
            self,
            width=80,
            height=20,
            headers=[
                "ID",
                "Name",
                "Start",
                "End",
                "RA",
                "Dec",
                "Priority",
                "Observer",
                "Code",
                "Title",
                "Filename",
                "Type",
                "Backend",
                "Filter",
                "Exposure",
                "Nexp",
                "Repositioning",
                "Shutter",
                "Readout",
                "Binning",
                "Frame Position",
                "Frame Size",
                "PM RAcosDec",
                "PM Dec",
                "Comment",
                "sch",
                "Status",
                "Message",
                "Sched Time",
                "Constraints",
            ],
        )
        self.schedule.grid(column=0, row=4, columnspan=3, sticky="new")

        self.log_text = tk.ScrolledText(self, width=80, height=20, state="disabled")
        self.log_text.grid(column=0, row=5, columnspan=3, sticky="new")

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(4, weight=1)
        self.rowconfigure(5, weight=1)

        log_handler = _TextHandler(self.log_text)
        log_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        log_handler.setFormatter(log_formatter)
        logger.addHandler(log_handler)

    def _update(self):
        self.system_status_widget.update()
        self.previous_block_widget.update(self._telrun.previous_block)
        self.current_block_widget.update(self._telrun.current_block)
        self.next_block_widget.update(self._telrun.next_block)

        if self._telrun._schedule is None:
            self.schedule.set_sheet_data([[]])
        else:
            self.sheet.set_sheet_data(
                [
                    [
                        self._telrun.schedule["ID"][i],
                        self._telrun.schedule["name"][i],
                        self._telrun.schedule["start_time"][i].iso,
                        self._telrun.schedule["end_time"][i].iso,
                        self._telrun.schedule["target"][i].ra.to_string("hms"),
                        self._telrun.schedule["target"][i].dec.to_string("dms"),
                        self._telrun.schedule["priority"][i],
                        str(self._telrun.schedule["observer"][i]),
                        self._telrun.schedule["code"][i],
                        self._telrun.schedule["title"][i],
                        self._telrun.schedule["filename"][i],
                        self._telrun.schedule["type"][i],
                        self._telrun.schedule["backend"][i],
                        self._telrun.schedule["filter"][i],
                        self._telrun.schedule["exposure"][i],
                        self._telrun.schedule["nexp"][i],
                        str(self._telrun.schedule["respositioning"][i][0])
                        + ","
                        + str(self._telrun.schedule["respositioning"][i][1]),
                        self._telrun.schedule["shutter_state"][i],
                        self._telrun.schedule["readout"][i],
                        str(self._telrun.schedule["binning"][i][0])
                        + "x"
                        + str(self._telrun.schedule["binning"][i][1]),
                        str(self._telrun.schedule["frame_position"][i][0])
                        + ","
                        + str(self._telrun.schedule["frame_position"][i][1]),
                        str(self._telrun.schedule["frame_size"][i][0])
                        + "x"
                        + str(self._telrun.schedule["frame_size"][i][1]),
                        str(
                            self._telrun.schedule["pm_ra_cosdec"][i].to_value(
                                u.arcsec / u.hour
                            )
                        ),
                        str(
                            self._telrun.schedule["pm_dec"][i].to_value(
                                u.arcsec / u.hour
                            )
                        ),
                        self._telrun.schedule["comment"][i],
                        self._telrun.schedule["sch"][i],
                        self._telrun.schedule["status"][i],
                        self._telrun.schedule["message"][i],
                        self._telrun.schedule["sched_time"][i].iso,
                        str(self._telrun.schedule["constraints"][i]),
                    ]
                    for i in range(len(self._telrun.schedule))
                ]
            )

        self.after(1000, self._update)


class _SystemStatusWidget(ttk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self._parent = parent

        self.build_gui()
        self.update()

    def build_gui(self):
        rows0 = _Rows(self, 0)
        self.operator_mode = rows0.add_row("Op Mode:")
        self.sun_elevation = rows0.add_row("Sun El:")
        self.moon_elevation = rows0.add_row("Moon El:")
        self.moon_illumination = rows0.add_row("Moon Ill:")
        self.lst = rows0.add_row("LST:")
        self.ut = rows0.add_row("UT:")
        self.last_autofocus_time = rows0.add_row("Last AF:")
        self.time_until_next_autofocus = rows0.add_row("Next AF:")
        self.current_block_index = rows0.add_row("Block Idx:")
        self.skipped_block_count = rows0.add_row("Skipped:")
        self.total_block_count = rows0.add_row("Total:")
        self.sched_fname = rows0.add_row("Sched File:")

        rows1 = _Rows(self, 2)
        self.autofocus_status = rows1.add_row("AF:")
        self.camera_status = rows1.add_row("Cam:")
        self.cover_calibrator_status = rows1.add_row("CC:")
        self.dome_status = rows1.add_row("Dome:")
        self.filter_wheel_status = rows1.add_row("FW:")
        self.focuser_status = rows1.add_row("Foc:")
        self.observing_conditions_status = rows1.add_row("OC:")
        self.rotator_status = rows1.add_row("Rot:")
        self.safety_monitor_status = rows1.add_row("SM:")
        self.switch_status = rows1.add_row("Sw:")
        self.telescope_status = rows1.add_row("Tel:")
        self.wcs_status = rows1.add_row("WCS:")

        self.rows = [rows0, rows1]

    def update(self):
        info_dict = self._parent._telrun.telrun_info
        self.operator_mode.set(info_dict["OPMODE"][0])
        self.sun_elevation.set(info_dict["SUNELEV"][0])
        self.moon_elevation.set(info_dict["MOONELEV"][0])
        self.moon_illumination.set(info_dict["MOONILL"][0])
        self.lst.set(info_dict["LST"][0])
        self.ut.set(info_dict["UT"][0])
        self.last_autofocus_time.set(info_dict["LASTAUTO"][0])
        self.time_until_next_autofocus.set(info_dict["NEXTAUTO"][0])
        self.current_block_index.set(info_dict["CURRIDX"][0])
        self.skipped_block_count.set(info_dict["SKIPPED"][0])
        self.total_block_count.set(info_dict["TOTALBLK"][0])
        self.sched_fname.set(info_dict["SCHEDFN"][0])

        self.autofocus_status.set(info_dict["AUTOSTAT"][0])
        self.camera_status.set(info_dict["CAMSTAT"][0])
        self.cover_calibrator_status.set(info_dict["COVSTAT"][0])
        self.dome_status.set(info_dict["DOMSTAT"][0])
        self.filter_wheel_status.set(info_dict["FILSTAT"][0])
        self.focuser_status.set(info_dict["FOCSTAT"][0])
        self.observing_conditions_status.set(info_dict["OBSSTAT"][0])
        self.rotator_status.set(info_dict["ROTSTAT"][0])
        self.safety_monitor_status.set(info_dict["SAFESTAT"][0])
        self.switch_status.set(info_dict["SWITSTAT"][0])
        self.telescope_status.set(info_dict["TELSTAT"][0])
        self.wcs_status.set(info_dict["WCSSTAT"][0])


class _BlockWidget(ttk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self._parent = parent

        self.build_gui()
        self.update()

    def build_gui(self):
        self.rows = _Rows(self, 0)

        self.id = self.rows.add_row("ID:")
        self.name = self.rows.add_row("Name:")
        self.start_time = self.rows.add_row("Start:")
        self.end_time = self.rows.add_row("End:")
        self.ra = self.rows.add_row("RA:")
        self.dec = self.rows.add_row("Dec:")
        self.priority = self.rows.add_row("Priority:")
        self.observer = self.rows.add_row("Observer:")
        self.code = self.rows.add_row("Code:")
        self.title = self.rows.add_row("Title:")
        self.filename = self.rows.add_row("Filename:")
        self.type = self.rows.add_row("Type:")
        self.backend = self.rows.add_row("Backend:")
        self.filter = self.rows.add_row("Filter:")
        self.exposure = self.rows.add_row("Exposure:")
        self.nexp = self.rows.add_row("Nexp:")
        self.repositioning = self.rows.add_row("Repositioning:")
        self.shutter_state = self.rows.add_row("Shutter:")
        self.readout = self.rows.add_row("Readout:")
        self.binning = self.rows.add_row("Binning:")
        self.frame_position = self.rows.add_row("Frame Pos:")
        self.frame_size = self.rows.add_row("Frame Size:")
        self.pm_ra_cosdec = self.rows.add_row("PM RAcosDec:")
        self.pm_dec = self.rows.add_row("PM Dec:")
        self.comment = self.rows.add_row("Comment:")
        self.sch = self.rows.add_row("sch:")
        self.status = self.rows.add_row("Status:")
        self.message = self.rows.add_row("Message:")
        self.sched_time = self.rows.add_row("Sched Time:")
        self.constraints = self.rows.add_row("Constraints:")

    def update(self, block):
        if block is None:
            self.id.set("")
            self.name.set("")
            self.start_time.set("")
            self.end_time.set("")
            self.ra.set("")
            self.dec.set("")
            self.priority.set("")
            self.observer.set("")
            self.code.set("")
            self.title.set("")
            self.filename.set("")
            self.type.set("")
            self.backend.set("")
            self.filter.set("")
            self.exposure.set("")
            self.nexp.set("")
            self.repositioning.set("")
            self.shutter_state.set("")
            self.readout.set("")
            self.binning.set("")
            self.frame_position.set("")
            self.frame_size.set("")
            self.pm_ra_cosdec.set("")
            self.pm_dec.set("")
            self.comment.set("")
            self.sch.set("")
            self.status.set("")
            self.message.set("")
            self.sched_time.set("")
            self.constraints.set("")
        else:
            self.id.set(str(block["ID"]))
            self.name.set(block["name"])
            self.start_time.set(block["start_time"].iso)
            self.end_time.set(block["end_time"].iso)
            self.ra.set(block["target"].ra.to_string("hms"))
            self.dec.set(block["target"].dec.to_string("dms"))
            self.priority.set(str(block["priority"]))
            self.observer.set(str(block["observer"]))
            self.code.set(block["code"])
            self.title.set(block["title"])
            self.filename.set(block["filename"])
            self.type.set(block["type"])
            self.backend.set(block["backend"])
            self.filter.set(block["filter"])
            self.exposure.set(str(block["exposure"]))
            self.nexp.set(str(block["nexp"]))
            self.repositioning.set(
                str(block["respositioning"][0]) + "," + str(block["respositioning"][1])
            )
            self.shutter_state.set(block["shutter_state"])
            self.readout.set(str(block["readout"]))
            self.binning.set(str(block["binning"][0]) + "x" + str(block["binning"][1]))
            self.frame_position.set(
                str(block["frame_position"][0]) + "," + str(block["frame_position"][1])
            )
            self.frame_size.set(
                str(block["frame_size"][0]) + "x" + str(block["frame_size"][1])
            )
            self.pm_ra_cosdec.set(
                str(block["pm_ra_cosdec"].to_value(u.arcsec / u.hour))
            )
            self.pm_dec.set(str(block["pm_dec"].to_value(u.arcsec / u.hour)))
            self.comment.set(block["comment"])
            self.sch.set(block["sch"])
            self.status.set(block["status"])
            self.message.set(block["message"])
            self.sched_time.set(block["sched_time"].iso)
            self.constraints.set(str(block["constraints"]))
        self.update_idletasks()


class _Rows:
    def __init__(self, parent, column):
        self._parent = parent
        self._column = column
        self._next_row = 0

        self.labels = []
        self.string_vars = []

    def add_row(self, label_text):
        label = ttk.Label(self._parent, text=label_text)
        label.grid(column=self._column, row=self._next_row, sticky="e")
        self.labels.append(label)

        string_var = tk.StringVar()
        entry = ttk.Entry(self._parent, textvariable=string_var)
        entry.grid(column=self._column + 1, row=self._next_row, sticky="ew")
        self.string_vars.append(string_var)

        self._next_row += 1

        return string_var


class _TextHandler(logging.Handler):
    # This class allows you to log to a Tkinter Text or ScrolledText widget
    # Adapted from Moshe Kaplan: https://gist.github.com/moshekaplan/c425f861de7bbf28ef06

    def __init__(self, text):
        # run the regular Handler __init__
        logging.Handler.__init__(self)
        # Store a reference to the Text it will log to
        self.text = text

    def emit(self, record):
        msg = self.format(record)

        def append():
            self.text.configure(state="normal")
            self.text.insert(tk.END, msg + "\n")
            self.text.configure(state="disabled")
            # Autoscroll to the bottom
            self.text.yview(tk.END)

        # This is necessary because we can't modify the Text from other threads
        self.text.after(0, append)
