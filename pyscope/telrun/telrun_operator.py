import atexit
import configparser
import logging
import os
import shutil
import threading
import tkinter as tk
import tkinter.ttk as ttk
from io import StringIO
from tkinter import font

import astroplan
import tksheet
from astropy import coordinates as coord
from astropy import table
from astropy import time as astrotime
from astropy import units as u
from astroquery import mpc

from ..observatory import Observatory
from . import TelrunException, schedtab

logger = logging.getLogger(__name__)


class TelrunOperator:
    def __init__(self, config_path="./config/", gui=True, **kwargs):
        # Private attributes
        self._config = configparser.ConfigParser()
        self._gui = gui
        self._execution_thread = None
        self._execution_event = threading.Event()
        self._status_event = threading.Event()
        self._schedule = None
        self._schedule_last_modified = 0
        self._best_focus_result = None
        self._wcs_threads = []

        # Read-only attributes with no constructor arguments
        self._do_periodic_autofocus = True
        self._last_autofocus_time = 0
        self._skipped_block_count = 0
        self._current_block = None
        self._current_block_index = None
        self._previous_block = None
        self._previous_block_index = None
        self._next_block = None
        self._next_block_index = None
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

        # Read-only attributes with constructor arguments
        self._config_path = os.path.abspath(config_path)
        self._schedules_path = "./schedules/"
        self._images_path = "./images/"
        self._log_path = "./logs/"
        self._dome_type = None  # None, 'dome' or 'safety-monitor' or 'both'

        # Public attributes with constructor arguments
        self._initial_home = True
        self._wait_for_sun = True
        self._max_solar_elev = -12
        self._check_safety_monitors = True
        self._wait_for_cooldown = True
        self._default_readout = 0
        self._check_block_status = True
        self._update_block_status = True
        self._write_to_schedule_log = True
        self._write_to_status_log = True
        self._autofocus_interval = 3600
        self._initial_autofocus = True
        self._autofocus_filters = None
        self._autofocus_exposure = 5
        self._autofocus_midpoint = 0
        self._autofocus_nsteps = 5
        self._autofocus_step_size = 500
        self._autofocus_use_current_pointing = False
        self._autofocus_timeout = 180
        self._wait_for_block_start_time = True
        self._max_block_late_time = 60
        self._preslew_time = 60
        self._recenter_filters = None
        self._recenter_initial_offset_dec = 0
        self._recenter_check_and_refine = True
        self._recenter_max_attempts = 5
        self._recenter_tolerance = 3
        self._recenter_exposure = 10
        self._recenter_save_images = False
        self._recenter_save_path = "./"
        self._recenter_sync_mount = False
        self._hardware_timeout = 120
        self._wcs_filters = None
        self._wcs_timeout = 30

        save_at_end = False
        if os.path.isdir(self._config_path):
            logger.info("config_path is a directory, looking for telrun.cfg")
            read_path = os.path.join(self._config_path, "telrun.cfg")
        elif os.path.isfile(self._config_path):
            logger.info("config_path is a file, setting config directory to its parent")
            read_path = self._config_path
            self._config_path = os.path.abspath(os.path.dirname(self._config_path))
        else:
            logger.info(
                "config_path does not exist, creating a directory at %s"
                % self._config_path
            )
            os.mkdir(self._config_path)
            save_at_end = True

        # Load config file if there
        if os.path.isfile(read_path):
            logger.info(
                "Using config file to initialize telrun: %s" % self._config_path
            )
            try:
                self._config.read(self._config_path)
            except:
                raise TelrunException(
                    "Could not read config file: %s" % self._config_path
                )

            self._schedules_path = self._config.get(
                "default", "schedules_path", fallback=self._schedules_path
            )
            self._images_path = self._config.get(
                "default", "images_path", fallback=self._images_path
            )
            self._logs_path = self._config.get(
                "default", "logs_path", fallback=self._logs_path
            )
            self._dome_type = self._config.get(
                "default", "dome_type", fallback=self._dome_type
            )
            self._initial_home = self._config.getboolean(
                "default", "initial_home", fallback=self._initial_home
            )
            self._wait_for_sun = self._config.getboolean(
                "default", "wait_for_sun", fallback=self._wait_for_sun
            )
            self._max_solar_elev = self._config.getfloat("default", "max_solar_elev")
            self._check_safety_monitors = self._config.getboolean(
                "default", "check_safety_monitors", fallback=self._check_safety_monitors
            )
            self._wait_for_cooldown = self._config.getboolean(
                "default", "wait_for_cooldown", fallback=self._wait_for_cooldown
            )
            self._default_readout = self._config.getint(
                "default", "default_readout", fallback=self._default_readout
            )
            self._check_block_status = self._config.getboolean(
                "default", "check_block_status", fallback=self._check_block_status
            )
            self._update_block_status = self._config.getboolean(
                "default", "update_block_status", fallback=self._update_block_status
            )
            self._write_to_schedule_log = self._config.getboolean(
                "default", "write_to_schedule_log", fallback=self._write_to_schedule_log
            )
            self._write_to_status_log = self._config.getboolean(
                "default", "write_to_status_log", fallback=self._write_to_status_log
            )
            self._autofocus_interval = self._config.getfloat(
                "default", "autofocus_interval", fallback=self._autofocus_interval
            )
            self._initial_autofocus = self._config.getboolean(
                "default", "initial_autofocus", fallback=self._initial_autofocus
            )

            self._autofocus_filters = [
                f.strip()
                for f in self._config.get("default", "autofocus_filters").split(",")
            ]

            self._autofocus_exposure = self._config.getfloat(
                "default", "autofocus_exposure", fallback=self._autofocus_exposure
            )
            self._autofocus_midpoint = self._config.getfloat(
                "default", "autofocus_midpoint", fallback=self._autofocus_midpoint
            )
            self._autofocus_nsteps = self._config.getint(
                "default", "autofocus_nsteps", fallback=self._autofocus_nsteps
            )
            self._autofocus_step_size = self._config.getfloat(
                "default", "autofocus_step_size", fallback=self._autofocus_step_size
            )
            self._autofocus_use_current_pointing = self._config.getboolean(
                "default",
                "autofocus_use_current_pointing",
                fallback=self._autofocus_use_current_pointing,
            )
            self._autofocus_timeout = self._config.getfloat(
                "default", "autofocus_timeout", fallback=self._autofocus_timeout
            )
            self._wait_for_block_start_time = self._config.getboolean(
                "default",
                "wait_for_block_start_time",
                fallback=self._wait_for_block_start_time,
            )
            self._max_block_late_time = self._config.getfloat(
                "default", "max_block_late_time", fallback=self._max_block_late_time
            )
            self._preslew_time = self._config.getfloat(
                "default", "preslew_time", fallback=self._preslew_time
            )

            self._recenter_filters = [
                f.strip()
                for f in self._config.get("default", "recenter_filters").split(",")
            ]

            self._recenter_initial_offset_dec = self._config.getfloat(
                "default",
                "recenter_initial_offset_dec",
                fallback=self._recenter_initial_offset_dec,
            )
            self._recenter_check_and_refine = self._config.getboolean(
                "default",
                "recenter_check_and_refine",
                fallback=self._recenter_check_and_refine,
            )
            self._recenter_max_attempts = self._config.getint(
                "default", "recenter_max_attempts", fallback=self._recenter_max_attempts
            )
            self._recenter_tolerance = self._config.getfloat(
                "default", "recenter_tolerance", fallback=self._recenter_tolerance
            )
            self._recenter_exposure = self._config.getfloat(
                "default", "recenter_exposure", fallback=self._recenter_exposure
            )
            self._recenter_save_images = self._config.getboolean(
                "default", "recenter_save_images", fallback=self._recenter_save_images
            )
            self._recenter_save_path = self._config.get(
                "default", "recenter_save_path", fallback=self._recenter_save_path
            )
            self._recenter_sync_mount = self._config.getboolean(
                "default", "recenter_sync_mount", fallback=self._recenter_sync_mount
            )
            self._hardware_timeout = self._config.getfloat(
                "default", "hardware_timeout", fallback=self._hardware_timeout
            )

            self._wcs_filters = [
                f.strip() for f in self._config.get("default", "wcs_filters").split(",")
            ]

            self._wcs_timeout = self._config.getfloat(
                "default", "wcs_timeout", fallback=self._wcs_timeout
            )

        # Parse observatory
        self._observatory = os.path.join(self._config_path, "observatory.cfg")
        self._observatory = kwargs.get("observatory", self._observatory)
        if type(self._observatory) is str:
            logger.info(
                "Observatory is string, loading from config file and saving to config path"
            )
            self._observatory = Observatory(config_path=self._observatory)
            self.observatory.save_config(
                os.path.join(self._config_path, "observatory.cfg")
            )
        elif type(self._observatory) is Observatory:
            logger.info("Observatory is Observatory object, saving to config path")
            self.observatory.save_config(
                os.path.join(self._config_path, "observatory.cfg")
            )
        else:
            raise TelrunException(
                "observatory must be a string representing an observatory config file path or an Observatory object"
            )

        # Load kwargs
        self._schedules_path = os.path.abspath(
            kwargs.get("schedules_path", self._schedules_path)
        )
        self._images_path = os.path.abspath(
            kwargs.get("images_path", self._images_path)
        )
        self._logs_path = os.path.abspath(kwargs.get("logs_path", self._logs_path))

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
        self._config["default"]["dome_type"] = str(self._dome_type)

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
        self.write_to_schedule_log = kwargs.get(
            "write_to_schedule_log", self._write_to_schedule_log
        )
        self.write_to_status_log = kwargs.get(
            "write_to_status_log", self._write_to_status_log
        )
        self.autofocus_interval = kwargs.get(
            "autofocus_interval", self._autofocus_interval
        )
        self.initial_autofocus = kwargs.get(
            "initial_autofocus", self._initial_autofocus
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
        self.wait_for_block_start_time = kwargs.get(
            "wait_for_block_start_time", self._wait_for_block_start_time
        )
        self.max_block_late_time = kwargs.get(
            "max_block_late_time", self._max_block_late_time
        )
        self.preslew_time = kwargs.get("preslew_time", self._preslew_time)
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
        self.hardware_timeout = kwargs.get("hardware_timeout", self._hardware_timeout)
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
        atexit.register(self._terminate())
        logger.debug("Registered")

        # Open GUI if requested
        if self._gui:
            logger.info("Starting GUI")
            root = tk.Tk()
            root.tk.call("source", "../gui/themeSetup.tcl")
            root.tk.call("set_theme", "dark")
            # icon_photo = tk.PhotoImage(file='images/UILogo.png')
            # root.iconphoto(False, icon_photo)
            self._gui = _TelrunGUI(root, self, self.write_to_status_log)
            self._gui.mainloop()
            logger.info("GUI started")
        elif self.write_to_status_log:
            raise TelrunException("Cannot write to status log without GUI")

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
        if self.observatory.wcs is not None:
            self._wcs_status = "Idle"

        if save_at_end:
            self.save_config("telrun.cfg")

    def save_config(self, filename):
        logger.debug("Saving config to %s" % filename)
        self.observatory.save_config(os.path.join(self.config_path, "observatory.cfg"))
        with open(os.path.join(self.config_path, filename), "w") as config_file:
            self._config.write(config_file)

    def mainloop(self):
        if self.observatory.observing_conditions is not None:
            logger.info("Starting the observing_conditions update thread...")
            self._observing_conditions_status = "Update thread running"
            self.observatory.start_observing_conditions_thread()
            logger.info("Started.")

        logger.info("Starting main operation loop...")
        while True:
            # Check for new schedule
            filename = (
                self.schedules_path
                + "telrun_"
                + self.observatory.observatory_time().strftime("%m-%d-%Y")
                + ".ecsv"
            )
            if os.path.exists(filename):
                if (
                    os.path.getmtime(self.schedules_path + "telrun.ecsv")
                    > self._schedule_last_modified
                ):
                    logger.info("New schedule detected, reloading...")
                    self._schedule_last_modified = os.path.getmtime(filename)

                    if self._execution_thread is not None:
                        logger.info("Terminating current schedule execution thread...")
                        self._execution_event.set()
                        self._execution_thread.join()

                    logger.info("Clearing execution event...")
                    self._execution_event.clear()

                    logger.info("Reading new schedule...")
                    schedule = table.Table.read(filename, format="ascii.ecsv")

                    logger.info("Starting new schedule execution thread...")
                    self._execution_thread = threading.Thread(
                        target=self._execute_schedule,
                        args=(schedule, filename),
                        daemon=True,
                        name="Telrun Schedule Execution Thread",
                    )
                    self._execution_thread.start()
                    logger.info("Started.")

                else:
                    logger.debug("Waiting for new schedule...")
                    time.sleep(1)
            else:
                logger.debug("Waiting for new schedule...")
                time.sleep(1)

    def execute_schedule(self, schedule, *args):
        if schedule is str:
            schedule = table.Table.read(schedule, format="ascii.ecsv")
        elif type(schedule) is not table.QTable:
            logger.exception(
                "schedule must be a path to an ECSV file or an astropy QTable"
            )
            return

        # Schedule validation
        logger.info("Validating schedule...")
        if "target" not in schedule.colnames:
            logger.exception("schedule must have a target column")
            return
        if "start time (UTC)" not in schedule.colnames:
            logger.exception("schedule must have a start time (UTC) column")
            return
        if "end time (UTC)" not in schedule.colnames:
            logger.exception("schedule must have an end time (UTC) column")
            return
        if "duration (minutes)" not in schedule.colnames:
            logger.exception("schedule must have a duration (minutes) column")
            return
        if "ra" not in schedule.colnames:
            logger.exception("schedule must have an ra column")
            return
        if "dec" not in schedule.colnames:
            logger.exception("schedule must have a dec column")
            return
        if "configuration" not in schedule.colnames:
            logger.exception("schedule must have a configuration column")
            return

        logger.info("Validating observing blocks...")
        for i in range(len(schedule)):
            logger.debug("Validating block %i of %i" % (i + 1, len(schedule)))
            try:
                block = validate_ob(schedule[i])
            except Exception as e:
                logger.exception(
                    "Block %i of %i is invalid, removing from schedule"
                    % (i + 1, len(schedule))
                )
                logger.exception(e)
                block["configuration"]["status"] = "F"
                block["configuration"]["message"] = str(e)
            schedule[i] = block

        self._schedule = schedule

        if args[0] is not None:
            filename = args[0]
        else:
            filename = (
                self.schedules_path
                + "telrun_"
                + self.observatory.observatory_time().strftime("%m-%d-%Y")
                + ".ecsv"
            )

        if self.write_to_schedule_log:
            logger.info("Writing to schedule log.")
            if os.path.exists(
                self.logs_path + filename.split(".ecsv")[0].split("/")[-1] + "-log.ecsv"
            ):
                schedule_log = table.Table.read(
                    self.logs_path
                    + filename.split(".")[0].split("/")[-1]
                    + "-log.ecsv",
                    format="ascii.ecsv",
                )
            else:
                schedule_log = astroplan.Schedule(0, 0).to_table()

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
        if self.autofocus_interval > 0:
            self._do_periodic_autofocus = True

        if self.initial_autofocus and self.do_periodic_autofocus:
            self._last_autofocus_time = time.time() - self.autofocus_interval - 1

        for block_index, block in enumerate(self._schedule):
            if not self._execution_event.is_set():
                logger.info(
                    "Processing block %i of %i" % (block_index + 1, len(self._schedule))
                )
                logger.info(block)

                if block_index != 0:
                    self._previous_block = self._schedule[block_index - 1]
                else:
                    self._previous_block = None

                if block_index != len(self._schedule) - 1:
                    self._next_block = self._schedule[block_index + 1]
                else:
                    self._next_block = None

                status, message, block = self.execute_block(block)

                if status == "F":
                    self._skipped_block_count += 1

                self._schedule[block_index] = block

                self._schedule.write(filename, format="ascii.ecsv", overwrite=True)
                self._schedule_last_modified = os.path.getmtime(filename)

                if self.write_to_schedule_log:
                    schedule_log.add_row(block)
                    schedule_log.write(
                        self.logs_path
                        + filename.split(".ecsv")[0].split("/")[-1]
                        + "-log.ecsv",
                        format="ascii.ecsv",
                    )

            else:
                logger.info(
                    "Execution event has been set (likely by a new schedule detected), stopping block loop"
                )
                self._skipped_block_count = 0
                self._previous_block = None
                self._next_block = None
                return

        logger.info("Block loop complete")
        self._skipped_block_count = 0
        self._previous_block = None
        self._next_block = None
        return

        logger.info("Generating summary report")
        """summary_report(self.telpath+'/schedules/telrun.sls', self.telpath+'/logs/'+
            self._schedule[0].start_time.datetime.strftime('%m-%d-%Y')+'_telrun-report.txt')"""

        self._schedule = None

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

        # Turn ObservingBlock into Row
        if type(block) is astroplan.ObservingBlock:
            block = table.Row(
                [
                    block.target.name,
                    None,
                    None,
                    block.duration.minute,
                    block.target.ra.dms,
                    block.target.dec.dms,
                    block.configuration,
                ],
                names=(
                    "target",
                    "start time (UTC)",
                    "end time (UTC)",
                    "duration (minutes)",
                    "ra",
                    "dec",
                    "configuration",
                ),
            )
        elif type(block) is None:
            try:
                block = table.Row(
                    [
                        kwargs.get("target", None),
                        None,
                        None,
                        kwargs.get("duration", None),
                        kwargs["ra"],
                        kwargs["dec"],
                        kwargs["configuration"],
                    ],
                    names=(
                        "target",
                        "start time (UTC)",
                        "end time (UTC)",
                        "duration (minutes)",
                        "ra",
                        "dec",
                        "configuration",
                    ),
                )
            except:
                logger.exception(
                    "Passed block is None and no kwargs were given, skipping this block"
                )
                return
        elif type(block) is not table.Row:
            logger.exception(
                "Block must be an astroplan ObservingBlock or astropy Row object."
            )
            return

        # Logging setup for writing to FITS headers
        # From: https://stackoverflow.com/questions/31999627/storing-logger-messages-in-a-string
        str_output = StringIO()
        str_handler = logging.StreamHandler(str_output)
        str_handler.setLevel(logging.INFO)
        str_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        str_handler.setFormatter(str_formatter)
        logger.addHandler(str_handler)

        try:
            val_block = validate_ob(block)
        except Exception as e:
            logger.exception("Block failed validation, skipping...")
            logger.exception(e)
            return ("F", f"Block failed validation: {e}", block)

            logger.info("Block passed validation, continuing...")
            block = val_block

        self._current_block = block

        # Check 1: Block status?
        if self.check_block_status:
            if block["configuration"]["status"] != "N":
                logger.info("Block status is not N, skipping...")
                self._current_block = None
                if self.update_block_status:
                    block["configuration"]["status"] = "F"
                    block["configuration"][
                        "message"
                    ] = "Block already attempted to be processed"
                return ("F", "Block already attempted to be processed", block)

        # Check 2: Wait for block start time?
        if block["start time (UTC)"] is None:
            logger.info("No block start time, starting now...")
            block["start time (UTC)"] = astrotime.Time.now()

        seconds_until_start_time = (
            block["start time (UTC)"] - self.observatory.observatory_time()
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
            if self.update_block_status:
                block["configuration"]["status"] = "F"
                block["configuration"]["message"] = "Exceeded max_block_late_time"
            return ("F", "Exceeded max_block_late_time", block)
        elif (
            self.wait_for_block_start_time
            and seconds_until_start_time > self.max_block_late_time
        ):
            logger.info(
                "Block start time exceeded max_block_late_time of %i seconds, skipping..."
                % self.max_block_late_time
            )
            self._current_block = None
            if self.update_block_status:
                block["configuration"]["status"] = "F"
                block["configuration"]["message"] = "Exceeded max_block_late_time"
            return ("F", "Exceeded max_block_late_time", block)
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
                block["start time (UTC)"] - self.observatory.observatory_time()
            ).sec
        else:
            if seconds_until_start_time > 0:
                logger.info(
                    "Block start time in %.1f seconds" % seconds_until_start_time
                )

        # Check 3: Dome status?
        match self.dome_type:
            case "dome":
                if (
                    self.observatory.dome is not None
                    and self.observatory.dome.CanSetShutter
                ):
                    if self.observatory.dome.ShutterStatus != 0:
                        logger.info("Dome shutter is not open, skipping...")
                        self._current_block = None
                        if self.update_block_status:
                            block["configuration"]["status"] = "F"
                            block["configuration"][
                                "message"
                            ] = "Dome shutter is not open"
                        return ("F", "Dome shutter is not open", block)

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
                        if self.update_block_status:
                            block["configuration"]["status"] = "F"
                            block["configuration"][
                                "message"
                            ] = "Safety monitor indicates unsafe"
                        return ("F", "Dome safety monitor indicates unsafe", block)

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
                        if self.update_block_status:
                            block["configuration"]["status"] = "F"
                            block["configuration"][
                                "message"
                            ] = "Safety monitor indicates unsafe"
                        return ("F", "Dome safety monitor indicates unsafe", block)

                if (
                    self.observatory.dome is not None
                    and self.observatory.dome.CanSetShutter
                ):
                    if self.observatory.dome.ShutterStatus != 0:
                        logger.info("Dome shutter is not open, skipping...")
                        self._current_block = None
                        if self.update_block_status:
                            block["configuration"]["status"] = "F"
                            block["configuration"][
                                "message"
                            ] = "Dome shutter is not open"
                        return ("F", "Dome shutter is not open", block)

        # Check 4: Check safety monitors?
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
                if self.update_block_status:
                    block["configuration"]["status"] = "F"
                    block["configuration"][
                        "message"
                    ] = "Safety monitor indicates unsafe"
                return ("F", "Safety monitor indicates unsafe", block)

        # Check 5: Wait for sun?
        sun_alt_degs = self.observatory.sun_altaz()[0]
        if self.wait_for_sun and sun_alt_degs > self.max_solar_elev:
            logger.info(
                "Sun altitude: %.3f degs (above limit of %s), skipping..."
                % (sun_alt_degs, self.max_solar_elev)
            )
            self._current_block = None
            if self.update_block_status:
                block["configuration"]["status"] = "F"
                block["configuration"]["message"] = "Sun altitude above limit"
            return ("F", "Sun altitude above limit", block)

        # Check 6: Is autofocus needed?
        self._best_focus_result = None
        if (
            self.observatory.focuser is not None
            and self.do_periodic_autofocus
            and time.time() - self.last_autofocus_time > self.autofocus_interval
            and not block["configuration"]["do_not_interrupt"]
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
                self._last_autofocus_time = time.time()
                logger.info(
                    "Autofocus complete, best focus position: %i"
                    % self._best_focus_result
                )

        # Check 7: Camera temperature
        while (
            self.observatory.camera.CCDTemperature
            > self.observatory.cooler_setpoint + self.observatory.cooler_tolerance
            and self.wait_for_cooldown
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
            "CCD temperature: %.3f degs (below limit of %.3f with %.3f tolerance), continuing..."
            % (
                self.observatory.camera.CCDTemperature,
                self.observatory.cooler_setpoint,
                self.observatory.cooler_tolerance,
            )
        )
        self._camera_status = "Idle"

        # Is the previous target different?
        slew = True
        if self.previous_block is not None:
            if (
                self.previous_block["ra"].hourangle == block["ra"].hourangle
                and self.previous_block["dec"].deg == block["dec"].deg
                and self.previous_block["configuration"]["status"] == "S"
                and best_focus_result is not None
                and not block["configuration"]["do_not_interrupt"]
            ):
                logger.info(
                    "Previous target is same ra and dec, skipping initial slew..."
                )
                slew = False
            else:
                logger.info("Previous target is different ra and dec, slewing...")

                logger.info("Turning off tracking...")
                self.observatory.telescope.Tracking = False
                self.observatory.telescope.DeclinationRate = 0
                self.observatory.telescope.RightAscensionRate = 0

                if self.observatory.rotator is not None:
                    logger.info("Turning off derotation...")
                    self.observatory.stop_derotation_thread()

        # Update ephem for non-sidereal targets
        if (
            block["configuration"]["pm_ra_cosdec"].value != 0
            or block["configuration"]["pm_dec"].value != 0
        ):
            logger.info("Non-zero proper motion specified, updating ephemeris...")
            try:
                ephemerides = mpc.MPC.get_ephemeris(
                    target=block["name"],
                    location=self.observatory.observatory_location,
                    start=block["start time (UTC)"],
                    number=1,
                    proper_motion="sky",
                )
                block["ra"] = ephemerides["RA"][0]
                block["dec"] = ephemerides["Dec"][0]
                block["configuration"]["pm_ra_cosdec"] = (
                    ephemerides["dRA cos(Dec)"][0] * u.arcsec / u.hour
                )
                block["configuration"]["pm_dec"] = (
                    ephemerides["dDec"][0] * u.arcsec / u.hour
                )
            except Exception as e1:
                try:
                    logger.warning(
                        f"Failed to find proper motions for {row['name']} on line {line_number}, trying to find proper motions using astropy.coordinates.get_body: {e1}"
                    )
                    pos_l = coord.get_body(
                        block["name"],
                        block["start time (UTC)"] - 10 * u.minute,
                        location=self.observatory.observatory_location,
                    )
                    pos_m = coord.get_body(
                        block["name"], block["start time (UTC)"], location=location
                    )
                    pos_h = coord.get_body(
                        block["name"],
                        block["start time (UTC)"] + 10 * u.minute,
                        location=self.observatory.observatory_location,
                    )
                    block["ra"] = pos_m.ra.to_string(
                        "hourangle", sep="hms", precision=3
                    )
                    block["dec"] = pos_m.dec.to_string("deg", sep="dms", precision=2)
                    block["configuration"]["pm_ra_cosdec"] = (
                        (
                            pos_h.ra * np.cos(pos_h.dec.rad)
                            - pos_l.ra * np.cos(pos_l.dec.rad)
                        )
                        / (pos_h.obstime - pos_l.obstime)
                    ).to(u.arcsec / u.hour)
                    block["configuration"]["pm_dec"] = (
                        (pos_h.dec - pos_l.dec) / (pos_h.obstime - pos_l.obstime)
                    ).to(u.arcsec / u.hour)
                except Exception as e2:
                    logger.warning(
                        f"Failed to find proper motions for {block['name']}, keeping old ephemerides: {e2}"
                    )
            logger.info("Ephemeris updated")

        target = coord.SkyCoord(
            ra=block["ra"].hourangle, dec=block["dec"].deg, unit=(u.hourangle, u.deg)
        )

        # Perform centering if requested
        centered = None
        if None not in (
            block["configuration"]["respositioning"][0],
            block["configuration"]["respositioning"][1],
            block["ra"],
            block["dec"],
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
            self._wcs_status = "Recentering" if self.observatory.wcs is not None else ""
            self._dome_status = (
                "Recentering" if self.observatory.dome is not None else ""
            )
            self._rotator_status = (
                "Recentering" if self.observatory.rotator is not None else ""
            )
            centered = self.observatory.recenter(
                obj=target,
                target_x_pixel=block["configuration"]["respositioning"][0],
                target_y_pixel=block["configuration"]["respositioning"][1],
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
        elif slew and target is not None:
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
                obj=target,
                control_dome=(self.dome is not None),
                control_rotator=(self.rotator is not None),
                wait_for_slew=False,
                track=False,
            )
            self._status_event.set()
            t.join()
            self._status_event.clear()

        # Set filter and focus offset
        if self.filter_wheel is not None:
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
            self.observatory.set_filter_offset_focuser(
                filter_name=block["configuration"]["filter"]
            )
            self._status_event.set()
            t.join()
            self._status_event.clear()
            self._filter_wheel_status = "Idle"
            self._focuser_status = (
                "Idle" if self.observatory.focuser is not None else ""
            )

        # Set binning
        if (
            block["configuration"]["binning"][0] >= 1
            and block["configuration"]["binning"][0] <= self.observatory.camera.MaxBinX
        ):
            logger.info("Setting binx to %i" % block["configuration"]["binning"][0])
            self.observatory.camera.BinX = block["configuration"]["binning"][0]
        else:
            logger.warning(
                "Requested binx of %i is not supported, skipping..."
                % block["configuration"]["binning"][0]
            )
            self._current_block = None
            if self.update_block_status:
                block["configuration"]["status"] = "F"
                block["configuration"]["message"] = (
                    "Requested binx of %i is not supported"
                    % block["configuration"]["binning"][0]
                )
            return (
                "F",
                "Requested binx of %i is not supported"
                % block["configuration"]["binning"][0],
                block,
            )

        if (
            block["configuration"]["binning"][1] >= 1
            and block["configuration"]["binning"][1] <= self.observatory.camera.MaxBinY
            and (
                self.observatory.camera.CanAsymmetricBin
                or block["configuration"]["binning"][1]
                == block["configuration"]["binning"][0]
            )
        ):
            logger.info("Setting biny to %i" % block["configuration"]["binning"][1])
            self.observatory.camera.BinY = block["configuration"]["binning"][1]
        else:
            logger.warning(
                "Requested biny of %i is not supported, skipping..."
                % block["configuration"]["binning"][1]
            )
            self._current_block = None
            if self.update_block_status:
                block["configuration"]["status"] = "F"
                block["configuration"]["message"] = (
                    "Requested biny of %i is not supported"
                    % block["configuration"]["binning"][1]
                )
            return (
                "F",
                "Requested biny of %i is not supported"
                % block["configuration"]["binning"][1],
                block,
            )

        # Set subframe
        if block["configuration"]["frame_size"][0] == 0:
            block["configuration"]["frame_size"][0] = int(
                self.observatory.camera.CameraXSize / self.observatory.camera.BinX
            )
        if block["configuration"]["frame_size"][1] == 0:
            block["configuration"]["frame_size"][1] = int(
                self.observatory.camera.CameraYSize / self.observatory.camera.BinY
            )

        if block["configuration"]["frame_position"][0] + block["configuration"][
            "frame_size"
        ][0] < int(self.observatory.camera.CameraXSize / self.observatory.camera.BinX):
            logger.info(
                "Setting startx and numx to %i, %i"
                % (
                    block["configuration"]["frame_position"][0],
                    block["configuration"]["frame_size"][0],
                )
            )
            self.observatory.camera.StartX = block["configuration"]["frame_position"][0]
            self.observatory.camera.NumX = block["configuration"]["frame_size"][0]
        else:
            logger.warning(
                "Requested startx and numx of %i, %i is not supported, skipping..."
                % (
                    block["configuration"]["frame_position"][0],
                    block["configuration"]["frame_size"][0],
                )
            )
            return (
                "F",
                "Requested startx and numx of %i, %i is not supported"
                % (
                    block["configuration"]["frame_position"][0],
                    block["configuration"]["frame_size"][0],
                ),
            )

        if block["configuration"]["frame_position"][1] + block["configuration"][
            "frame_size"
        ][1] < int(self.observatory.camera.CameraYSize / self.observatory.camera.BinY):
            logger.info(
                "Setting starty and numy to %i, %i"
                % (
                    block["configuration"]["frame_position"][1],
                    block["configuration"]["frame_size"][1],
                )
            )
            self.observatory.camera.StartY = block["configuration"]["frame_position"][1]
            self.observatory.camera.NumY = block["configuration"]["frame_size"][1]
        else:
            logger.warning(
                "Requested starty and numy of %i, %i is not supported, skipping..."
                % (
                    block["configuration"]["frame_position"][1],
                    block["configuration"]["frame_size"][1],
                )
            )
            self._current_block = None
            if self.update_block_status:
                block["configuration"]["status"] = "F"
                block["configuration"]["message"] = (
                    "Requested starty and numy of %i, %i is not supported"
                    % (
                        block["configuration"]["frame_position"][1],
                        block["configuration"]["frame_size"][1],
                    )
                )
            return (
                "F",
                "Requested starty and numy of %i, %i is not supported"
                % (
                    block["configuration"]["frame_position"][1],
                    block["configuration"]["frame_size"][1],
                ),
                block,
            )

        # Set readout mode
        try:
            logger.info(
                "Setting readout mode to %i" % block["configuration"]["readout"]
            )
            self.observatory.camera.ReadoutMode = block["configuration"]["readout"]
        except:
            logger.warning(
                "Requested readout mode of %i is not supported, setting to default of %i"
                % (block["configuration"]["readout"], self.default_readout)
            )
            self.observatory.camera.ReadoutMode = self.default_readout

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

        # Start tracking
        logger.info("Starting tracking...")
        self._telescope_status = "Tracking"
        self.observatory.telescope.Tracking = True

        # Check for pm exceeding two pixels in one hour
        if block["configuration"]["pm_ra_cosdec"].to_value(
            u.arcsec / u.second
        ) > 2 * self.observatory.pixel_scale[0] / (60 * 60) or block["configuration"][
            "pm_dec"
        ].to_value(
            u.arcsec / u.second
        ) > 2 * self.observatory.pixel_scale[
            1
        ] / (
            60 * 60
        ):
            logger.info("Switching to non-sidereal tracking...")
            self._telescope_status = "Non-sidereal tracking"
            self.observatory.mount.RightAscensionRate = (
                block["configuration"]["pm_ra_cosdec"].to_value(u.arcsec / u.second)
                * 0.997269567
                / 15.041
                * (1 / np.cos(block["dec"].rad))
            )
            self.observatory.mount.DeclinationRate = block["configuration"][
                "pm_dec"
            ].to_value(u.arcsec / u.second)
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

        # Wait for focuser, dome motion to complete
        condition = True
        logger.info("Waiting for focuser or dome motion to complete...")
        while condition:
            if self.observatory.focuser is not None:
                condition = self.observatory.focuser.IsMoving
                if not condition:
                    self._focuser_status = "Idle"
            if self.observatory.dome is not None:
                if not self.observatory.Slewing:
                    self._dome_status = "Idle"
                condition = condition or self.observatory.dome.Slewing
            time.sleep(0.1)

        # If still time before block start, wait
        seconds_until_start_time = (
            block["start time (UTC)"] - self.observatory.observatory_time()
        ).sec
        if seconds_until_start_time > 0 and self.wait_for_block_start_time:
            logging.info(
                "Waiting %.1f seconds until start time" % seconds_until_start_time
            )
            time.sleep(seconds_until_start_time - 0.1)

        # Define custom header
        custom_header = {
            "OBSERVER": (block["configuration"]["observer"], "Name of observer"),
            "OBSCODE": (block["configuration"]["code"], "Observing code"),
            "TARGET": (block["target"], "Name of target if provided"),
            "SCHEDTIT": (block["configuration"]["title"], "Title if provided"),
            "SCHEDCOM": (block["configuration"]["comment"], "Comment if provided"),
            "SCHEDRA": (block["ra"].to_string("hms"), "Requested RA"),
            "SCHEDDEC": (block["dec"].to_string("dms"), "Requested Dec"),
            "SCHEDPRA": (
                block["configuration"]["pm_ra_cosdec"].to_value(u.arsec / u.hour),
                "Requested proper motion in RAcosDec [arcsec/hr]",
            ),
            "SCHEDPDEC": (
                block["configuration"]["pm_dec"].to_value(u.arcsec / u.hour),
                "Requested proper motion in Dec [arcsec/hr]",
            ),
            "SCHEDSRT": (block["start time (UTC)"].fits, "Requested start time"),
            "SCHEDINT": (
                block["configuration"]["do_not_interrupt"],
                "Whether the block can be interrupted by autofocus or other blocks",
            ),
            "CENTERED": (
                centered,
                "Whether the target underwent the centering routine",
            ),
            "SCHEDPSX": (
                block["configuration"]["respositioning"][0],
                "Requested x pixel for recentering",
            ),
            "SCHEDPSY": (
                block["configuration"]["respositioning"][1],
                "Requested y pixel for recentering",
            ),
            "LASTAUTO": (
                self.last_autofocus_time,
                "When the last autofocus was performed",
            ),
        }

        # Start exposures
        for i in range(block["configuration"]["nexp"]):
            logger.info(
                "Beginning exposure %i of %i" % (i + 1, block["configuration"]["nexp"])
            )
            logger.info(
                "Starting %4.4g second exposure..." % block["configuration"]["exposure"]
            )
            self._camera_status = "Exposing"
            t0 = time.time()
            self.observatory.camera.Expose(
                block["configuration"]["exposure"],
                block["configuration"]["shutter_state"],
            )
            logger.info("Waiting for image...")
            while (
                not self.observatory.camera.ImageReady
                and time.time()
                < t0 + block["configuration"]["exposure"] + self.hardware_timeout
            ):
                time.sleep(0.1)
            self._camera_status = "Idle"

            # Append integer to filename if multiple exposures
            if block["configuration"]["nexp"] > 1:
                block["configuration"]["filename"] = (
                    block["configuration"]["filename"] + "_%i" % i
                )

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
                        self.images_path + block["configuration"]["filename"] + ".tmp",
                        frametyp=block["configuration"]["shutter_state"],
                        custom_header=custom_header,
                        history=hist,
                    )
                    self._wcs_threads.append(
                        threading.Thread(
                            target=self._async_wcs_solver,
                            args=(
                                self.images_path
                                + block["configuration"]["filename"]
                                + ".tmp",
                            ),
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
                        self.images_path + block["configuration"]["filename"],
                        frametyp=block["configuration"]["shutter_state"],
                        custom_header=custom_header,
                        history=hist,
                    )
            else:
                logger.info("No filter wheel, attempting WCS solve...")
                hist = str_output.getvalue().split("\n")
                save_success = self.observatory.save_last_image(
                    self.images_path + block["configuration"]["filename"] + ".tmp",
                    frametyp=block["configuration"]["shutter_state"],
                    custom_header=custom_header,
                    history=hist,
                )
                self._wcs_threads.append(
                    threading.Thread(
                        target=self._async_wcs_solver,
                        args=(
                            self.images_path
                            + block["configuration"]["filename"]
                            + ".tmp",
                        ),
                        daemon=True,
                        name="wcs_threads",
                    )
                )
                self._wcs_threads[-1].start()

        # If multiple exposures, update filename as a list
        if block["configuration"]["nexp"] > 1:
            block["configuration"]["filename"] = [
                block["configuration"]["filename"] + "_%i" % i
                for i in range(block["configuration"]["nexp"])
            ]

        # Set block status to done
        self._current_block = None
        if self.update_block_status:
            block["configuration"]["status"] = "S"
            block["configuration"]["message"] = "Success"
        return ("S", "Success", block)

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
            logger.warning("Process timed out after %.1f seconds" % timeout)
            # TODO: Add auto-recovery capability for the affected hardware

    def _terminate(self):
        self.observatory.shutdown()

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
    def telpath(self):
        return self._telpath

    @property
    def observatory(self):
        return self._observatory

    @property
    def dome_type(self):
        return self._dome_type

    @property
    def initial_home(self):
        return self._initial_home

    @initial_home.setter
    def initial_home(self, value):
        self._initial_home = bool(value)
        self._config["default"]["initial_home"] = str(self._initial_home)

    @property
    def wait_for_sun(self):
        return self._wait_for_sun

    @wait_for_sun.setter
    def wait_for_sun(self, value):
        self._wait_for_sun = bool(value)
        self._config["default"]["wait_for_sun"] = str(self._wait_for_sun)

    @property
    def max_solar_elev(self):
        return self._max_solar_elev

    @max_solar_elev.setter
    def max_solar_elev(self, value):
        self._max_solar_elev = float(value)
        self._config["default"]["max_solar_elev"] = str(self._max_solar_elev)

    @property
    def check_safety_monitors(self):
        return self._check_safety_monitors

    @check_safety_monitors.setter
    def check_safety_monitors(self, value):
        self._check_safety_monitors = bool(value)
        self._config["default"]["check_safety_monitors"] = str(
            self._check_safety_monitors
        )

    @property
    def _wait_for_cooldown(self):
        return self._wait_for_cooldown

    @_wait_for_cooldown.setter
    def _wait_for_cooldown(self, value):
        self._wait_for_cooldown = bool(value)
        self._config["default"]["wait_for_cooldown"] = str(self._wait_for_cooldown)

    @property
    def default_readout(self):
        return self._default_readout

    @default_readout.setter
    def default_readout(self, value):
        self._default_readout = int(value)
        self._config["default"]["default_readout"] = str(self._default_readout)

    @property
    def check_block_status(self):
        return self._check_block_status

    @check_block_status.setter
    def check_block_status(self, value):
        self._check_block_status = bool(value)
        self._config["default"]["check_block_status"] = str(self._check_block_status)

    @property
    def update_block_status(self):
        return self._update_block_status

    @update_block_status.setter
    def update_block_status(self, value):
        self._update_block_status = bool(value)
        self._config["default"]["update_block_status"] = str(self._update_block_status)

    @property
    def write_to_schedule_log(self):
        return self._write_to_schedule_log

    @write_to_schedule_log.setter
    def write_to_schedule_log(self, value):
        self._write_to_schedule_log = bool(value)
        self._config["default"]["write_to_schedule_log"] = str(
            self._write_to_schedule_log
        )

    @property
    def write_to_status_log(self):
        return self._write_to_status_log

    @write_to_status_log.setter
    def write_to_status_log(self, value):
        self._write_to_status_log = bool(value)
        self._config["default"]["write_to_status_log"] = str(self._write_to_status_log)

    @property
    def autofocus_interval(self):
        return self._autofocus_interval

    @autofocus_interval.setter
    def autofocus_interval(self, value):
        self._autofocus_interval = float(value)
        self._config["default"]["autofocus_interval"] = str(self._autofocus_interval)

    @property
    def initial_autofocus(self):
        return self._initial_autofocus

    @initial_autofocus.setter
    def initial_autofocus(self, value):
        self._initial_autofocus = bool(value)
        self._config["default"]["initial_autofocus"] = str(self._initial_autofocus)

    @property
    def autofocus_filters(self):
        return self._autofocus_filters

    @autofocus_filters.setter
    def autofocus_filters(self, value):
        self._autofocus_filters = iter(value)
        for v in value:
            self._config["default"]["autofocus_filters"] += str(v) + ","

    @property
    def autofocus_exposure(self):
        return self._autofocus_exposure

    @autofocus_exposure.setter
    def autofocus_exposure(self, value):
        self._autofocus_exposure = float(value)
        self._config["default"]["autofocus_exposure"] = str(self._autofocus_exposure)

    @property
    def autofocus_midpoint(self):
        return self._autofocus_midpoint

    @autofocus_midpoint.setter
    def autofocus_midpoint(self, value):
        self._autofocus_midpoint = float(value)
        self._config["default"]["autofocus_midpoint"] = str(self._autofocus_midpoint)

    @property
    def autofocus_nsteps(self):
        return self._autofocus_nsteps

    @autofocus_nsteps.setter
    def autofocus_nsteps(self, value):
        self._autofocus_nsteps = int(value)
        self._config["default"]["autofocus_nsteps"] = str(self._autofocus_nsteps)

    @property
    def autofocus_step_size(self):
        return self._autofocus_step_size

    @autofocus_step_size.setter
    def autofocus_step_size(self, value):
        self._autofocus_step_size = int(value)
        self._config["default"]["autofocus_step_size"] = str(self._autofocus_step_size)

    @property
    def autofocus_use_current_pointing(self):
        return self._autofocus_use_current_pointing

    @autofocus_use_current_pointing.setter
    def autofocus_use_current_pointing(self, value):
        self._autofocus_use_current_pointing = bool(value)
        self._config["default"]["autofocus_use_current_pointing"] = str(
            self._autofocus_use_current_pointing
        )

    @property
    def autofocus_timeout(self):
        return self._autofocus_timeout

    @autofocus_timeout.setter
    def autofocus_timeout(self, value):
        self._autofocus_timeout = float(value)
        self._config["default"]["autofocus_timeout"] = str(self._autofocus_timeout)

    @property
    def wait_for_block_start_time(self):
        return self._wait_for_block_start_time

    @wait_for_block_start_time.setter
    def wait_for_block_start_time(self, value):
        self._wait_for_block_start_time = bool(value)
        self._config["default"]["wait_for_block_start_time"] = str(
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
        self._config["default"]["max_block_late_time"] = str(self._max_block_late_time)

    @property
    def preslew_time(self):
        return self._preslew_time

    @preslew_time.setter
    def preslew_time(self, value):
        self._preslew_time = float(value)
        self._config["default"]["preslew_time"] = str(self._preslew_time)

    @property
    def recenter_filters(self):
        return self._recenter_filters

    @recenter_filters.setter
    def recenter_filters(self, value):
        self._recenter_filters = iter(value)
        for v in value:
            self._config["default"]["recenter_filters"] += str(v) + ","

    @property
    def recenter_initial_offset_dec(self):
        return self._recenter_initial_offset_dec

    @recenter_initial_offset_dec.setter
    def recenter_initial_offset_dec(self, value):
        self._recenter_initial_offset_dec = float(value)
        self._config["default"]["recenter_initial_offset_dec"] = str(
            self._recenter_initial_offset_dec
        )

    @property
    def recenter_check_and_refine(self):
        return self._recenter_check_and_refine

    @recenter_check_and_refine.setter
    def recenter_check_and_refine(self, value):
        self._recenter_check_and_refine = bool(value)
        self._config["default"]["recenter_check_and_refine"] = str(
            self._recenter_check_and_refine
        )

    @property
    def recenter_max_attempts(self):
        return self._recenter_max_attempts

    @recenter_max_attempts.setter
    def recenter_max_attempts(self, value):
        self._recenter_max_attempts = int(value)
        self._config["default"]["recenter_max_attempts"] = str(
            self._recenter_max_attempts
        )

    @property
    def recenter_tolerance(self):
        return self._recenter_tolerance

    @recenter_tolerance.setter
    def recenter_tolerance(self, value):
        self._recenter_tolerance = float(value)
        self._config["default"]["recenter_tolerance"] = str(self._recenter_tolerance)

    @property
    def recenter_exposure(self):
        return self._recenter_exposure

    @recenter_exposure.setter
    def recenter_exposure(self, value):
        self._recenter_exposure = float(value)
        self._config["default"]["recenter_exposure"] = str(self._recenter_exposure)

    @property
    def recenter_save_images(self):
        return self._recenter_save_images

    @recenter_save_images.setter
    def recenter_save_images(self, value):
        self._recenter_save_images = bool(value)
        self._config["default"]["recenter_save_images"] = str(
            self._recenter_save_images
        )

    @property
    def recenter_save_path(self):
        return self._recenter_save_path

    @recenter_save_path.setter
    def recenter_save_path(self, value):
        self._recenter_save_path = os.path.abspath(value)
        self._config["default"]["recenter_save_path"] = str(self._recenter_save_path)

    @property
    def recenter_sync_mount(self):
        return self._recenter_sync_mount

    @recenter_sync_mount.setter
    def recenter_sync_mount(self, value):
        self._recenter_sync_mount = bool(value)
        self._config["default"]["recenter_sync_mount"] = str(self._recenter_sync_mount)

    @property
    def hardware_timeout(self):
        return self._hardware_timeout

    @hardware_timeout.setter
    def hardware_timeout(self, value):
        self._hardware_timeout = float(value)
        self._config["default"]["hardware_timeout"] = str(self._hardware_timeout)

    @property
    def wcs_filters(self):
        return self._wcs_filters

    @wcs_filters.setter
    def wcs_filters(self, value):
        self._wcs_filters = iter(value)
        for v in value:
            self._config["default"]["wcs_filters"] += str(v) + ","

    @property
    def wcs_timeout(self):
        return self._wcs_timeout

    @wcs_timeout.setter
    def wcs_timeout(self, value):
        self._wcs_timeout = float(value)
        self._config["default"]["wcs_timeout"] = str(self._wcs_timeout)


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
                "Target",
                "Start Time",
                "End Time",
                "Duration",
                "RA",
                "Dec",
                "Observer",
                "Observer Code",
                "Title",
                "Filename",
                "Filter",
                "Exposure",
                "Num Exposures",
                "Do Not Interrupt",
                "Repositioning",
                "Shutter State",
                "Readout",
                "Binning",
                "Frame Position",
                "Frame Size",
                "PM RAcosDec",
                "PM Dec",
                "Comment",
                "Status",
                "Status Message",
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
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
                        self._telrun.schedule[i]["target"],
                        self._telrun.schedule[i]["start time (UTC)"].iso,
                        self._telrun.schedule[i]["end time (UTC)"].iso,
                        self._telrun.schedule[i]["duration (minutes)"] * 60,
                        self._telrun.schedule[i]["ra"].hms,
                        self._telrun.schedule[i]["dec"].dms,
                        self._telrun.schedule[i]["configuration"]["observer"],
                        self._telrun.schedule[i]["configuration"]["code"],
                        self._telrun.schedule[i]["configuration"]["title"],
                        self._telrun.schedule[i]["configuration"]["filename"],
                        self._telrun.schedule[i]["configuration"]["filter"],
                        self._telrun.schedule[i]["configuration"]["exposure"],
                        self._telrun.schedule[i]["configuration"]["nexp"],
                        self._telrun.schedule[i]["configuration"]["do_not_interrupt"],
                        self._telrun.schedule[i]["configuration"]["repositioning"][0]
                        + ","
                        + self._telrun.schedule[i]["configuration"]["repositioning"][1],
                        self._telrun.schedule[i]["configuration"]["shutter_state"],
                        self._telrun.schedule[i]["configuration"]["readout"],
                        self._telrun.schedule[i]["configuration"]["binning"][0]
                        + "x"
                        + self._telrun.schedule[i]["configuration"]["binning"][1],
                        self._telrun.schedule[i]["configuration"]["frame_position"][0]
                        + ","
                        + self._telrun.schedule[i]["configuration"]["frame_position"][
                            1
                        ],
                        self._telrun.schedule[i]["configuration"]["frame_size"][0]
                        + "x"
                        + self._telrun.schedule[i]["configuration"]["frame_size"][1],
                        self._telrun.schedule[i]["configuration"][
                            "pm_ra_cosdec"
                        ].to_string(),
                        self._telrun.schedule[i]["configuration"]["pm_dec"].to_string(),
                        self._telrun.schedule[i]["configuration"]["comment"],
                        self._telrun.schedule[i]["status"],
                        self._telrun.schedule[i]["message"],
                    ]
                    for i in range(len(self._telrun.schedule))
                ]
            )

        if self._telrun.write_to_status_log:
            with open(os.path.join(self._telrun.log_path, "telrun_status.log"), "w"):
                f.write("# System Status")
                for row in self.system_status_widget.rows:
                    for i in range(len(row.labels)):
                        f.write(row.labels[i] + " " + row.string_vars[i].get() + "\n")

                f.write("\n# Previous Block")
                for i in range(len(previous_block_widget.rows.labels)):
                    f.write(
                        previous_block_widget.rows.labels[i]
                        + " "
                        + previous_block_widget.rows.string_vars[i].get()
                        + "\n"
                    )

                f.write("\n# Current Block")
                for i in range(len(current_block_widget.rows.labels)):
                    f.write(
                        current_block_widget.rows.labels[i]
                        + " "
                        + current_block_widget.rows.string_vars[i].get()
                        + "\n"
                    )

                f.write("\n# Next Block")
                for i in range(len(next_block_widget.rows.labels)):
                    f.write(
                        next_block_widget.rows.labels[i]
                        + " "
                        + next_block_widget.rows.string_vars[i].get()
                        + "\n"
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
        self.operator_mode = rows0.add_row("Operator Mode:")
        self.sun_elevation = rows0.add_row("Sun Elevation:")
        self.moon_elevation = rows0.add_row("Moon Elevation:")
        self.moon_illumination = rows0.add_row("Moon Illumination:")
        self.lst = rows0.add_row("LST:")
        self.ut = rows0.add_row("UT:")
        self.last_autofocus_time = rows0.add_row("Last Autofocus Time:")
        self.time_until_next_autofocus = rows0.add_row("Time Until Next Autofocus:")
        self.time_until_block_start = rows0.add_row("Time Until Block Start:")
        self.skipped_block_count = rows0.add_row("Skipped Block Count:")
        self.total_block_count = rows0.add_row("Total Block Count:")
        self.schedule_last_modified = rows0.add_row("Schedule Last Modified:")

        rows1 = _Rows(self, 2)
        self.autofocus_status = rows1.add_row("Autofocus Status:")
        self.camera_status = rows1.add_row("Camera Status:")
        self.cover_calibrator_status = rows1.add_row("Cover Calibrator Status:")
        self.dome_status = rows1.add_row("Dome Status:")
        self.filter_wheel_status = rows1.add_row("Filter Wheel Status:")
        self.focuser_status = rows1.add_row("Focuser Status:")
        self.observing_conditions_status = rows1.add_row("Observing Conditions Status:")
        self.rotator_status = rows1.add_row("Rotator Status:")
        self.safety_monitor_status = rows1.add_row("Safety Monitor Status:")
        self.switch_status = rows1.add_row("Switch Status:")
        self.telescope_status = rows1.add_row("Telescope Status:")
        self.wcs_status = rows1.add_row("WCS Status:")

        rows2 = _Rows(self, 4)
        self.cloud_cover = rows2.add_row("Cloud Cover:")
        self.dew_point = rows2.add_row("Dew Point:")
        self.humidity = rows2.add_row("Humidity:")
        self.pressure = rows2.add_row("Pressure:")
        self.rainrate = rows2.add_row("Rain Rate:")
        self.sky_brightness = rows2.add_row("Sky Brightness:")
        self.sky_quality = rows2.add_row("Sky Quality:")
        # self.sky_temperature = rows2.add_row('Sky Temperature:')
        self.star_fwhm = rows2.add_row("Star FWHM:")
        self.temperature = rows2.add_row("Temperature:")
        self.wind_direction = rows2.add_row("Wind Direction:")
        self.wind_gust = rows2.add_row("Wind Gust:")
        self.wind_speed = rows2.add_row("Wind Speed:")

        rows3 = _Rows(self, 6)
        self.telpath = rows3.add_row("Telhome:")
        self.site_name = rows3.add_row("Site Name:")
        self.dome_type = rows3.add_row("Dome Type:")
        self.initial_home = rows3.add_row("Initial Home:")
        self.wait_for_sun = rows3.add_row("Wait For Sun:")
        self.max_solar_elev = rows3.add_row("Max Solar Elevation:")
        self.check_safety_monitors = rows3.add_row("Check Safety Monitors:")
        self.wait_for_cooldown = rows3.add_row("Wait For Cooldown:")
        self.default_readout = rows3.add_row("Default Readout:")
        self.check_block_status = rows3.add_row("Check Block Status:")
        self.update_block_status = rows3.add_row("Update Block Status:")
        self.write_to_schedule_log = rows3.add_row("Write To Schedule Log:")
        self.write_to_status_log = rows3.add_row("Write To Status Log:")

        rows4 = _Rows(self, 8)
        self.autofocus_interval = rows4.add_row("Autofocus Interval:")
        self.autofocus_exposure = rows4.add_row("Autofocus Exposure:")
        self.autofocus_midpoint = rows4.add_row("Autofocus Midpoint:")
        self.autofocus_nsteps = rows4.add_row("Autofocus NSteps:")
        self.autofocus_step_size = rows4.add_row("Autofocus Step Size:")
        self.autofocus_use_current_pointing = rows4.add_row(
            "Autofocus Use Current Pointing:"
        )
        self.autofocus_timeout = rows4.add_row("Autofocus Timeout:")
        self.wait_for_block_start_time = rows4.add_row("Wait For Block Start Time:")
        self.max_block_late_time = rows4.add_row("Max Block Late Time:")
        self.preslew_time = rows4.add_row("Preslew Time:")

        rows5 = _Rows(self, 10)
        self.recenter_filters = rows5.add_row("Recenter Filters:")
        self.recenter_initial_offset_dec = rows5.add_row("Recenter Initial Offset Dec:")
        self.recenter_check_and_refine = rows5.add_row("Recenter Check And Refine:")
        self.recenter_max_attempts = rows5.add_row("Recenter Max Attempts:")
        self.recenter_tolerance = rows5.add_row("Recenter Tolerance:")
        self.recenter_exposure = rows5.add_row("Recenter Exposure:")
        self.recenter_save_images = rows5.add_row("Recenter Save Images:")
        self.recenter_save_path = rows5.add_row("Recenter Save Path:")
        self.recenter_sync_mount = rows5.add_row("Recenter Sync Mount:")
        self.hardware_timeout = rows5.add_row("Hardware Timeout:")
        self.wcs_filters = rows5.add_row("WCS Filters:")
        self.wcs_timeout = rows5.add_row("WCS Timeout:")

        self.rows = [rows0, rows1, rows2, rows3, rows4, rows5]

    def update(self):
        if self._parent._telrun._execution_thread is not None:
            operator_mode = "Fully robotic"
        else:
            operator_mode = "Interactive"
        self.operator_mode.set(operator_mode)
        self.sun_elevation.set(str(self._parent._telrun.observatory.sun_altaz()[0]))
        self.moon_elevation.set(str(self._parent._telrun.observatory.moon_altaz()[0]))
        self.moon_illumination.set(
            str(self._parent._telrun.observatory.moon_illumination())
        )
        self.lst.set(self._parent._telrun.observatory.lst().iso)
        self.ut.set(self._parent._telrun.observatory.observatory_time.iso)
        self.last_autofocus_time.set(
            astrotime.Time(self._parent._telrun.last_autofocus_time, format="unix").iso
        )
        self.time_until_next_autofocus.set(
            str(
                self._parent._telrun.last_autofocus_time
                + self._parent._telrun.autofocus_interval
                - time.time()
            )
        )
        self.time_until_block_start.set(
            (
                self._parent._telrun.current_block["start time (UTC)"]
                - self._parent._telrun.observatory.observatory_time()
            ).second
            if self._parent._telrun.current_block is not None
            else ""
        )
        self.skipped_block_count.set(str(self._parent._telrun.skipped_block_count))
        self.total_block_count.set(str(len(self._parent._telrun._schedule)))
        self.schedule_last_modified.set(
            str(
                astrotime.Time(
                    self._parent._telrun.schedule_last_modified, format="unix"
                ).iso
            )
        )

        self.autofocus_status.set(self._parent._telrun.autofocus_status)
        self.camera_status.set(self._parent._telrun.camera_status)
        self.cover_calibrator_status.set(self._parent._telrun.cover_calibrator_status)
        self.dome_status.set(self._parent._telrun.dome_status)
        self.filter_wheel_status.set(self._parent._telrun.filter_wheel_status)
        self.focuser_status.set(self._parent._telrun.focuser_status)
        self.observing_conditions_status.set(
            self._parent._telrun.observing_conditions_status
        )
        self.rotator_status.set(self._parent._telrun.rotator_status)
        self.safety_monitor_status.set(self._parent._telrun.safety_monitor_status)
        self.switch_status.set(self._parent._telrun.switch_status)
        self.telescope_status.set(self._parent._telrun.telescope_status)
        self.wcs_status.set(self._parent._telrun.wcs_status)

        self.cloud_cover.set(str(self._parent._telrun.observing_conditions.CloudCover))
        self.dew_point.set(str(self._parent._telrun.observing_conditions.DewPoint))
        self.humidity.set(str(self._parent._telrun.observing_conditions.Humidity))
        self.pressure.set(str(self._parent._telrun.observing_conditions.Pressure))
        self.rain_rate.set(str(self._parent._telrun.observing_conditions.RainRate))
        self.sky_brightness.set(
            str(self._parent._telrun.observing_conditions.SkyBrightness)
        )
        self.sky_quality.set(str(self._parent._telrun.observing_conditions.SkyQuality))
        self.star_fwhm.set(str(self._parent._telrun.observing_conditions.StarFWHM))
        self.temperature.set(str(self._parent._telrun.observing_conditions.Temperature))
        self.wind_direction.set(
            str(self._parent._telrun.observing_conditions.WindDirection)
        )
        self.wind_gust.set(str(self._parent._telrun.observing_conditions.WindGust))
        self.wind_speed.set(str(self._parent._telrun.observing_conditions.WindSpeed))

        self.telpath.set(self._parent._telrun.telpath)
        self.site_name.set(self._parent._telrun.observatory.site_name)
        self.dome_type.set(self._parent._telrun.dome_type)
        self.initial_home.set(str(self._parent._telrun.initial_home))
        self.wait_for_sun.set(str(self._parent._telrun.wait_for_sun))
        self.max_solar_elev.set(str(self._parent._telrun.max_solar_elev))
        self.check_safety_monitors.set(str(self._parent._telrun.check_safety_monitors))
        self.wait_for_cooldown.set(str(self._parent._telrun.wait_for_cooldown))
        self.default_readout.set(str(self._parent._telrun.default_readout))
        self.check_block_status.set(str(self._parent._telrun.check_block_status))
        self.update_block_status.set(str(self._parent._telrun.update_block_status))
        self.write_to_schedule_log.set(str(self._parent._telrun.write_to_schedule_log))
        self.write_to_status_log.set(str(self._parent._telrun.write_to_status_log))

        self.autofocus_interval.set(str(self._parent._telrun.autofocus_interval))
        self.autofocus_exposure.set(str(self._parent._telrun.autofocus_exposure))
        self.autofocus_midpoint.set(str(self._parent._telrun.autofocus_midpoint))
        self.autofocus_nsteps.set(str(self._parent._telrun.autofocus_nsteps))
        self.autofocus_step_size.set(str(self._parent._telrun.autofocus_step_size))
        self.autofocus_use_current_pointing.set(
            str(self._parent._telrun.autofocus_use_current_pointing)
        )
        self.autofocus_timeout.set(str(self._parent._telrun.autofocus_timeout))
        self.wait_for_block_start_time.set(
            str(self._parent._telrun.wait_for_block_start_time)
        )
        self.max_block_late_time.set(str(self._parent._telrun.max_block_late_time))
        self.preslew_time.set(str(self._parent._telrun.preslew_time))

        self.recenter_filters.set(str(self._parent._telrun.recenter_filters))
        self.recenter_initial_offset_dec.set(
            str(self._parent._telrun.recenter_initial_offset_dec)
        )
        self.recenter_check_and_refine.set(
            str(self._parent._telrun.recenter_check_and_refine)
        )
        self.recenter_max_attempts.set(str(self._parent._telrun.recenter_max_attempts))
        self.recenter_tolerance.set(str(self._parent._telrun.recenter_tolerance))
        self.recenter_exposure.set(str(self._parent._telrun.recenter_exposure))
        self.recenter_save_images.set(str(self._parent._telrun.recenter_save_images))
        self.recenter_save_path.set(str(self._parent._telrun.recenter_save_path))
        self.recenter_sync_mount.set(str(self._parent._telrun.recenter_sync_mount))
        self.hardware_timeout.set(str(self._parent._telrun.hardware_timeout))
        self.wcs_filters.set(str(self._parent._telrun.wcs_filters))
        self.wcs_timeout.set(str(self._parent._telrun.wcs_timeout))


class _BlockWidget(ttk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self._parent = parent

        self.build_gui()
        self.update()

    def build_gui(self):
        self.rows = _Rows(self, 0)

        self.target = rows.add_row("Target:")
        self.start_time = rows.add_row("Start Time:")
        self.duration = rows.add_row("Duration:")
        self.ra = rows.add_row("RA:")
        self.dec = rows.add_row("Dec:")
        self.observer = rows.add_row("Observer:")
        self.code = rows.add_row("Observer Code:")
        self.title = rows.add_row("Title:")
        self.filename = rows.add_row("Filename:")
        self.filter = rows.add_row("Filter:")
        self.exposure = rows.add_row("Exposure:")
        self.nexp = rows.add_row("N Exp:")
        self.do_not_interrupt = rows.add_row("Do Not Interrupt:")
        self.respositioning = rows.add_row("Respositioning:")
        self.shutter_state = rows.add_row("Shutter State:")
        self.readout = rows.add_row("Readout:")
        self.binning = rows.add_row("Binning:")
        self.frame_position = rows.add_row("Frame Position:")
        self.frame_size = rows.add_row("Frame Size:")
        self.pm_ra_cosdec = rows.add_row("PM RA cosDec:")
        self.pm_dec = rows.add_row("PM Dec:")
        self.comment = rows.add_row("Comment:")
        self.status = rows.add_row("Status:")
        self.message = rows.add_row("Status Message:")

    def update(self, block):
        if block is None:
            self.target.set("")
            self.start_time.set("")
            self.duration.set("")
            self.ra.set("")
            self.dec.set("")
            self.observer.set("")
            self.code.set("")
            self.title.set("")
            self.filename.set("")
            self.filter.set("")
            self.exposure.set("")
            self.nexp.set("")
            self.do_not_interrupt.set("")
            self.respositioning.set("")
            self.shutter_state.set("")
            self.readout.set("")
            self.binning.set("")
            self.frame_position.set("")
            self.frame_size.set("")
            self.pm_ra_cosdec.set("")
            self.pm_dec.set("")
            self.comment.set("")
            self.status.set("")
            self.message.set("")
        else:
            self.target.set(block["target"])
            self.start_time.set(block["start_time"].iso)
            self.duration.set(block["duration"].second)
            self.ra.set(block["ra"].to_string("hms"))
            self.dec.set(block["dec"].to_string("dms"))
            self.observer.set(block["observer"])
            self.code.set(block["code"])
            self.title.set(block["title"])
            self.filename.set(block["filename"])
            self.filter.set(block["filter"])
            self.exposure.set(str(block["exposure"]))
            self.nexp.set(str(block["nexp"]))
            self.do_not_interrupt.set(str(block["do_not_interrupt"]))
            self.respositioning.set(
                str(block["respositioning"][0]) + "x" + str(block["respositioning"][1])
            )
            self.shutter_state.set(str(block["shutter_state"]))
            self.readout.set(str(block["readout"]))
            self.binning.set(str(block["binning"][0]) + "x" + str(block["binning"][1]))
            self.frame_position.set(
                str(block["frame_position"][0]) + "," + str(block["frame_position"][1])
            )
            self.frame_size.set(
                str(block["frame_size"][0]) + "x" + str(block["frame_size"][1])
            )
            self.pm_ra_cosdec.set(str(block["pm_ra_cosdec"]))
            self.pm_dec.set(str(block["pm_dec"]))
            self.comment.set(block["comment"])
            self.status.set(block["status"])
            self.message.set(block["message"])


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
