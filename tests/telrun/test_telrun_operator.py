import json
import logging
import pprint
from pathlib import Path

import pytest
from astroplan import FixedTarget, ObservingBlock
from astropy import coordinates as coord
from astropy import time
from astropy import units as u

from pyscope import logger
from pyscope.observatory import Observatory
from pyscope.telrun import TelrunOperator, schedtel


def test_telrun_operator(tmp_path):

    # set up logging for debugging
    logger.setLevel("INFO")
    logger.addHandler(logging.StreamHandler())
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s"
    )
    logger.handlers[0].setFormatter(formatter)

    # define a new longitude
    t_now = time.Time.now()
    t_mid = time.Time(int(t_now.jd + 1), format="jd")
    new_longitude = t_mid - t_now - 12 * u.hour
    new_longitude = new_longitude.to(u.hour).value * 15 * u.deg

    # create a new observatory
    obs = Observatory(
        config_path="./tests/bin/simulator_observatory.cfg", longitude=new_longitude
    )

    # get the local sidereal time
    lst = obs.observatory_time.sidereal_time("apparent", longitude=obs.longitude)
    lst -= 35 / (60 * 60) * u.hourangle  # 35 seconds before the next LST

    # create a telrun operator
    telrun = TelrunOperator(telhome=tmp_path, observatory=obs, gui=False)

    # create objects to observe
    catalog = [
        ObservingBlock(
            target=coord.SkyCoord(ra=lst.to(u.hourangle), dec=obs.latitude),
            duration=1 * u.minute,
            priority=1,
            name="Test Obs",
            configuration={
                "observer": ["test_observer"],
                "code": "aaa",
                "title": "test_obs",
                "filename": "",
                "type": "light",
                "backend": 0,
                "filter": "R",
                "exposure": 2.0,
                "nexp": 2,
                "repositioning": (0, 0),
                "shutter_state": True,
                "readout": 0,
                "binning": (1, 1),
                "frame_position": (0, 0),
                "frame_size": (0, 0),
                "pm_ra_cosdec": 0 * u.arcsec / u.hour,
                "pm_dec": 0 * u.arcsec / u.hour,
                "comment": "",
                "sch": "",
                "status": "U",
                "message": "Unscheduled",
                "sched_time": None,
            },
        ),
    ]

    # schedule the objects
    schedule = schedtel(
        catalog=catalog,
        observatory=obs,
        filename=tmp_path / "schedules/execute/test_schedtel.ecsv",
    )

    # have telrun execute the schedule
    telrun.execute_schedule(tmp_path / "schedules/execute/test_schedtel.ecsv")


if __name__ == "__main__":
    test_telrun_operator(Path("./my_test/").resolve())
