import logging
import platform
import time

import astropy.coordinates as coord
import astropy.io.fits as pyfits

logger = logging.getLogger(__name__)

from ..utils import _force_async
from .wcs import WCS


class PinpointWCS(WCS):
    """PinpointWCS is a wrapper around the PinPoint plate solver. \b

    PinPoint is a commercial plate solver. This class is a wrapper around the
    COM interface to PinPoint. The 64-bit version of PinPoint is required (it is
    a separate download from the 32-bit version, and may be called an add-on).
    """

    def __init__(self):
        logger.debug("PinpointWCS.__init__() called")
        if platform.system() != "Windows":
            raise Exception("PinPoint is only available on Windows.")
        else:
            from win32com.client import Dispatch

            self._solver = Dispatch("PinPoint.Plate")

    def Solve(
        self,
        filepath,
        scale_est,
        ra_key="RA",
        dec_key="DEC",
        ra=None,
        dec=None,
        ra_dec_units=("hour", "deg"),
        solve_timeout=60,
        catalog=3,
        catalog_path="C:\\GSC11",
    ):
        logger.debug(
            f"""PinpointWCS.Solve(
            {filepath}, {scale_est}, {ra_key}, {dec_key}, {ra}, {dec},
            {ra_dec_units}, {solve_timeout}, {catalog}, {catalog_path}
        ) called"""
        )

        self._solver.AttachFITS(filepath)

        if ra is None or dec is None:
            with pyfits.open(filepath) as hdul:
                ra = hdul[0].header[ra_key]
                dec = hdul[0].header[dec_key]
        else:
            ra = self._solver.TargetRightAscension
            dec = self._solver.TargetDeclination

        print(f"RA: {ra}, DEC: {dec}")

        obj = coord.SkyCoord(ra, dec, unit=ra_dec_units, frame="icrs")
        self._solver.RightAscension = obj.ra.hour
        self._solver.Declination = obj.dec.deg

        self._solver.ArcsecPerPixelHoriz = scale_est
        self._solver.ArcsecPerPixelVert = scale_est
        self._solver.Catalog = catalog
        self._solver.CatalogPath = catalog_path

        solved = None
        # solved = self._async_solve()
        solved = self._solver.Solve()
        start_solve = time.time()
        while time.time() - start_solve < solve_timeout and solved is None:
            time.sleep(1)
        else:
            if solved is None:
                raise Exception("Pinpoint solve timed out.")
            else:
                print("Solved!")
                print(f"RA: {self._solver.RightAscension}")
                print(f"DEC: {self._solver.Declination}")
                # return solved

        self._solver.UpdateFITS()
        self._solver.DetachFITS()
        return True

    @_force_async
    def _async_solve(self):
        # Currently doesn't work properly. Needed to synchronously solve.
        self._solver.Solve()
