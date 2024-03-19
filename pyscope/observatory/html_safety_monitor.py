import logging
import urllib.request

from ..utils import _get_number_from_line
from .safety_monitor import SafetyMonitor

logger = logging.getLogger(__name__)


class HTMLSafetyMonitor(SafetyMonitor):
    def __init__(self, url, check_phrase=b"ROOFPOSITION=OPEN"):
        logger.debug(
            f"""HTMLSafetyMonitor.__init__(
            {url}, check_phrase={check_phrase}) called"""
        )

        self._url = url
        self._check_phrase = check_phrase

    @property
    def IsSafe(self):
        logger.debug(f"""HTMLSafetyMonitor.IsSafe property called""")
        safe = None

        stream = urllib.request.urlopen(self._url)
        lines = stream.readlines()

        try:
            units = self._check_phrase.split(b" ")[1]
        except IndexError:
            units = ""

        for line in lines:
            s = _get_number_from_line(
                line,
                self._check_phrase.split(b"=")[0],
                units,
                False,
            )
            if s == self._check_phrase.split(b"=")[1]:
                safe = True
                break
        else:
            safe = False

        return safe

    @property
    def DriverVersion(self):
        logger.debug(f"""HTMLSafetyMonitor.DriverVersion property called""")
        return "1.0"

    @property
    def DriverInfo(self):
        logger.debug(f"""HTMLSafetyMonitor.DriverInfo property called""")
        return "HTML Safety Monitor"

    @property
    def InterfaceVersion(self):
        logger.debug(f"""HTMLSafetyMonitor.InterfaceVersion property called""")
        return "1.0"

    @property
    def Description(self):
        logger.debug(f"""HTMLSafetyMonitor.Description property called""")
        return "HTML Safety Monitor"

    @property
    def SupportedActions(self):
        logger.debug(f"""HTMLSafetyMonitor.SupportedActions property called""")
        return []

    @property
    def Name(self):
        logger.debug(f"""HTMLSafetyMonitor.Name property called""")
        return self._url
