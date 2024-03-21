import logging
import platform
import time

from .autofocus import Autofocus

logger = logging.getLogger(__name__)


class PWIAutofocus(Autofocus):
    def __init__(self):
        logger.debug("PWIAutofocus.__init__ called")
        if platform.system() != "Windows":
            raise Exception("This class is only available on Windows.")
        else:
            from win32com.client import Dispatch

            self._com_object = Dispatch("PlaneWave.AutoFocus")

            self._com_object.StartPwiIfNeeded
            self._forward_autofocus_messages()
            self._com_object.ConnectFocuser
            self._forward_autofocus_messages()

            # Wait up to 3 seconds for a confirmed connection
            t = time.time() + 3
            while not self._com_object.IsFocuserConnected:
                if time.time() > t:
                    raise Exception("Unable to connect to focuser.")
                time.sleep(0.1)
                self._forward_autofocus_messages()

            self._com_object.PreventFilterChange = True

    def Run(self, exposure=10, timeout=120):
        logger.debug(
            f"PWIAutofocus.Run called with args: exposure={exposure}, timeout={timeout}"
        )
        self._com_object.ExposureLengthSeconds = exposure

        if not self._com_object.IsFocuserConnected:
            raise Exception(
                "Unable to run PlaneWave AutoFocus: focuser is not connected"
            )

        self._com_object.StartAutoFocus

        t = time.time() + timeout
        while self._com_object.IsAutoFocusRunning:
            self._forward_autofocus_messages()
            time.sleep(0.2)

            if time.time() > t:
                raise Exception(
                    "Autofocus took longer than %g seconds to complete" % timeout
                )

        if self._com_object.Success:
            return self._com_object.BestPosition
        else:
            return None

    def Abort(self):
        logger.debug("PWIAutofocus.Abort called")
        self._com_object.StopAutofocus

    def _forward_autofocus_messages(self):
        while True:
            log_line = self._com_object.NextLogMessage
            if log_line is None:
                return
            logger.info(log_line)
