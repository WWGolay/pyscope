import logging
import platform

from .device import Device
from .observatory_exception import ObservatoryException

logger = logging.getLogger(__name__)


class ASCOMDevice(Device):
    def __init__(self, identifier, alpaca=False, device_type="Device", **kwargs):
        logger.debug(f"ASCOMDevice.__init__({identifier}, alpaca={alpaca}, {kwargs})")
        self._identifier = identifier
        self._device = None

        if alpaca:
            import alpaca

            self._device = getattr(
                getattr(
                    __import__("alpaca." + device_type.lower()), device_type.lower()
                ),
                device_type,
            )(self._identifier, **kwargs)
        elif platform.system() == "Windows":
            from win32com.client import Dispatch

            self._device = Dispatch(self._identifier)
        else:
            raise ObservatoryException("If you are not on Windows, you must use Alpaca")

    def Action(self, ActionName, *ActionParameters):  # pragma: no cover
        logger.debug(f"ASCOMDevice.Action({ActionName}, {ActionParameters})")
        return self._device.Action(ActionName, *ActionParameters)

    def CommandBlind(self, Command, Raw):  # pragma: no cover
        """
        .. deprecated:: 0.1.1
            ASCOM is deprecating this method.
        """

        logger.debug(f"ASCOMDevice.CommandBlind({Command}, {Raw})")
        self._device.CommandBlind(Command, Raw)

    def CommandBool(self, Command, Raw):  # pragma: no cover
        """
        .. deprecated:: 0.1.1
            ASCOM is deprecating this method.
        """

        logger.debug(f"ASCOMDevice.CommandBool({Command}, {Raw})")
        return self._device.CommandBool(Command, Raw)

    def CommandString(self, Command, Raw):  # pragma: no cover
        """
        .. deprecated:: 0.1.1
            ASCOM is deprecating this method.
        """

        logger.debug(f"ASCOMDevice.CommandString({Command}, {Raw})")
        return self._device.CommandString(Command, Raw)

    @property
    def Connected(self):
        logger.debug(f"ASCOMDevice.Connected property")
        return self._device.Connected

    @Connected.setter
    def Connected(self, value):
        logger.debug(f"ASCOMDevice.Connected property set to {value}")
        self._device.Connected = value

    @property
    def Description(self):
        logger.debug(f"ASCOMDevice.Description property")
        return self._device.Description

    @property
    def DriverInfo(self):
        logger.debug(f"ASCOMDevice.DriverInfo property")
        return self._device.DriverInfo

    @property
    def DriverVersion(self):
        logger.debug(f"ASCOMDevice.DriverVersion property")
        return self._device.DriverVersion

    @property
    def InterfaceVersion(self):
        logger.debug(f"ASCOMDevice.InterfaceVersion property")
        return self._device.InterfaceVersion

    @property
    def Name(self):
        logger.debug(f"ASCOMDevice.Name property")
        return self._device.Name

    @property
    def SupportedActions(self):
        logger.debug(f"ASCOMDevice.SupportedActions property")
        return self._device.SupportedActions
