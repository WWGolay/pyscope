import logging

from .ascom_device import ASCOMDevice
from .telescope import Telescope

logger = logging.getLogger(__name__)


class ASCOMTelescope(ASCOMDevice, Telescope):
    def __init__(self, identifier, alpaca=False, device_number=0, protocol="http"):
        super().__init__(
            identifier,
            alpaca=alpaca,
            device_type=self.__class__.__name__.replace("ASCOM", ""),
            device_number=device_number,
            protocol=protocol,
        )

    def AbortSlew(self):
        logger.debug("ASCOMTelescope.AbortSlew() called")
        self._device.AbortSlew

    def AxisRates(self, Axis):
        logger.debug(f"ASCOMTelescope.AxisRates({Axis}) called")
        return self._device.AxisRates(Axis)

    def CanMoveAxis(self, Axis):
        logger.debug(f"ASCOMTelescope.CanMoveAxis({Axis}) called")
        return self._device.CanMoveAxis(Axis)

    def DestinationSideOfPier(self, RightAscension, Declination):
        logger.debug(
            f"ASCOMTelescope.DestinationSideOfPier({RightAscension}, {Declination}) called"
        )
        return self._device.DestinationSideOfPier(RightAscension, Declination)

    def FindHome(self):
        logger.debug("ASCOMTelescope.FindHome() called")
        self._device.FindHome

    def MoveAxis(self, Axis, Rate):
        logger.debug(f"ASCOMTelescope.MoveAxis({Axis}, {Rate}) called")
        self._device.MoveAxis(Axis, Rate)

    def Park(self):
        logger.debug("ASCOMTelescope.Park() called")
        self._device.Park

    def PulseGuide(self, Direction, Duration):
        logger.debug(f"ASCOMTelescope.PulseGuide({Direction}, {Duration}) called")
        self._device.PulseGuide(Direction, Duration)

    def SetPark(self):
        logger.debug("ASCOMTelescope.SetPark() called")
        self._device.SetPark

    def SlewToAltAz(self, Azimuth, Altitude):  # pragma: no cover
        """
        .. deprecated:: 0.1.1
            ASCOM is deprecating this method.
        """
        logger.debug(f"ASCOMTelescope.SlewToAltAz({Azimuth}, {Altitude}) called")
        self._device.SlewToAltAz(Azimuth, Altitude)

    def SlewToAltAzAsync(self, Azimuth, Altitude):
        logger.debug(f"ASCOMTelescope.SlewToAltAzAsync({Azimuth}, {Altitude}) called")
        self._device.SlewToAltAzAsync(Azimuth, Altitude)

    def SlewToCoordinates(self, RightAscension, Declination):  # pragma: no cover
        """
        .. deprecated:: 0.1.1
            ASCOM is deprecating this method.
        """
        logger.debug(
            f"ASCOMTelescope.SlewToCoordinates({RightAscension}, {Declination}) called"
        )
        self._device.SlewToCoordinates(RightAscension, Declination)

    def SlewToCoordinatesAsync(self, RightAscension, Declination):
        logger.debug(
            f"ASCOMTelescope.SlewToCoordinatesAsync({RightAscension}, {Declination}) called"
        )
        self._device.SlewToCoordinatesAsync(RightAscension, Declination)

    def SlewToTarget(self):  # pragma: no cover
        """
        .. deprecated:: 0.1.1
            ASCOM is deprecating this method.
        """
        logger.debug("ASCOMTelescope.SlewToTarget() called")
        self._device.SlewToTarget

    def SlewToTargetAsync(self):
        logger.debug("ASCOMTelescope.SlewToTargetAsync() called")
        self._device.SlewToTargetAsync

    def SyncToAltAz(self, Azimuth, Altitude):
        logger.debug(f"ASCOMTelescope.SyncToAltAz({Azimuth}, {Altitude}) called")
        self._device.SyncToAltAz(Azimuth, Altitude)

    def SyncToCoordinates(self, RightAscension, Declination):
        logger.debug(
            f"ASCOMTelescope.SyncToCoordinates({RightAscension}, {Declination}) called"
        )
        self._device.SyncToCoordinates(RightAscension, Declination)

    def SyncToTarget(self):
        logger.debug("ASCOMTelescope.SyncToTarget() called")
        self._device.SyncToTarget

    def Unpark(self):
        logger.debug("ASCOMTelescope.Unpark() called")
        self._device.Unpark

    @property
    def AlignmentMode(self):
        logger.debug("ASCOMTelescope.AlignmentMode property accessed")
        return self._device.AlignmentMode

    @property
    def Altitude(self):
        logger.debug("ASCOMTelescope.Altitude property accessed")
        return self._device.Altitude

    @property
    def ApertureArea(self):
        logger.debug("ASCOMTelescope.ApertureArea property accessed")
        return self._device.ApertureArea

    @property
    def ApertureDiameter(self):
        logger.debug("ASCOMTelescope.ApertureDiameter property accessed")
        return self._device.ApertureDiameter

    @property
    def AtHome(self):
        logger.debug("ASCOMTelescope.AtHome property accessed")
        return self._device.AtHome

    @property
    def AtPark(self):
        logger.debug("ASCOMTelescope.AtPark property accessed")
        return self._device.AtPark

    @property
    def Azimuth(self):
        logger.debug("ASCOMTelescope.Azimuth property accessed")
        return self._device.Azimuth

    @property
    def CanFindHome(self):
        logger.debug("ASCOMTelescope.CanFindHome property accessed")
        return self._device.CanFindHome

    @property
    def CanPark(self):
        logger.debug("ASCOMTelescope.CanPark property accessed")
        return self._device.CanPark

    def CanPulseGuide(self, Direction):
        logger.debug(f"ASCOMTelescope.CanPulseGuide({Direction}) called")
        return self._device.CanPulseGuide(Direction)

    @property
    def CanSetDeclinationRate(self):
        logger.debug("ASCOMTelescope.CanSetDeclinationRate property accessed")
        return self._device.CanSetDeclinationRate

    @property
    def CanSetGuideRates(self):
        logger.debug("ASCOMTelescope.CanSetGuideRates property accessed")
        return self._device.CanSetGuideRates

    @property
    def CanSetPark(self):
        logger.debug("ASCOMTelescope.CanSetPark property accessed")
        return self._device.CanSetPark

    @property
    def CanSetPierSide(self):
        logger.debug("ASCOMTelescope.CanSetPierSide property accessed")
        return self._device.CanSetPierSide

    @property
    def CanSetRightAscensionRate(self):
        logger.debug("ASCOMTelescope.CanSetRightAscensionRate property accessed")
        return self._device.CanSetRightAscensionRate

    @property
    def CanSetTracking(self):
        logger.debug("ASCOMTelescope.CanSetTracking property accessed")
        return self._device.CanSetTracking

    @property
    def CanSlew(self):  # pragma: no cover
        """
        .. deprecated:: 0.1.1
            ASCOM is deprecating this property.
        """
        logger.debug("ASCOMTelescope.CanSlew property accessed")
        return self._device.CanSlew

    @property
    def CanSlewAltAz(self):  # pragma: no cover
        """
        .. deprecated:: 0.1.1
            ASCOM is deprecating this property.
        """
        logger.debug("ASCOMTelescope.CanSlewAltAz property accessed")
        return self._device.CanSlewAltAz

    @property
    def CanSlewAltAzAsync(self):
        logger.debug("ASCOMTelescope.CanSlewAltAzAsync property accessed")
        return self._device.CanSlewAltAzAsync

    @property
    def CanSlewAsync(self):
        logger.debug("ASCOMTelescope.CanSlewAsync property accessed")
        return self._device.CanSlewAsync

    @property
    def CanSync(self):
        logger.debug("ASCOMTelescope.CanSync property accessed")
        return self._device.CanSync

    @property
    def CanSyncAltAz(self):
        logger.debug("ASCOMTelescope.CanSyncAltAz property accessed")
        return self._device.CanSyncAltAz

    @property
    def CanUnpark(self):
        logger.debug("ASCOMTelescope.CanUnpark property accessed")
        return self._device.CanUnpark

    @property
    def Declination(self):
        logger.debug("ASCOMTelescope.Declination property accessed")
        return self._device.Declination

    @property
    def DeclinationRate(self):
        logger.debug("ASCOMTelescope.DeclinationRate property accessed")
        return self._device.DeclinationRate

    @DeclinationRate.setter
    def DeclinationRate(self, value):
        logger.debug(f"ASCOMTelescope.DeclinationRate set to {value}")
        self._device.DeclinationRate = value

    @property
    def DoesRefraction(self):
        logger.debug("ASCOMTelescope.DoesRefraction property accessed")
        return self._device.DoesRefraction

    @DoesRefraction.setter
    def DoesRefraction(self, value):
        logger.debug(f"ASCOMTelescope.DoesRefraction set to {value}")
        self._device.DoesRefraction = value

    @property
    def EquatorialSystem(self):
        logger.debug("ASCOMTelescope.EquatorialSystem property accessed")
        return self._device.EquatorialSystem

    @property
    def FocalLength(self):
        logger.debug("ASCOMTelescope.FocalLength property accessed")
        return self._device.FocalLength

    @property
    def GuideRateDeclination(self):
        logger.debug("ASCOMTelescope.GuideRateDeclination property accessed")
        return self._device.GuideRateDeclination

    @GuideRateDeclination.setter
    def GuideRateDeclination(self, value):
        logger.debug(f"ASCOMTelescope.GuideRateDeclination set to {value}")
        self._device.GuideRateDeclination = value

    @property
    def GuideRateRightAscension(self):
        logger.debug("ASCOMTelescope.GuideRateRightAscension property accessed")
        return self._device.GuideRateRightAscension

    @GuideRateRightAscension.setter
    def GuideRateRightAscension(self, value):
        logger.debug(f"ASCOMTelescope.GuideRateRightAscension set to {value}")
        self._device.GuideRateRightAscension = value

    @property
    def IsPulseGuiding(self):
        logger.debug("ASCOMTelescope.IsPulseGuiding property accessed")
        return self._device.IsPulseGuiding

    @property
    def RightAscension(self):
        logger.debug("ASCOMTelescope.RightAscension property accessed")
        return self._device.RightAscension

    @property
    def RightAscensionRate(self):
        logger.debug("ASCOMTelescope.RightAscensionRate property accessed")
        return self._device.RightAscensionRate

    @RightAscensionRate.setter
    def RightAscensionRate(self, value):
        logger.debug(f"ASCOMTelescope.RightAscensionRate set to {value}")
        self._device.RightAscensionRate = value

    @property
    def SideOfPier(self):
        logger.debug("ASCOMTelescope.SideOfPier property accessed")
        return self._device.SideOfPier

    @SideOfPier.setter
    def SideOfPier(self, value):
        logger.debug(f"ASCOMTelescope.SideOfPier set to {value}")
        self._device.SideOfPier = value

    @property
    def SiderealTime(self):
        logger.debug("ASCOMTelescope.SiderealTime property accessed")
        return self._device.SiderealTime

    @property
    def SiteElevation(self):
        logger.debug("ASCOMTelescope.SiteElevation property accessed")
        return self._device.SiteElevation

    @SiteElevation.setter
    def SiteElevation(self, value):
        logger.debug(f"ASCOMTelescope.SiteElevation set to {value}")
        self._device.SiteElevation = value

    @property
    def SiteLatitude(self):
        logger.debug("ASCOMTelescope.SiteLatitude property accessed")
        return self._device.SiteLatitude

    @SiteLatitude.setter
    def SiteLatitude(self, value):
        logger.debug(f"ASCOMTelescope.SiteLatitude set to {value}")
        self._device.SiteLatitude = value

    @property
    def SiteLongitude(self):
        logger.debug("ASCOMTelescope.SiteLongitude property accessed")
        return self._device.SiteLongitude

    @SiteLongitude.setter
    def SiteLongitude(self, value):
        logger.debug(f"ASCOMTelescope.SiteLongitude set to {value}")
        self._device.SiteLongitude = value

    @property
    def Slewing(self):
        logger.debug("ASCOMTelescope.Slewing property accessed")
        return self._device.Slewing

    @property
    def SlewSettleTime(self):
        logger.debug("ASCOMTelescope.SlewSettleTime property accessed")
        return self._device.SlewSettleTime

    @SlewSettleTime.setter
    def SlewSettleTime(self, value):
        logger.debug(f"ASCOMTelescope.SlewSettleTime set to {value}")
        self._device.SlewSettleTime = value

    @property
    def TargetDeclination(self):
        logger.debug("ASCOMTelescope.TargetDeclination property accessed")
        return self._device.TargetDeclination

    @TargetDeclination.setter
    def TargetDeclination(self, value):
        logger.debug(f"ASCOMTelescope.TargetDeclination set to {value}")
        self._device.TargetDeclination = value

    @property
    def TargetRightAscension(self):
        logger.debug("ASCOMTelescope.TargetRightAscension property accessed")
        return self._device.TargetRightAscension

    @TargetRightAscension.setter
    def TargetRightAscension(self, value):
        logger.debug(f"ASCOMTelescope.TargetRightAscension set to {value}")
        self._device.TargetRightAscension = value

    @property
    def Tracking(self):
        logger.debug("ASCOMTelescope.Tracking property accessed")
        return self._device.Tracking

    @Tracking.setter
    def Tracking(self, value):
        logger.debug(f"ASCOMTelescope.Tracking set to {value}")
        self._device.Tracking = value

    @property
    def TrackingRate(self):
        logger.debug("ASCOMTelescope.TrackingRate property accessed")
        return self._device.TrackingRate

    @TrackingRate.setter
    def TrackingRate(self, value):
        logger.debug(f"ASCOMTelescope.TrackingRate set to {value}")
        self._device.TrackingRate = value

    @property
    def TrackingRates(self):
        logger.debug("ASCOMTelescope.TrackingRates property accessed")
        return self._device.TrackingRates

    @property
    def UTCDate(self):
        logger.debug("ASCOMTelescope.UTCDate property accessed")
        return self._device.UTCDate

    @UTCDate.setter
    def UTCDate(self, value):
        logger.debug(f"ASCOMTelescope.UTCDate set to {value}")
        self._device.UTCDate = value
