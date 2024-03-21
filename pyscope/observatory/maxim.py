import logging
import platform
import time

from astropy.time import Time

from .autofocus import Autofocus
from .camera import Camera
from .device import Device
from .filter_wheel import FilterWheel
from .wcs import WCS

logger = logging.getLogger(__name__)


class Maxim(Device):
    def __init__(self):
        logger.debug("Maxim.Maxim __init__ called")
        if platform.system() != "Windows":
            raise Exception("This class is only available on Windows.")
        else:
            from win32com.client import Dispatch

            self._app = Dispatch("MaxIm.Application")
            self._app.LockApp = True

            self._camera = _MaximCamera()
            self._camera.Connected = True

            self._filter_wheel = None
            self._autofocus = None
            if self._camera.HasFilterWheel:
                self._filter_wheel = _MaximFilterWheel(self.camera)
            if self._app.FocuserConnected:
                self._autofocus = _MaximAutofocus(self.app)

            self._wcs = _MaximPinpointWCS(self.camera)

    @property
    def Connected(self):
        logger.debug("Maxim.Connected called")
        return self.camera.Connected

    @Connected.setter
    def Connected(self, value):
        logger.debug(f"Connected setter called with value={value}")
        self.camera.Connected = value

    @property
    def Name(self):
        logger.debug("Maxim.Name called")
        return "MaxIm DL"

    @property
    def app(self):
        logger.debug("Maxim.app called")
        return self._app

    @property
    def autofocus(self):
        logger.debug("Maxim.autofocus called")
        return self._autofocus

    @property
    def camera(self):
        logger.debug("Maxim.camera called")
        return self._camera


class _MaximAutofocus(Autofocus):
    def __init__(self, maxim):
        logger.debug("_MaximAutofocus.MaximAutofocus __init__ called")
        self.maxim = maxim

    def Run(self, exposure=10):
        logger.debug(f"Run called with exposure={exposure}")
        self.maxim.Autofocus(exposure)

        while self.maxim.AutofocusStatus == -1:
            time.sleep(1)

        if self.maxim.AutofocusStatus == 1:
            return True
        else:
            return False

    def Abort(self):
        logger.debug("_MaximAutofocus.Abort called")
        raise NotImplementedError


class _MaximCamera(Camera):
    def __init__(self):
        logger.debug("_MaximCamera_MaximCamera __init__ called")
        self._com_object = Dispatch("MaxIm.CCDCamera")
        self._com_object.DisableAutoShutdown = True

        self._last_exposure_duration = None
        self._last_exposure_start_time = None

    @property
    def Connected(self):
        logger.debug("_MaximCameraConnected called")
        return self._com_object is not None and self._com_object.LinkEnabled

    @Connected.setter
    def Connected(self, value):
        logger.debug(f"Connected setter called with value={value}")
        self._com_object.LinkEnabled = value

    @property
    def Name(self):
        logger.debug("_MaximCameraName called")
        return self._com_object.CameraName

    def AbortExposure(self):
        logger.debug("_MaximCameraAbortExposure called")
        self._com_object.AbortExposure()

    def PulseGuide(self, Direction, Duration):
        logger.debug(
            f"PulseGuide called with Direction={Direction}, Duration={Duration}"
        )
        raise NotImplementedError

    def SaveImageAsFits(self, filename):
        logger.debug(f"SaveImageAsFits called with filename={filename}")
        self._com_object.SaveImage(filename)

    def StartExposure(self, Duration, Light):
        logger.debug(f"StartExposure called with Duration={Duration}, Light={Light}")
        self._last_exposure_duration = Duration
        self._last_exposure_start_time = str(Time.now())
        self._com_object.Expose(Duration, Light)

    def StopExposure(self):
        logger.debug("_MaximCameraStopExposure called")
        self._com_object.AbortExposure()

    def VerifyLatestExposure(self):
        """Verify that the last exposure is complete. \b

        Make sure that the image that was returned by Maxim was in fact generated
        by the most recent call to Expose(). I have seen cases where a camera
        dropout occurs (e.g. if a USB cable gets unplugged and plugged back in)
        where Maxim will claim that an exposure is complete but just return the
        same (old) image over and over again.

        Return without error if the image from Maxim appears to be newer than
        the supplied UTC datetime object based on the DATE-OBS header.
        Raise an exception if the image is older or if there is an error
        accessing the image.
        """
        logger.debug("_MaximCameraVerifyLatestExposure called")

        try:
            # Change to document
            image = self._com_object.Document
        except Exception as e:
            raise Exception(f"Unable to access MaxIm camera image: {e}")

        if image is None:
            raise Exception("No current image available from MaxIm")

        # Get the DATE-OBS header
        image_timestamp = image.GetFITSKey("DATE-OBS")
        image_datetime = Time(image_timestamp, format="fits")

        logger.debug(f"Image timestamp: {image_timestamp}")
        logger.debug(f"Image timestamp UTC: {image_datetime}")
        logger.debug(f"Exposure start time: {self._last_exposure_start_time}")

        if image_datetime < self._last_exposure_start_time:
            raise Exception(
                "Image is too old; possibly the result of an earlier exposure. There may be a connection problem with the camera"
            )

    @property
    def BayerOffsetX(self):
        logger.debug("_MaximCameraBayerOffsetX called")
        raise NotImplementedError

    @property
    def BayerOffsetY(self):
        logger.debug("_MaximCameraBayerOffsetY called")
        raise NotImplementedError

    @property
    def BinX(self):
        logger.debug("_MaximCameraBinX called")
        return self._com_object.BinX

    @BinX.setter
    def BinX(self, value):
        logger.debug(f"BinX setter called with value={value}")
        self._com_object.BinX = value

    @property
    def BinY(self):
        logger.debug("_MaximCameraBinY called")
        return self._com_object.BinY

    @BinY.setter
    def BinY(self, value):
        logger.debug(f"BinY setter called with value={value}")
        self._com_object.BinY = value

    @property
    def CameraState(self):
        logger.debug("_MaximCameraCameraState called")
        return self._com_object.CameraStatus

    @property
    def CameraXSize(self):
        logger.debug("_MaximCameraCameraXSize called")
        return self._com_object.CameraXSize

    @property
    def CameraYSize(self):
        logger.debug("_MaximCameraCameraYSize called")
        return self._com_object.CameraYSize

    @property
    def CanAbortExposure(self):
        logger.debug("_MaximCameraCanAbortExposure called")
        return True

    @property
    def CanAsymmetricBin(self):
        logger.debug("_MaximCameraCanAsymmetricBin called")
        return self._com_object.XYBinning

    @property
    def CanFastReadout(self):
        logger.debug("_MaximCameraCanFastReadout called")
        # Not implemented, return False
        return False

    @property
    def CanGetCoolerPower(self):
        logger.debug("_MaximCameraCanGetCoolerPower called")
        return True

    @property
    def CanPulseGuide(self):
        logger.debug("_MaximCameraCanPulseGuide called")
        return False

    @property
    def CanSetCCDTemperature(self):
        logger.debug("_MaximCameraCanSetCCDTemperature called")
        return self._com_object.CanSetTemperature

    @property
    def CanStopExposure(self):
        logger.debug("_MaximCameraCanStopExposure called")
        return True

    @property
    def CCDTemperature(self):
        logger.debug("_MaximCameraCCDTemperature called")
        return self._com_object.Temperature

    @property
    def CoolerOn(self):
        logger.debug("_MaximCameraCoolerOn called")
        return self._com_object.CoolerOn

    @CoolerOn.setter
    def CoolerOn(self, value):
        logger.debug(f"CoolerOn setter called with value={value}")
        self._com_object.CoolerOn = value

    @property
    def CoolerPower(self):
        logger.debug("_MaximCameraCoolerPower called")
        return self._com_object.CoolerPower

    @property
    def Description(self):
        logger.debug("_MaximCameraDescription called")
        return "MaxIm camera for pyscope"

    @property
    def DriverVersion(self):
        logger.debug("_MaximCameraDriverVersion called")
        return "Custom pyscope MaxIm driver"

    @property
    def DriverInfo(self):
        logger.debug("_MaximCameraDriverInfo called")
        return "Custom pyscope MaxIm driver"

    @property
    def ElectronsPerADU(self):
        logger.debug("_MaximCameraElectronsPerADU called")
        raise NotImplementedError

    @property
    def ExposureMax(self):
        logger.debug("_MaximCameraExposureMax called")
        raise NotImplementedError

    @property
    def ExposureMin(self):
        logger.debug("_MaximCameraExposureMin called")
        raise NotImplementedError

    @property
    def ExposureResolution(self):
        logger.debug("_MaximCameraExposureResolution called")
        raise NotImplementedError

    @property
    def FastReadout(self):
        logger.debug("_MaximCameraFastReadout called")
        raise DeprecationWarning

    @FastReadout.setter
    def FastReadout(self, value):
        logger.debug(f"FastReadout setter called with value={value}")
        raise DeprecationWarning

    @property
    def FullWellCapacity(self):
        logger.debug("_MaximCameraFullWellCapacity called")
        raise NotImplementedError

    @property
    def Gain(self):
        logger.debug("_MaximCameraGain called")
        return self._com_object.Speed

    @Gain.setter
    def Gain(self, value):
        logger.debug(f"Gain setter called with value={value}")
        self._com_object.Speed = value

    @property
    def GainMax(self):
        logger.debug("_MaximCameraGainMax called")
        raise NotImplementedError

    @property
    def GainMin(self):
        logger.debug("_MaximCameraGainMin called")
        raise NotImplementedError

    @property
    def Gains(self):
        logger.debug("_MaximCameraGains called")
        return self._com_object.Speeds.replace("(", "").replace(")", "")

    @property
    def HasFilterWheel(self):
        logger.debug("_MaximCameraHasFilterWheel called")
        return self._com_object.HasFilterWheel

    @property
    def HasShutter(self):
        logger.debug("_MaximCameraHasShutter called")
        return self._com_object.HasShutter

    @property
    def HeatSinkTemperature(self):
        logger.debug("_MaximCameraHeatSinkTemperature called")
        return self._com_object.AmbientTemperature

    @property
    def ImageArray(self):
        logger.debug("_MaximCameraImageArray called")
        return self._com_object.ImageArray

    @property
    def ImageArrayVariant(self):
        logger.debug("_MaximCameraImageArrayVariant called")
        return float(self._com_object.ImageArray)

    @property
    def ImageReady(self):
        logger.debug("_MaximCameraImageReady called")
        return self._com_object.ImageReady

    @property
    def InterfaceVersion(self):
        logger.debug("_MaximCameraInterfaceVersion called")
        raise NotImplementedError

    @property
    def IsPulseGuiding(self):
        logger.debug("_MaximCameraIsPulseGuiding called")
        raise NotImplementedError

    @property
    def LastExposureDuration(self):
        logger.debug("_MaximCameraLastExposureDuration called")
        return self._last_exposure_duration

    @property
    def LastExposureStartTime(self):
        logger.debug("_MaximCameraLastExposureStartTime called")
        return self._last_exposure_start_time

    @property
    def MaxADU(self):
        logger.debug("_MaximCameraMaxADU called")
        raise NotImplementedError

    @property
    def MaxBinX(self):
        logger.debug("_MaximCameraMaxBinX called")
        return self._com_object.MaxBinX

    @property
    def MaxBinY(self):
        logger.debug("_MaximCameraMaxBinY called")
        return self._com_object.MaxBinY

    @property
    def NumX(self):
        logger.debug("_MaximCameraNumX called")
        return self._com_object.NumX

    @NumX.setter
    def NumX(self, value):
        logger.debug(f"NumX setter called with value={value}")
        self._com_object.NumX = value

    @property
    def NumY(self):
        logger.debug("_MaximCameraNumY called")
        return self._com_object.NumY

    @NumY.setter
    def NumY(self, value):
        logger.debug(f"NumY setter called with value={value}")
        self._com_object.NumY = value

    @property
    def Offset(self):
        logger.debug("_MaximCameraOffset called")
        raise NotImplementedError

    @Offset.setter
    def Offset(self, value):
        logger.debug(f"Offset setter called with value={value}")
        raise NotImplementedError

    @property
    def OffsetMax(self):
        logger.debug("_MaximCameraOffsetMax called")
        raise NotImplementedError

    @property
    def OffsetMin(self):
        logger.debug("_MaximCameraOffsetMin called")
        raise NotImplementedError

    @property
    def Offsets(self):
        logger.debug("_MaximCameraOffsets called")
        raise NotImplementedError

    @property
    def PercentCompleted(self):
        logger.debug("_MaximCameraPercentCompleted called")
        raise NotImplementedError

    @property
    def PixelSizeX(self):
        logger.debug("_MaximCameraPixelSizeX called")
        return self._com_object.PixelSizeX

    @property
    def PixelSizeY(self):
        logger.debug("_MaximCameraPixelSizeY called")
        return self._com_object.PixelSizeY

    @property
    def ReadoutMode(self):
        logger.debug("_MaximCameraReadoutMode called")
        return self._com_object.ReadoutMode

    @ReadoutMode.setter
    def ReadoutMode(self, value):
        logger.debug(f"ReadoutMode setter called with value={value}")
        self._com_object.ReadoutMode = value

    @property
    def ReadoutModes(self):
        logger.debug("_MaximCameraReadoutModes called")
        return self._com_object.ReadoutModes

    @property
    def SensorName(self):
        logger.debug("_MaximCameraSensorName called")
        raise NotImplementedError

    @property
    def SensorType(self):
        logger.debug("_MaximCameraSensorType called")
        raise NotImplementedError

    @property
    def SetCCDTemperature(self):
        logger.debug("_MaximCameraSetCCDTemperature called")
        return self._com_object.TemperatureSetpoint

    @SetCCDTemperature.setter
    def SetCCDTemperature(self, value):
        logger.debug(f"SetCCDTemperature setter called with value={value}")
        self._com_object.TemperatureSetpoint = value

    @property
    def StartX(self):
        logger.debug("_MaximCameraStartX called")
        return self._com_object.StartX

    @StartX.setter
    def StartX(self, value):
        logger.debug(f"StartX setter called with value={value}")
        self._com_object.StartX = value

    @property
    def StartY(self):
        logger.debug("_MaximCameraStartY called")
        return self._com_object.StartY

    @StartY.setter
    def StartY(self, value):
        logger.debug(f"StartY setter called with value={value}")
        self._com_object.StartY = value

    @property
    def SubExposureDuration(self):
        logger.debug("_MaximCameraSubExposureDuration called")
        raise NotImplementedError

    @SubExposureDuration.setter
    def SubExposureDuration(self, value):
        logger.debug(f"SubExposureDuration setter called with value={value}")
        raise NotImplementedError


class _MaximFilterWheel(FilterWheel):
    def __init__(self, maxim_camera):
        logger.debug("_MaximFilterWheelMaximFilterWheel __init__ called")
        self.maxim_camera = maxim_camera

    @property
    def Connected(self):
        logger.debug("_MaximFilterWheelConnected called")
        return self.maxim_camera.Connected and self.maxim_camera.HasFilterWheel

    @Connected.setter
    def Connected(self, value):
        logger.debug(f"Connected setter called with value={value}")
        self.maxim_camera.Connected = value

    @property
    def Name(self):
        logger.debug("_MaximFilterWheelName called")
        return self.maxim_camera._com_object.FilterWheelName

    @property
    def FocusOffsets(self):
        logger.debug("_MaximFilterWheelFocusOffsets called")
        raise NotImplementedError

    @property
    def Names(self):
        logger.debug("_MaximFilterWheelNames called")
        return self.maxim_camera._com_object.FilterNames

    @property
    def Position(self):
        logger.debug("_MaximFilterWheelPosition called")
        return self.maxim_camera._com_object.Filter

    @Position.setter
    def Position(self, value):
        logger.debug(f"Position setter called with value={value}")
        self.maxim_camera._com_object.Filter = value


class _MaximPinpointWCS(WCS):
    def __init__(self, maxim_camera):
        logger.debug("_MaximPinpointWCS __init__ called")
        self.maxim_camera = maxim_camera

    def Solve(self, ra=None, dec=None, scale_est=None, *args, **kwargs):
        logger.debug(
            f"_MaximPinpointWCS.Solve called with ra={ra}, dec={dec}, and scale_est={scale_est}"
        )
        self.maxim_camera.document.PinpointSolve(ra, dec, scale_est, scale_est)

        while self.maxim_camera.document.PinpointStatus == 3:
            time.sleep(1)

        if self.maxim_camera.document.PinpointStatus == 2:
            return True
        else:
            return False
