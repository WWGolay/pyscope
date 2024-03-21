import logging

import numpy as np
from astropy.time import Time

from .ascom_device import ASCOMDevice
from .camera import Camera

logger = logging.getLogger(__name__)


class ASCOMCamera(ASCOMDevice, Camera):
    def __init__(self, identifier, alpaca=False, device_number=0, protocol="http"):
        super().__init__(
            identifier,
            alpaca=alpaca,
            device_type=self.__class__.__name__.replace("ASCOM", ""),
            device_number=device_number,
            protocol=protocol,
        )
        self._last_exposure_duration = None
        self._last_exposure_start_time = None
        self._image_data_type = None
        self._DoTranspose = True
        self._camera_time = True

    def AbortExposure(self):
        logger.debug(f"ASCOMCamera.AbortExposure() called")
        self._device.AbortExposure()

    def SetImageDataType(self):
        """Determine the data type of the image array based on the MaxADU property.

        This method is called automatically when the ImageArray property is called
        if it has not already been set (initializes to `None`).
        It will choose from the following data types based on the MaxADU property:

        - numpy.uint8 : (if MaxADU <= 255)
        - numpy.uint16 : (default if MaxADU is not defined, or if MaxADU <= 65535)
        - numpy.uint32 : (if MaxADU > 65535)

        See Also
        --------
        numpy.uint8
        numpy.uint16
        numpy.uint32
        MaxADU : ASCOM Camera interface property `ASCOM Documentation <https://ascom-standards.org/Help/Developer/html/P_ASCOM_DriverAccess_Camera_MaxADU.htm>`_
        """
        logger.debug(f"ASCOMCamera.SetImageDataType() called")
        try:
            max_adu = self.MaxADU
            if max_adu <= 255:
                self._image_data_type = np.uint8
            elif max_adu <= 65535:
                self._image_data_type = np.uint16
            else:
                self._image_data_type = np.uint32
        except:
            self._image_data_type = np.uint16

    def PulseGuide(self, Direction, Duration):
        logger.debug(f"ASCOMCamera.PulseGuide({Direction}, {Duration}) called")
        self._device.PulseGuide(Direction, Duration)

    def StartExposure(self, Duration, Light):
        logger.debug(f"ASCOMCamera.StartExposure({Duration}, {Light}) called")
        self._last_exposure_duration = Duration
        self._last_exposure_start_time = str(Time.now())
        self._device.StartExposure(Duration, Light)

    def StopExposure(self):
        logger.debug(f"ASCOMCamera.StopExposure() called")
        self._device.StopExposure()

    @property
    def BayerOffsetX(self):  # pragma: no cover
        """
        .. warning::
            This property is not implemented in the ASCOM Alpaca protocol.
        """
        logger.debug(f"ASCOMCamera.BayerOffsetX property called")
        return self._device.BayerOffsetX

    @property
    def BayerOffsetY(self):  # pragma: no cover
        """
        .. warning::
            This property is not implemented in the ASCOM Alpaca protocol.
        """
        logger.debug(f"ASCOMCamera.BayerOffsetY property called")
        return self._device.BayerOffsetY

    @property
    def BinX(self):
        logger.debug(f"ASCOMCamera.BinX property called")
        return self._device.BinX

    @BinX.setter
    def BinX(self, value):
        logger.debug(f"ASCOMCamera.BinX property set to {value}")
        self._device.BinX = value

    @property
    def BinY(self):
        logger.debug(f"ASCOMCamera.BinY property called")
        return self._device.BinY

    @BinY.setter
    def BinY(self, value):
        logger.debug(f"ASCOMCamera.BinY property set to {value}")
        self._device.BinY = value

    @property
    def CameraState(self):
        logger.debug(f"ASCOMCamera.CameraState property called")
        return self._device.CameraState

    @property
    def CameraXSize(self):
        logger.debug(f"ASCOMCamera.CameraXSize property called")
        return self._device.CameraXSize

    @property
    def CameraYSize(self):
        logger.debug(f"ASCOMCamera.CameraYSize property called")
        return self._device.CameraYSize

    @property
    def CameraTime(self):
        logger.debug(f"ASCOMCamera.CameraTime property called")
        return self._camera_time

    @property
    def CanAbortExposure(self):
        logger.debug(f"ASCOMCamera.CanAbortExposure property called")
        return self._device.CanAbortExposure

    @property
    def CanAsymmetricBin(self):
        logger.debug(f"ASCOMCamera.CanAsymmetricBin property called")
        return self._device.CanAsymmetricBin

    @property
    def CanFastReadout(self):
        logger.debug(f"ASCOMCamera.CanFastReadout property called")
        return self._device.CanFastReadout

    @property
    def CanGetCoolerPower(self):
        logger.debug(f"ASCOMCamera.CanGetCoolerPower property called")
        return self._device.CanGetCoolerPower

    @property
    def CanPulseGuide(self):
        logger.debug(f"ASCOMCamera.CanPulseGuide property called")
        return self._device.CanPulseGuide

    @property
    def CanSetCCDTemperature(self):
        logger.debug(f"ASCOMCamera.CanSetCCDTemperature property called")
        return self._device.CanSetCCDTemperature

    @property
    def CanStopExposure(self):
        logger.debug(f"ASCOMCamera.CanStopExposure property called")
        return self._device.CanStopExposure

    @property
    def CCDTemperature(self):
        logger.debug(f"ASCOMCamera.CCDTemperature property called")
        return self._device.CCDTemperature

    @property
    def CoolerOn(self):
        logger.debug(f"ASCOMCamera.CoolerOn property called")
        return self._device.CoolerOn

    @CoolerOn.setter
    def CoolerOn(self, value):
        logger.debug(f"ASCOMCamera.CoolerOn property set to {value}")
        self._device.CoolerOn = value

    @property
    def CoolerPower(self):
        logger.debug(f"ASCOMCamera.CoolerPower property called")
        return self._device.CoolerPower

    @property
    def ElectronsPerADU(self):
        logger.debug(f"ASCOMCamera.ElectronsPerADU() property called")
        return self._device.ElectronsPerADU

    @property
    def ExposureMax(self):
        logger.debug(f"ASCOMCamera.ExposureMax property called")
        return self._device.ExposureMax

    @property
    def ExposureMin(self):
        logger.debug(f"ASCOMCamera.ExposureMin property called")
        return self._device.ExposureMin

    @property
    def ExposureResolution(self):
        logger.debug(f"ASCOMCamera.ExposureResolution property called")
        return self._device.ExposureResolution

    @property
    def FastReadout(self):
        logger.debug(f"ASCOMCamera.FastReadout property called")
        return self._device.FastReadout

    @FastReadout.setter
    def FastReadout(self, value):
        logger.debug(f"ASCOMCamera.FastReadout property set to {value}")
        self._device.FastReadout = value

    @property
    def FullWellCapacity(self):
        logger.debug(f"ASCOMCamera.FullWellCapacity property called")
        return self._device.FullWellCapacity

    @property
    def Gain(self):
        logger.debug(f"ASCOMCamera.Gain property called")
        return self._device.Gain

    @Gain.setter
    def Gain(self, value):
        logger.debug(f"ASCOMCamera.Gain property set to {value}")
        self._device.Gain = value

    @property
    def GainMax(self):
        logger.debug(f"ASCOMCamera.GainMax property called")
        return self._device.GainMax

    @property
    def GainMin(self):
        logger.debug(f"ASCOMCamera.GainMin property called")
        return self._device.GainMin

    @property
    def Gains(self):
        logger.debug(f"ASCOMCamera.Gains property called")
        return self._device.Gains

    @property
    def HasShutter(self):
        logger.debug(f"ASCOMCamera.HasShutter property called")
        return self._device.HasShutter

    @property
    def HeatSinkTemperature(self):
        logger.debug(f"ASCOMCamera.HeatSinkTemperature property called")
        return self._device.HeatSinkTemperature

    @property
    def ImageArray(self):
        """Return the image array as a numpy array of the correct data type and in
        standard FITS orientation. \b

        Return the image array as a numpy array of the correct data type. The
        data type is determined by the MaxADU property. If the MaxADU property
        is not defined, or if it is less than or equal to 65535, the data type
        will be numpy.uint16. If the MaxADU property is greater than 65535, the
        data type will be numpy.uint32.

        .. Note::
            The image array is returned in the standard FITS orientation, which
            deviates from the ASCOM standard (see below).

        The image array is returned in the standard FITS orientation, with the
        rows and columns transposed (if `_DoTranspose` is `True`). This is the same orientation as the
        astropy.io.fits package. This is done because the ASCOM standard
        specifies that the image array should be returned with the first index
        being the column and the second index being the row. This is the
        opposite of the FITS standard, which specifies that the first index
        should be the row and the second index should be the column. The
        astropy.io.fits package follows the FITS standard, so the image array
        returned by the pyscope ASCOM driver is transposed to match the FITS
        standard.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            The image array as a numpy array of the correct data type.
            Rows and columns are transposed to match the FITS standard.

        """
        logger.debug(f"ASCOMCamera.ImageArray property called")
        img_array = self._device.ImageArray
        # Convert to numpy array and check if it is the correct data type
        if self._image_data_type is None:
            self.SetImageDataType()
        img_array = np.array(img_array, dtype=self._image_data_type)
        if self._DoTranspose:
            img_array = np.transpose(img_array)
        return img_array

    @property
    def ImageReady(self):
        logger.debug(f"ASCOMCamera.ImageReady property called")
        return self._device.ImageReady

    @property
    def IsPulseGuiding(self):
        logger.debug(f"ASCOMCamera.IsPulseGuiding property called")
        return self._device.IsPulseGuiding

    @property
    def LastExposureDuration(self):
        logger.debug(f"ASCOMCamera.LastExposureDuration property called")
        last_exposure_duration = self._device.LastExposureDuration
        if last_exposure_duration is None or last_exposure_duration == 0:
            last_exposure_duration = self.LastInputExposureDuration
            self._camera_time = False
        return last_exposure_duration

    @property
    def LastExposureStartTime(self):
        logger.debug(f"ASCOMCamera.LastExposureStartTime property called")
        last_time = self._device.LastExposureStartTime
        """ This code is needed to handle the case of the ASCOM ZWO driver
        which returns an empty string instead of None if the camera does not
        support the property """
        return (
            last_time
            if last_time != "" and last_time != None
            else self._last_exposure_start_time
        )

    @property
    def LastInputExposureDuration(self):
        logger.debug(f"ASCOMCamera.LastInputExposureDuration property called")
        return self._last_exposure_duration

    @LastInputExposureDuration.setter
    def LastInputExposureDuration(self, value):
        logger.debug(f"ASCOMCamera.LastInputExposureDuration property set to {value}")
        self._last_exposure_duration = value

    @property
    def MaxADU(self):
        logger.debug(f"ASCOMCamera.MaxADU property called")
        return self._device.MaxADU

    @property
    def MaxBinX(self):
        logger.debug(f"ASCOMCamera.MaxBinX property called")
        return self._device.MaxBinX

    @property
    def MaxBinY(self):
        logger.debug(f"ASCOMCamera.MaxBinY property called")
        return self._device.MaxBinY

    @property
    def NumX(self):
        logger.debug(f"ASCOMCamera.NumX property called")
        return self._device.NumX

    @NumX.setter
    def NumX(self, value):
        logger.debug(f"ASCOMCamera.NumX property set to {value}")
        self._device.NumX = value

    @property
    def NumY(self):
        logger.debug(f"ASCOMCamera.NumY property called")
        return self._device.NumY

    @NumY.setter
    def NumY(self, value):
        logger.debug(f"ASCOMCamera.NumY property set to {value}")
        self._device.NumY = value

    @property
    def Offset(self):
        logger.debug(f"ASCOMCamera.Offset property called")
        return self._device.Offset

    @Offset.setter
    def Offset(self, value):
        logger.debug(f"ASCOMCamera.Offset property set to {value}")
        self._device.Offset = value

    @property
    def OffsetMax(self):
        logger.debug(f"ASCOMCamera.OffsetMax property called")
        return self._device.OffsetMax

    @property
    def OffsetMin(self):
        logger.debug(f"ASCOMCamera.OffsetMin property called")
        return self._device.OffsetMin

    @property
    def Offsets(self):
        logger.debug(f"ASCOMCamera.Offsets property called")
        return self._device.Offsets

    @property
    def PercentCompleted(self):
        logger.debug(f"ASCOMCamera.PercentCompleted property called")
        return self._device.PercentCompleted

    @property
    def PixelSizeX(self):
        logger.debug(f"ASCOMCamera.PixelSizeX property called")
        return self._device.PixelSizeX

    @property
    def PixelSizeY(self):
        logger.debug(f"ASCOMCamera.PixelSizeY property called")
        return self._device.PixelSizeY

    @property
    def ReadoutMode(self):
        logger.debug(f"ASCOMCamera.ReadoutMode property called")
        return self._device.ReadoutMode

    @ReadoutMode.setter
    def ReadoutMode(self, value):
        logger.debug(f"ASCOMCamera.ReadoutMode property set to {value}")
        self._device.ReadoutMode = value

    @property
    def ReadoutModes(self):
        logger.debug(f"ASCOMCamera.ReadoutModes property called")
        return self._device.ReadoutModes

    @property
    def SensorName(self):
        logger.debug(f"ASCOMCamera.SensorName property called")
        return self._device.SensorName

    @property
    def SensorType(self):
        logger.debug(f"ASCOMCamera.SensorType property called")
        return self._device.SensorType

    @property
    def SetCCDTemperature(self):
        logger.debug(f"ASCOMCamera.SetCCDTemperature property called")
        return self._device.SetCCDTemperature

    @SetCCDTemperature.setter
    def SetCCDTemperature(self, value):
        logger.debug(f"ASCOMCamera.SetCCDTemperature property set to {value}")
        self._device.SetCCDTemperature = value

    @property
    def StartX(self):
        logger.debug(f"ASCOMCamera.StartX property called")
        return self._device.StartX

    @StartX.setter
    def StartX(self, value):
        logger.debug(f"ASCOMCamera.StartX property set to {value}")
        self._device.StartX = value

    @property
    def StartY(self):
        logger.debug(f"ASCOMCamera.StartY property called")
        return self._device.StartY

    @StartY.setter
    def StartY(self, value):
        logger.debug(f"ASCOMCamera.StartY property set to {value}")
        self._device.StartY = value

    @property
    def SubExposureDuration(self):
        logger.debug(f"ASCOMCamera.SubExposureDuration property called")
        return self._device.SubExposureDuration

    @SubExposureDuration.setter
    def SubExposureDuration(self, value):
        logger.debug(f"ASCOMCamera.SubExposureDuration property set to {value}")
        self._device.SubExposureDuration = value
