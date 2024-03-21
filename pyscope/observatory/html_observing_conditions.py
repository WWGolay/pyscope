import logging
import urllib.request

from ..utils import _get_number_from_line
from .observing_conditions import ObservingConditions

logger = logging.getLogger(__name__)


class HTMLObservingConditions(ObservingConditions):
    def __init__(
        self,
        url,
        cloud_cover_keyword=b"CLOUDCOVER",
        cloud_cover_units=b"%",
        cloud_cover_numeric=True,
        dew_point_keyword=b"DEWPOINT",
        dew_point_units=b"F",
        dew_point_numeric=True,
        humidity_keyword=b"HUMIDITY",
        humidity_units=b"%",
        humidity_numeric=True,
        pressure_keyword=b"PRESSURE",
        pressure_units=b"inHg",
        pressure_numeric=True,
        rain_rate_keyword=b"RAINRATE",
        rain_rate_units=b"inhr",
        rain_rate_numeric=True,
        sky_brightness_keyword=b"SKYBRIGHTNESS",
        sky_brightness_units="magdeg2",
        sky_brightness_numeric=True,
        sky_quality_keyword=b"SKYQUALITY",
        sky_quality_units="",
        sky_quality_numeric=True,
        sky_temperature_keyword=b"SKYTEMPERATURE",
        sky_temperature_units=b"F",
        sky_temperature_numeric=True,
        star_fwhm_keyword=b"STARFWHM",
        star_fwhm_units=b"arcsec",
        star_fwhm_numeric=True,
        temperature_keyword=b"TEMPERATURE",
        temperature_units=b"F",
        temperature_numeric=True,
        wind_direction_keyword=b"WINDDIRECTION",
        wind_direction_units=b"EofN",
        wind_direction_numeric=True,
        wind_gust_keyword=b"WINDGUST",
        wind_gust_units="mph",
        wind_gust_numeric=True,
        wind_speed_keyword=b"WINDSPEED",
        wind_speed_units="mph",
        wind_speed_numeric=True,
        last_updated_keyword=b"LASTUPDATED",
        last_updated_units="",
        last_updated_numeric=True,
    ):
        logger.debug(
            f"""HTMLObservingConditions.__init__(
                    {url},
                    cloud_cover_keyword={cloud_cover_keyword},
                    cloud_cover_units={cloud_cover_units},
                    cloud_cover_numeric={cloud_cover_numeric},
                    dew_point_keyword={dew_point_keyword},
                    dew_point_units={dew_point_units},
                    dew_point_numeric={dew_point_numeric},
                    humidity_keyword={humidity_keyword},
                    humidity_units={humidity_units},
                    humidity_numeric={humidity_numeric},
                    pressure_keyword={pressure_keyword},
                    pressure_units={pressure_units},
                    pressure_numeric={pressure_numeric},
                    rain_rate_keyword={rain_rate_keyword},
                    sky_brightness_keyword={sky_brightness_keyword},
                    sky_brightness_units={sky_brightness_units},
                    sky_brightness_numeric={sky_brightness_numeric},
                    sky_quality_keyword={sky_quality_keyword},
                    sky_quality_units={sky_quality_units},
                    sky_quality_numeric={sky_quality_numeric},
                    sky_temperature_keyword={sky_temperature_keyword},
                    sky_temperature_units={sky_temperature_units},
                    sky_temperature_numeric={sky_temperature_numeric},
                    star_fwhm_keyword={star_fwhm_keyword},
                    star_fwhm_units={star_fwhm_units},
                    star_fwhm_numeric={star_fwhm_numeric},
                    temperature_keyword={temperature_keyword},
                    temperature_units={temperature_units},
                    temperature_numeric={temperature_numeric},
                    wind_direction_keyword={wind_direction_keyword},
                    wind_direction_units={wind_direction_units},
                    wind_direction_numeric={wind_direction_numeric},
                    wind_gust_keyword={wind_gust_keyword},
                    wind_gust_units={wind_gust_units},
                    wind_gust_numeric={wind_gust_numeric},
                    wind_speed_keyword={wind_speed_keyword},
                    wind_speed_units={wind_speed_units},
                    wind_speed_numeric={wind_speed_numeric},
                    last_updated_keyword={last_updated_keyword},
                    last_updated_units={last_updated_units},
                    last_updated_numeric={last_updated_numeric}) called"""
        )

        self._url = url

        self._cloud_cover = None
        self._cloud_cover_keyword = cloud_cover_keyword
        self._cloud_cover_units = cloud_cover_units
        self._cloud_cover_numeric = cloud_cover_numeric
        self._dew_point = None
        self._dew_point_keyword = dew_point_keyword
        self._dew_point_units = dew_point_units
        self._dew_point_numeric = dew_point_numeric
        self._humidity = None
        self._humidity_keyword = humidity_keyword
        self._humidity_units = humidity_units
        self._humidity_numeric = humidity_numeric
        self._pressure = None
        self._pressure_keyword = pressure_keyword
        self._pressure_units = pressure_units
        self._pressure_numeric = pressure_numeric
        self._rain_rate = None
        self._rain_rate_keyword = rain_rate_keyword
        self._rain_rate_units = rain_rate_units
        self._rain_rate_numeric = rain_rate_numeric
        self._sky_brightness = None
        self._sky_brightness_keyword = sky_brightness_keyword
        self._sky_brightness_units = sky_brightness_units
        self._sky_brightness_numeric = sky_brightness_numeric
        self._sky_quality = None
        self._sky_quality_keyword = sky_quality_keyword
        self._sky_quality_units = sky_quality_units
        self._sky_quality_numeric = sky_quality_numeric
        self._sky_temperature = None
        self._sky_temperature_keyword = sky_temperature_keyword
        self._sky_temperature_units = sky_temperature_units
        self._sky_temperature_numeric = sky_temperature_numeric
        self._star_fwhm = None
        self._star_fwhm_keyword = star_fwhm_keyword
        self._star_fwhm_units = star_fwhm_units
        self._star_fwhm_numeric = star_fwhm_numeric
        self._temperature = None
        self._temperature_keyword = temperature_keyword
        self._temperature_units = temperature_units
        self._temperature_numeric = temperature_numeric
        self._wind_direction = None
        self._wind_direction_keyword = wind_direction_keyword
        self._wind_direction_units = wind_direction_units
        self._wind_direction_numeric = wind_direction_numeric
        self._wind_gust = None
        self._wind_gust_keyword = wind_gust_keyword
        self._wind_gust_units = wind_gust_units
        self._wind_gust_numeric = wind_gust_numeric
        self._wind_speed = None
        self._wind_speed_keyword = wind_speed_keyword
        self._wind_speed_units = wind_speed_units
        self._wind_speed_numeric = wind_speed_numeric
        self._last_updated = None
        self._last_updated_keyword = last_updated_keyword
        self._last_updated_units = last_updated_units
        self._last_updated_numeric = last_updated_numeric

        self.Refresh()

    def Refresh(self):
        logger.debug(f"HTMLObservingConditions.Refresh() called")
        stream = urllib.request.urlopen(self._url)
        lines = stream.readlines()

        for line in lines:
            cloud_cover = _get_number_from_line(
                line,
                self._cloud_cover_keyword,
                self._cloud_cover_units,
                self._cloud_cover_numeric,
            )
            dew_point = _get_number_from_line(
                line,
                self._dew_point_keyword,
                self._dew_point_units,
                self._dew_point_numeric,
            )
            humidity = _get_number_from_line(
                line,
                self._humidity_keyword,
                self._humidity_units,
                self._humidity_numeric,
            )
            pressure = _get_number_from_line(
                line,
                self._pressure_keyword,
                self._pressure_units,
                self._pressure_numeric,
            )
            rain_rate = _get_number_from_line(
                line,
                self._rain_rate_keyword,
                self._rain_rate_units,
                self._rain_rate_numeric,
            )
            sky_brightness = _get_number_from_line(
                line,
                self._sky_brightness_keyword,
                self._sky_brightness_units,
                self._sky_brightness_numeric,
            )
            sky_quality = _get_number_from_line(
                line,
                self._sky_quality_keyword,
                self._sky_quality_units,
                self._sky_quality_numeric,
            )
            sky_temperature = _get_number_from_line(
                line,
                self._sky_temperature_keyword,
                self._sky_temperature_units,
                self._sky_temperature_numeric,
            )
            star_fwhm = _get_number_from_line(
                line,
                self._star_fwhm_keyword,
                self._star_fwhm_units,
                self._star_fwhm_numeric,
            )
            temperature = _get_number_from_line(
                line,
                self._temperature_keyword,
                self._temperature_units,
                self._temperature_numeric,
            )
            wind_direction = _get_number_from_line(
                line,
                self._wind_direction_keyword,
                self._wind_direction_units,
                self._wind_direction_numeric,
            )
            wind_gust = _get_number_from_line(
                line,
                self._wind_gust_keyword,
                self._wind_gust_units,
                self._wind_gust_numeric,
            )
            wind_speed = _get_number_from_line(
                line,
                self._wind_speed_keyword,
                self._wind_speed_units,
                self._wind_speed_numeric,
            )
            last_updated = _get_number_from_line(
                line,
                self._last_updated_keyword,
                self._last_updated_units,
                self._last_updated_numeric,
            )

            if cloud_cover is not None:  # pragma: no cover
                self._cloud_cover = cloud_cover
            if dew_point is not None:  # pragma: no cover
                self._dew_point = dew_point
            if humidity is not None:  # pragma: no cover
                self._humidity = humidity
            if pressure is not None:  # pragma: no cover
                self._pressure = pressure
            if rain_rate is not None:  # pragma: no cover
                self._rain_rate = rain_rate
            if sky_brightness is not None:  # pragma: no cover
                self._sky_brightness = sky_brightness
            if sky_quality is not None:  # pragma: no cover
                self._sky_quality = sky_quality
            if sky_temperature is not None:  # pragma: no cover
                self._sky_temperature = sky_temperature
            if star_fwhm is not None:  # pragma: no cover
                self._star_fwhm = star_fwhm
            if temperature is not None:  # pragma: no cover
                self._temperature = temperature
            if wind_direction is not None:  # pragma: no cover
                self._wind_direction = wind_direction
            if wind_gust is not None:  # pragma: no cover
                self._wind_gust = wind_gust
            if wind_speed is not None:  # pragma: no cover
                self._wind_speed = wind_speed
            if last_updated is not None:  # pragma: no cover
                self._last_updated = last_updated

    def SensorDescription(self, PropertyName):
        logger.debug("HTMLObservingConditions.SensorDescription({PropertyName}) called")
        return str(eval(f"self._{PropertyName.lower()}_keyword"))

    def TimeSinceLastUpdate(self, PropertyName):
        logger.debug(
            "HTMLObservingConditions.TimeSinceLastUpdate({PropertyName}) called"
        )
        stream = urllib.request.urlopen(self._url)
        lines = stream.readlines()

        for line in lines:
            last_updated = _get_number_from_line(
                line,
                self._last_updated_keyword,
                self._last_updated_units,
                self._last_updated_numeric,
            )
            if last_updated is not None:
                self._last_updated = last_updated

        return self.LastUpdated

    @property
    def AveragePeriod(self):
        logger.debug("HTMLObservingConditions.AveragePeriod property called")
        return

    @AveragePeriod.setter
    def AveragePeriod(self, value):
        logger.debug(f"HTMLObservingConditions.AveragePeriod({value}) called")
        return

    @property
    def CloudCover(self):
        logger.debug("HTMLObservingConditions.CloudCover property called")
        return self._cloud_cover

    @property
    def Description(self):
        logger.debug("HTMLObservingConditions.Description property called")
        return "HTML Observing Conditions Driver"

    @property
    def DriverVersion(self):
        logger.debug("HTMLObservingConditions.DriverVersion property called")
        return None

    @property
    def DriverInfo(self):
        logger.debug("HTMLObservingConditions.DriverInfo property called")
        return "HTML Observing Conditions Driver"

    @property
    def DewPoint(self):
        logger.debug("HTMLObservingConditions.DewPoint property called")
        return self._dew_point

    @property
    def Humidity(self):
        logger.debug("HTMLObservingConditions.Humidity property called")
        return self._humidity

    @property
    def InterfaceVersion(self):
        logger.debug("HTMLObservingConditions.InterfaceVersion property called")
        return 1

    @property
    def Name(self):
        logger.debug("HTMLObservingConditions.Name property called")
        return self._url

    @property
    def Pressure(self):
        logger.debug("HTMLObservingConditions.Pressure property called")
        return self._pressure

    @property
    def RainRate(self):
        logger.debug("HTMLObservingConditions.RainRate property called")
        return self._rain_rate

    @property
    def SkyBrightness(self):
        logger.debug("HTMLObservingConditions.SkyBrightness property called")
        return self._sky_brightness

    @property
    def SkyQuality(self):
        logger.debug("HTMLObservingConditions.SkyQuality property called")
        return self._sky_quality

    @property
    def SkyTemperature(self):
        logger.debug("HTMLObservingConditions.SkyTemperature property called")
        return self._sky_temperature

    @property
    def StarFWHM(self):
        logger.debug("HTMLObservingConditions.StarFWHM property called")
        return self._star_fwhm

    @property
    def Temperature(self):
        logger.debug("HTMLObservingConditions.Temperature property called")
        return self._temperature

    @property
    def WindDirection(self):
        logger.debug("HTMLObservingConditions.WindDirection property called")
        return self._wind_direction

    @property
    def WindGust(self):
        logger.debug("HTMLObservingConditions.WindGust property called")
        return self._wind_gust

    @property
    def WindSpeed(self):
        logger.debug("HTMLObservingConditions.WindSpeed property called")
        return self._wind_speed

    @property
    def LastUpdated(self):
        logger.debug("HTMLObservingConditions.LastUpdated property called")
        return self._last_updated
