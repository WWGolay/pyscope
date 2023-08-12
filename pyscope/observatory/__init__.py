import logging
logger = logging.getLogger(__name__)

from ._docstring_inheritee import _DocstringInheritee

from .driver import Driver

from .autofocus import Autofocus
from .camera import Camera
from .cover_calibrator import CoverCalibrator
from .dome import Dome
from .filter_wheel import FilterWheel
from .focuser import Focuser
from .observing_conditions import ObservingConditions
from .rotator import Rotator
from .switch import Switch
from .telescope import Telescope

from .ascom_driver import ASCOMDriver
from .ascom_camera import ASCOMCamera
from .ascom_cover_calibrator import ASCOMCoverCalibrator
from .ascom_dome import ASCOMDome
from .ascom_filter_wheel import ASCOMFilterWheel
from .ascom_focuser import ASCOMFocuser
from .ascom_observing_conditions import ASCOMObservingConditions
from .ascom_rotator import ASCOMRotator
from .ascom_switch import ASCOMSwitch
from .ascom_telescope import ASCOMTelescope

from .astrometry_net_wcs import AstrometryNetWCS
from .html_observing_conditions import HTMLObservingConditions
from .html_safety_monitor import HTMLSafetyMonitor
from .ip_cover_calibrator import IPCoverCalibrator
from .maxim import Maxim
from .pinpoint_wcs import PinpointWCS
from .platesolve2_wcs import Platesolve2WCS
from .platesolve3_wcs import Platesolve3WCS
from .pwi_autofocus import PWIAutofocus
from .skyx import SkyX

from .observatory import Observatory
from .observatory_exception import ObservatoryException