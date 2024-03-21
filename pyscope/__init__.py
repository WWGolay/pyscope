"""
.. container::

  |License| |Zenodo| |PyPI Version| |PyPI Python Versions| |PyPI Downloads| |Astropy| |Donate|

`pyscope` is a pure-Python package for robotic scheduling, operation, and control of small
optical telescopes.

.. important::
  If you use `pyscope` for data collection or work
  presented in a publication or talk, please help the project by
  properly `acknowledging or citing </cite>`_ the package.

`pyscope` is an
`open-source <license>` project that provides a set of tools to rapidly and easily
control astronomical instrumentation. It is designed to be modular and extensible,
allowing users to easily add support for new devices and observatories.
`pyscope` is built on top of the
`ASCOM <https://ascom-standards.org/>`__ standard, but also provides support for
non-ASCOM devices. Users may also access their devices through third-party applications
such as `MaxIm DL <https://diffractionlimited.com/product/maxim-dl/>`__.

Observatories who use `pyscope` can take
advantage of the `~pyscope.telrun`
module, which provides a simple interface for fully-robotic observatory control.

`pyscope` is aiming to become an
`Astropy-affiliated package <https://www.astropy.org/affiliated/>`__.

Features
========
* Control observatory hardware with Python
* Support for both `ASCOM <https://ascom-standards.org/>`__ and non-ASCOM devices
* `~pyscope.observatory.Observatory` convenience methods like
  `~pyscope.observatory.Observatory.autofocus` and
  `~pyscope.observatory.Observatory.recenter`
* `~pyscope.telrun` module for fully-robotic operation of
  an observatory
* Basic data reduction tools like `~pyscope.reduction.avg_fits` and `~pyscope.reduction.ccd_calib`
* Simple analysis scripts like `~pyscope.analysis.calc_zmag`
* Powered by `Astropy <https://www.astropy.org/>`__,
  `Astropy-affiliated <https://www.astropy.org/affiliated/>`__
  packages, and `ASCOM <https://ascom-standards.org/>`__

.. |License| image:: https://img.shields.io/pypi/l/pyscope
    :target: https://pypi.org/project/pyscope/
    :alt: License

.. |Zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8403570.svg
    :target: https://doi.org/10.5281/zenodo.8403570
    :alt: Zenodo

.. |PyPI Version| image:: https://img.shields.io/pypi/v/pyscope
    :target: https://pypi.org/project/pyscope/
    :alt: PyPI Version

.. |PyPI Python Versions| image:: https://img.shields.io/pypi/pyversions/pyscope?logo=Python
    :target: https://pypi.org/project/pyscope/
    :alt: PyPI Python Versions

.. |PyPI Downloads| image:: https://img.shields.io/pypi/dm/pyscope?logo=python
    :target: https://pypi.org/project/pyscope/
    :alt: PyPI Downloads

.. |Astropy| image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy

.. |Donate| image:: https://img.shields.io/badge/Donate-to_pyscope-crimson
  :target: https://github.com/sponsors/WWGolay
  :alt: Donate

"""

# isort: skip_file

import logging

__version__ = "0.2.0"

from . import utils
from . import observatory
from . import telrun
from . import reduction
from . import analysis

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ["analysis", "observatory", "reduction", "telrun", "utils"]
