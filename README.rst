*******
pyscope
*******

.. container::

    |License| |Zenodo| |PyPI Version| |PyPI Python Versions| |PyPI Downloads| |Astropy| |GitHub CI| |Code Coverage| |Documentation Status| |Codespaces Status| |pre-commit| |Black| |isort| |Donate|

.. image:: https://github.com/WWGolay/pyscope/blob/main/docs/source/_static/pyscope_logo_white.png
    :alt: pyscope logo

This is the repository for `pyscope <https://pyscope.readthedocs.io/en/latest/>`_,
a pure-Python package for robotic scheduling, operation, and control of small
optical telescopes.

`pyscope <https://pyscope.readthedocs.io/en/latest/>`_ is an
`open-source <LICENSE>`_ project that provides a set of tools to rapidly and easily
control astronomical instrumentation. It is designed to be modular and extensible,
allowing users to easily add support for new devices and observatories.
`pyscope <https://pyscope.readthedocs.io/en/latest/>`_ is built on top of the
`ASCOM <https://ascom-standards.org/>`_ standard, but also provides support for
non-ASCOM devices. Users may also access their devices through third-party applications
such as `MaxIm DL <https://diffractionlimited.com/product/maxim-dl/>`_.

Observatories who use `pyscope <https://pyscope.readthedocs.io/en/latest/>`_ can take
advantage of the `telrun <https://pyscope.readthedocs.io/en/latest/api/pyscope.telrun.html>`_
module, which provides a simple interface for fully-robotic observatory control.

`pyscope <https://pyscope.readthedocs.io/en/latest/>`_ is aiming to become an
`astropy-affiliated package <https://www.astropy.org/affiliated/>`_.

Features
--------
* Control observatory hardware with Python

* Support for `ASCOM <https://ascom-standards.org/>`_ and non-ASCOM devices

* `Observatory <https://pyscope.readthedocs.io/en/latest/api/auto_api/pyscope.observatory.Observatory.html>`_
  convenience methods like `run_autofocus <https://pyscope.readthedocs.io/en/latest/api/auto_api/pyscope.observatory.Observatory.html#pyscope.observatory.Observatory.run_autofocus>`_
  and `recenter <https://pyscope.readthedocs.io/en/latest/api/auto_api/pyscope.observatory.Observatory.html#pyscope.observatory.Observatory.recenter>`_

* `telrun <https://pyscope.readthedocs.io/en/latest/api/pyscope.telrun.html>`_ module
  for fully-robotic operation of an observatory

* Basic data reduction tools like
  `avg_fits <https://pyscope.readthedocs.io/en/latest/api/auto_api/pyscope.reduction.avg_fits.html>`_
  and `ccd_calib <https://pyscope.readthedocs.io/en/latest/api/auto_api/pyscope.reduction.ccd_calib.html#pyscope.reduction.ccd_calib>`_

* Simple analysis scripts like
  `calc_zmag <https://pyscope.readthedocs.io/en/latest/api/auto_api/pyscope.analysis.calc_zmag.html#pyscope.analysis.calc_zmag>`_

* Powered by `Astropy <https://www.astropy.org/>`_,
  `Astropy-affiliated <https://www.astropy.org/affiliated/>`_
  packages, and `ASCOM <https://ascom-standards.org/>`_

Installation
------------
pyscope is available on PyPI and can be installed with pip:

.. code-block:: bash

    pip install pyscope

pyscope will be available on conda-forge soon.

Development Installation
========================
|Codespaces|

We recommend using a virtual environment for development. You may create a new
virtual environment with pip:

.. code-block:: bash

    python -m venv pyscope-dev
    source pyscope-dev/bin/activate

Or with conda:

.. code-block:: bash

    conda create -n pyscope-dev python=3.10.12
    conda activate pyscope-dev

To install pyscope for development, clone the repository and install with pip:

.. code-block:: bash

    git clone https://github.com/WWGolay/pyscope
    cd pyscope
    pip install -e ".[dev]"

Usage
-----
TBD

Documentation
-------------
All supporting documentation can be found at `readthedocs <https://pyscope.readthedocs.io/en/latest/>`_.

Citing
------
If you use this package in your research, please cite it using the following:

Contributing
------------
Please see the `developer documentation <https://pyscope.readthedocs.io/en/latest/development/>`_.

License
-------
This project is licensed under the `GNU AGPLv3 License <LICENSE>`_.

Issues
------
Please post any issues you find `here <https://github.com/WWGolay/pyscope/issues>`_.

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

.. |GitHub CI| image:: https://img.shields.io/github/actions/workflow/status/WWGolay/pyscope/ci.yml?logo=GitHub&label=CI
    :target: https://github.com/WWGolay/pyscope/actions/workflows/ci.yml
    :alt: GitHub CI

.. |Code Coverage| image:: https://codecov.io/gh/WWGolay/pyscope/branch/main/graph/badge.svg
    :target: https://app.codecov.io/gh/WWGolay/pyscope/
    :alt: Code Coverage

.. |Documentation Status| image:: https://img.shields.io/readthedocs/pyscope?logo=ReadtheDocs
    :target: https://pyscope.readthedocs.io/en/latest/
    :alt: Documentation Status

.. |Codespaces Status| image:: https://github.com/WWGolay/pyscope/actions/workflows/codespaces/create_codespaces_prebuilds/badge.svg
    :target: https://github.com/WWGolay/pyscope/actions/workflows/codespaces/create_codespaces_prebuilds
    :alt: Codespaces Status

.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
    :target: https://github.com/pre-commit/pre-commit
    :alt: pre-commit enabled

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code Style

.. |isort| image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
    :target: https://pycqa.github.io/isort/
    :alt: isort

.. |Donate| image:: https://img.shields.io/badge/Donate-to_pyscope-crimson
    :target: https://github.com/sponsors/WWGolay
    :alt: Donate

.. |Codespaces| image:: https://github.com/codespaces/badge.svg
    :target: https://codespaces.new/WWGolay/pyscope
    :alt: Codespaces
