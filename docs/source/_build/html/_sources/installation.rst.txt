Installation
============
pyscope is available on PyPI and can be installed with pip:

.. code-block:: bash

    pip install pyscope

pyscope will be availabl on conda-forge soon.

Development Installation
------------------------
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
