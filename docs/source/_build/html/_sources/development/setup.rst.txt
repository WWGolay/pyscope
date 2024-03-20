***************************************
Setting up your development environment
***************************************

Recommended tools
-----------------
Many of the tools listed below are optional, but highly recommended by the `pyscope`
development team. All of them are free, some require an academic email address.

1. `Miniconda <https://github.com/conda/conda>`_ (Python 3.10+ version) or if you use
   an ARM-based MacBook (M1+) `Miniforge <https://github.com/conda-forge/miniforge>`_.
   (Other recommended ARM tools can be found in `this GitHub repository <https://github.com/shreyashankar/m1-setup>`_
   from Shreyas Shankar and `this website <https://isapplesiliconready.com/>`_).
2. `GitHub <https://github.com/>`_ account. If you have an academic
   email address, we strongly recommend that you sign up for the
   `GitHub Student Developer Pack <https://education.github.com/pack>`_. If you are using Windows,
   you may have to install `Git <https://git-scm.com/>`_ separately.
3. `VSCode <https://code.visualstudio.com/>`_ is a powerful IDE with many extensions
   that make it a great choice for `pyscope` development. We recommend the following
   extensions:

   * `Python <https://marketplace.visualstudio.com/items?itemName=ms-python.python>`_: Python development
   * `Jupyter <https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter>`_: Jupyter notebook support
   * `black code formatter <https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter>`_
     for code formatting compatability with `pyscope`
   * `isort <https://marketplace.visualstudio.com/items?itemName=pycqa.isort>`_: import sorting compatability
     with `pyscope`
   * `GitHub Copilot <https://marketplace.visualstudio.com/items?itemName=GitHub.copilot>`_: AI-powered code
     completion (free with the GitHub Student Developer Pack)
   * `GitHub Actions <https://marketplace.visualstudio.com/items?itemName=github.vscode-github-actions>`_:
     GitHub Actions config file support
   * `GitLens <https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens>`_: line-by-line blame preview
     and other version control tools (Pro version free with the GitHub Student Developer Pack)
   * `ErrorLens <https://marketplace.visualstudio.com/items?itemName=usernamehw.errorlens>`_: inline error
     highlighting
   * `CodeCov <https://marketplace.visualstudio.com/items?itemName=codecov.codecov>`_: CodeCov integration
   * `esbonio <https://marketplace.visualstudio.com/items?itemName=swyddfa.esbonio>`_:
     `Sphinx <https://www.sphinx-doc.org/en/master/>`_ documentation preview window with on-the-fly compilation
   * `Simple RST <https://marketplace.visualstudio.com/items?itemName=trond-snekvikl.simple-rst>`_ for RST linting
   * `Markdown Lint <https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint>`_ for
     Markdown linting
   * `HTML Preview <https://marketplace.visualstudio.com/items?itemName=george-alisson.html-preview-vscode>`_ for
     inline HTML preview
   * `Path IntelliSense <https://marketplace.visualstudio.com/items?itemName=christian-kohler.path-intellisense>`_
     for path autocompletion
   * `Indent Rainbow <https://marketplace.visualstudio.com/items?itemName=oderwat.indent-rainbow>`_ for
     indentation highlighting (useful for Python specifically)
   * `VSCode Icons <https://marketplace.visualstudio.com/items?itemName=vscode-icons-team.vscode-icons>`_ for
     file type icons
   * `One Dark Pro <https://marketplace.visualstudio.com/items?itemName=zhuangtongfa.material-theme>`_ for
     a dark theme

Creating a new conda environment
--------------------------------
We recommend that you have an environment dedicated to `pyscope` development. To create
a new environment, run the following command in your terminal::

    conda create -n pyscope-dev python=3.11

This will create a new environment called ``pyscope-dev`` with Python 3.11 installed.
To activate the environment, run::

    conda activate pyscope-dev

You can deactivate the environment with::

        conda deactivate

Forking the repository
----------------------
To contribute to `pyscope`, you will need to fork the repository. To do this, go to the
`pyscope GitHub page <https://github.com/WWGolay/pyscope>`_ and click the "Fork" button
in the top right corner. This will create a copy of the repository in your GitHub account.
Move to a location on your computer where you want to store the repository and run::

    git clone https://github.com/<your-username>/pyscope

where ``<your-username>`` is your GitHub username. This will create a new folder called
``pyscope`` with the repository contents. Move into this folder with::

    cd pyscope

Installing dependencies
-----------------------
Once you have cloned the repository, you can install the development version of `pyscope`
with::

    pip install -e ".[dev]"

This will install `pyscope` in editable mode with all of the development dependencies.
We recommend that you use the `VSCode <https://code.visualstudio.com/>`_
IDE with the extensions listed above. You can open the repository in VSCode with::

    code .

GitHub Codespaces
-----------------
|Codespaces|

The `pyscope` repository includes a `devcontainer.json <https://code.visualstudio.com/docs/remote/devcontainerjson-reference>`_
file that allows you to develop `pyscope` in a
`GitHub Codespace <https://codespaces.new/WWGolay/pyscope>`_ with many of the above
tools pre-installed. This is a handy way to develop `pyscope` without having to go
through the setup process on your local machine. To use a Codespace, click the
"Code" button in the top right corner of the
`pyscope GitHub page <https://github.com/WWGolay/pyscope>`_ and select "Open with Codespaces".
This will open a Codespace in your browser. If you have `VSCode <https://code.visualstudio.com/>`_
installed, you can click the "Open with VS Code Desktop" button in the menu from the upper
left to use the Codespace from the familiar VSCode interface. You may be prompted to
install the `GitHub Codespaces <https://marketplace.visualstudio.com/items?itemName=GitHub.codespaces>`_
extension and other helper extensions.


.. |Codespaces| image:: https://github.com/codespaces/badge.svg
    :target: https://codespaces.new/WWGolay/pyscope
    :alt: Codespaces
