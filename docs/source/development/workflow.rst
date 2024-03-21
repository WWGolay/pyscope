*********************
Contribution workflow
*********************

Making changes
--------------
As you are modifying the code, you should incrementally commit your changes to
your local repository. This is done with the following commands::

    git add <files>
    git commit

When you are ready to publish your changes to the public repository, you
should first make sure that your local repository is synchronized with the
public repository::

    git pull --rebase

If there are any conflicts, you will need to resolve them. Once your local
repository is synchronized, you can publish your changes::

    git push


Code formatting
---------------
The development installation includes the `black <https://black.readthedocs.io/en/stable/>`_
and `isort <https://pycqa.github.io/isort/>`_ packages, which can be used to
automatically format your code::

    black .
    isort --profile black .

The `pyscope` repository also includes a configuration file for
`pre-commit <https://pre-commit.com/>`_ that will automatically run a series of
checks on your code before you commit it, including running
`black <https://black.readthedocs.io/en/stable/>`_ and
`isort <https://pycqa.github.io/isort/>`_ to format your code. To avoid having to
re-stage your changes, you can run the checks manually with::

    pre-commit run --all-files

You can also run the checks on a single file with::

    pre-commit run <file>

Running tests
-------------
The `pyscope` repository includes a `pytest <https://docs.pytest.org/en/latest/>`_
test suite. Once you have completed a feature or fixed a bug, you should add a test to the
test suite to ensure that the bug does not reappear in the future.
You can run the tests with::

    pytest

Documentation
-------------
The `pyscope` repository uses `Sphinx <https://www.sphinx-doc.org/en/master/>`_
to generate documentation. If your feature or bug fix requires changes to the
documentation, you should make those changes in the ``docs`` directory. You can
test a local build of the documentation with::

    sphinx-build -b html docs/source docs/build

Note that your pull requests should not include the ``docs/build`` directory.
The documentation will be automatically re-built and published when your pull request
is merged. You should ensure there are no warnings or errors in the build log
before submitting your pull request.

If you are adding new functionality, it may be useful to add an example to the
``examples`` directory.

Pull requests
-------------
.. important::

    Before submitting a pull request, be sure to update the CHANGELOG.md
    file with a description of your changes.

Once you have committed your changes to your local repository, you can submit
a pull request to the public repository. You can do this by visiting the
`<https://github.com/WWGolay/pyscope/pulls>`_ and clicking the
"New pull request" button. You should ensure that your pull request includes a
description of the changes you have made and the reason for making them.

.. hint::

    It may sometimes be helpful to create a pull request early in the development process
    as a draft and mentioning a specific `issue <https://github.com/WWGolay/pyscope/issues>`_
    number in the description. This will allow you to get feedback on your proposed
    changes before you have completed them and alert the development team that
    someone is working on the issue.

When you create a pull request, a number of checks will be run on your code. If
any of these checks fail, you will need to fix the issues before your pull
request can be merged. You can view the status of the checks by clicking on the
"Checks" tab in the pull request. If you are having trouble understanding the
results of the checks, you can ask for help in the
`discussions <https://github.com/WWGolay/pyscope/discussions>`_ section of the
repository.

Once you mark your pull request as ready for review, it will be reviewed by a
member of the development team. If there are any issues with your pull request,
you will be asked to make changes. Once your pull request has been approved, it
will be merged into the public repository.
