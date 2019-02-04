.. _writing_plugin:

Distributing your own plugin package
====================================

If you are looking to customize glue for your own use, you don't necessarily
need to create a plugin package - instead you can just use a ``config.py`` file
as described in :doc:`configuration`. However, if you are interested in sharing
your customizations with others, then the best approach is to develop and
distribute a plugin package.

Plugin packages use the same mechanism of registering customizations described
in :ref:`customization` as you would if you were using a ``config.py`` file -
the only real difference is the file structure you will need to use. To make
things easier, we provide a template plugin package at
https://github.com/glue-viz/glue-plugin-template to show you how files should be
organized.

Required files
--------------

To start with, any Python code that is part of the plugin package should be
placed in a directory with the name of the module for the plugin. In our
template, this is the ``myplugin`` directory - for some of the real plugins we
have developed in the past, this is for example ``glue_medical`` or
``glue_geospatial``. This directory should contain at least one Python file
that contains the customizations that you would otherwise have put in your
``config.py`` file. In the template example, this is the ``data_viewer.py`` file
which contains a custom data viewer.

In addition to this file (or multiple files), you will need an ``__init__.py``
file, which should contain a ``setup`` function. This function is used to
register any customizations with glue. In this function, you should import any
files containing customizations. If the customizations use a decorator to be
registered (e.g. ``@data_factory`` or ``@menubar_plugin``), then you are all set.
Otherwise, for registering e.g. custom_viewers, the ``setup`` function should
also do the registration - in the template, this looks like::

    def setup():
        from .data_viewer import MyViewer
        from glue.config import qt_client
        qt_client.add(MyViewer)

Finally, at the root of the package, you will need a ``setup.py`` file similar
to the one in the template (you can copy it over and edit the relevant parts).
One of the important parts is the definition of ``entry_points``, which is the
part that tells glue that this package is a plugin::

    entry_points = """
    [glue.plugins]
    myplugin=myplugin:setup
    """

The entry in ``[glue.plugins]`` has the form::

    plugin_name=module_name:setup_function_name

In general, to avoid confusion, the ``plugin_name`` and ``module_name`` should
both be set to the name of the directory containing the code (``myplugin`` in
our case).

Once you have this in place, you should be able to install the plugin in
'develop' mode (meaning that you can then make changes and have them be updated
in the installed version without having to re-install every time) with::

    pip install -e .

You can then start up glue with::

    glue -v

The startup log then contains information about whether plugins were
successfully loaded. You should either see something like::

    INFO:glue:Loading plugin myplugin succeeded

Or if you are unlucky::

    INFO:glue:Loading plugin myplugin failed (Exception: No module named 'numpyy')

In the latter case, the exception should help you figure out what went wrong.
For a more detailed error message, you can also just import your plugin package
and run the setup function::

    python -c 'from myplugin import setup; setup()'

Optional files
--------------

The only files that are really required are the directory with the source code
and the ``setup.py`` file - however, you should make sure you also include an
`open source license <https://choosealicense.com/>`_ if you are planning to
distribute the package, as well as a README file that describes your package,
its requirements, and how to install and use it.

Consider also adding tests (using e.g. the `pytest <https://www.pytest.org>`_
framework), as well as setting up continuous integration services such as
`Travis <https://travis-ci.org>`_ to run the tests any time a change is made.
Describing how to do this is beyond the scope of this tutorial, but there are
plenty of resources online to help you do this.

Distributing the package
------------------------

Since your package follows the standard layout for packages, you can follow the
`Packaging Python Projects <https://packaging.python.org/tutorials/packaging-projects/>`_
guide to release your package and upload it to PyPI. The usual release process
for glue plugins is as follows:

#. Start off by editing the changelog (if present) and change the release date
   from ``unreleased`` to today's date, in the ``YYYY-MM-DD`` format, for
   example::

      0.2 (2019-02-04)
      ----------------

#. Edit the ``version.py`` file in your package to not include the
   ``.dev0`` suffix, e.g.::

      version = '0.2'

#. Commit the changes, e.g.::

      git add CHANGES.rst */version.py
      git commit -m "Preparing release v0.2"

#. Remove any uncommitted files/changes with::

      git clean -fxd

#. Now create the release with::

      python setup.py sdist

   Provided you don't have any C extensions in your package, you can also make
   a so-called 'wheel' release::

      python setup.py bdist_wheel --universal

#. Go inside the ``dist`` directory and use the `twine
   <https://pypi.org/project/twine/>`_ tool to upload the files to PyPI::

      cd dist
      twine upload *.tar.gz *.whl

#. Optionally tag the release in git with::

      git tag -m v0.2 v0.2

#. Add a new section in the changelog file for the next release::

      0.3 (unreleased)
      ----------------

      - No changes yet.

   and update the ``version.py`` file to point to the next version, with a
   ``.dev0`` suffix::

      version = '0.3.dev0'

#. Finally, commit the changes and push to GitHub::

      git add CHANGES.rst */version.py
      git commit -m "Back to development: v0.3"
      git push --tags upstream master

If you are interested in including your package as a conda package in the
``glueviz`` channel, please let us know by opening an issue at
https://github.com/glue-viz/conda-dev.
