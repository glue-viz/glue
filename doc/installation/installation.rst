.. _installation:

Installing Glue
===============

.. _anaconda:

Several installation methods for Glue are outlined in the sections below. If you
run into issues, each page should provide relevant troubleshooting, and you can
also check the :ref:`known-issues` page which collects some more general issues.
If your problem is not described there, `open a new issue
<https://github.com/glue-viz/glue/issues>`_ on GitHub.

.. toctree::
   :maxdepth: 1

   conda
   pip
   app
   qt
   dependencies
   developer

Running Glue
------------

Installing glue from source will create a executable ``glue`` script
that should be in your path. Running ``glue`` from the command line will
start the program. Glue accepts a variety of command-line
arguments. See ``glue --help`` for examples.

.. note:: On Windows, installation creates an executable ``glue.exe`` file
          within the python script directory (e.g., ``C:\Python27\Scripts``).
          Windows users can create a desktop shortcut for this file, and run
          Glue by double clicking on the icon.
