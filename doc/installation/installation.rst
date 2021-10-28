.. _installation:

Installing and running glue
===========================

Several installation methods for glue are outlined in the sections below. If you
run into issues, each page should provide relevant troubleshooting, and you can
also check the :ref:`known-issues` page which collects some more general issues.
If your problem is not described there, `open a new issue
<https://github.com/glue-viz/glue/issues>`_ on GitHub.

.. toctree::
   :maxdepth: 1

   standalone
   conda
   pip
   qt
   dependencies
   developer

.. note:: If you are using Apple M1 hardware, be sure to read :ref:`apple-m1`
          before proceeding with the installation instructions.

Once glue is installed, you will be able to type::

    glue

in a terminal to start glue. Glue accepts a variety of command-line arguments.
See ``glue --help`` for examples. If you used the Anaconda Navigator, you can
also launch glue from the navigator, but be aware that errors may be hidden, so
if you have any issues, try running glue from the command-line.

You can also start glue with the ``-v`` option::

    glue -v

to get more verbose output, which may help diagnose issues.

On Windows, installation creates an executable ``glue.exe`` file within the
python script directory (e.g., ``C:\Python27\Scripts``). Windows users can
create a desktop shortcut for this file, and run Glue by double clicking on the
icon.

.. note:: If you have issues with Glue after installing it on MacOS X Big Sur,
          see :ref:`apple-bigsur`.
