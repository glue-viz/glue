:orphan:

Single-file installation
========================

About
-----

On this page, you will find installers for different platforms which will set up
a self-contained conda-based Python installation named **glueconda** that
includes Python, glue and a number of plugins by default. This is not the
recommended method of installation for regular use if you already have other
Python installations (instead you should try and install glue into your existing
installations using instructions in :doc:`installation`), but it can provide
an easy way to get started, especially during workshops/tutorials.

Important note
---------------

Using the installers below will create a ``glueconda`` folder in your home
directory and will optionally add ``glueconda/bin`` to your path. Since
``glueconda/bin`` contains a ``python`` executable, this may take precedence
over other Python installations, hence why it is preferable if possible to
install glue into your existing Python environments. See `Uninstalling`_ for
information about removing glueconda.

MacOS X
-------

There are two installation options for MacOS X - you can either download the
`glueconda-stable-MacOSX-x86_64.pkg <http://www.glueviz.org.s3-website-us-east-1.amazonaws.com/glueconda/glueconda-stable-MacOSX-x86_64.pkg>`__
graphical installer, and double click on it to run the installation, or you can
download the
`glueconda-stable-MacOSX-x86_64.sh <http://www.glueviz.org.s3-website-us-east-1.amazonaws.com/glueconda/glueconda-stable-MacOSX-x86_64.sh>`__
command-line installer then run the installer using::

    bash glueconda-stable-MacOSX-x86_64.sh

32-bit Linux
------------

Download
`glueconda-stable-Linux-x86.sh <http://www.glueviz.org.s3-website-us-east-1.amazonaws.com/glueconda/glueconda-stable-Linux-x86.sh>`__
then run the installer using::

    bash glueconda-stable-Linux-x86.sh

64-bit Linux
------------

Download
`glueconda-stable-Linux-x86_64.sh <http://www.glueviz.org.s3-website-us-east-1.amazonaws.com/glueconda/glueconda-stable-Linux-x86_64.sh>`__
then run the installer using::

    bash glueconda-stable-Linux-x86_64.sh

Windows
-------

Download and run the
`glueconda-stable-Windows-x86_64.exe <http://www.glueviz.org.s3-website-us-east-1.amazonaws.com/glueconda/glueconda-stable-Windows-x86_64.exe>`__
installer.

Uninstalling
------------

To uninstall, remove the ``glueconda`` folder from your home directory.

Updating packages
-----------------

If you decide to continue using **glueconda**, and a new version of glue or
other packages becomes available, you can easily update all packages using::

    conda update -c glueviz --all
