.. _anaconda_gui:

Anaconda Python Distribution (Navigator)
========================================

**Platforms:** MacOS X, Linux, and Windows

We recommend using the `Anaconda <http://continuum.io/downloads.html>`__ Python
distribution from Continuum Analytics (or the related Miniconda distribution).
Anaconda includes all of Glue's main dependencies. There are two ways of
installing Glue with the Anaconda Python Distribution: :ref:`graphically using the
Anaconda Navigator <anaconda_gui>`, or :ref:`using the conda command
<anaconda_cli>` on the command-line, both of which are described
below.


Once Anaconda is installed, go to the **Applications** folder and launch the
**Anaconda Navigator**:

.. image:: images/navigator_icon.png
   :align: center
   :width: 80

If you do not have the Anaconda Navigator icon, but have an Anaconda Launcher,
you are using an old version of Anaconda. See :ref:`here <anaconda_old_gui>` for
alternate instructions.

Assuming you have the navigator open, before installing glue first click on the
**Channels** button:

.. image:: images/navigator_channels_button.png
   :align: center
   :width: 373

If not already present, add **conda-forge** to the list of channels by clicking
on **Add**, typing **conda-forge**, and pressing enter, then click on **Update
channels**:

.. image:: images/navigator_channels_dialog.png
   :align: center
   :width: 414

You can now install the latest version of glue by clicking on **Install**:

.. image:: images/navigator_install.png
   :align: center
   :width: 264

Once the installation is complete, you can click on the **Launch** button (which
will replace the **Install** button).
