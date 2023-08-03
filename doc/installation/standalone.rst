Standalone MacOS X and Windows applications
===========================================

**Platforms:** MacOS X and Windows

On MacOS X and Windows, the easiest way to install glue along with a few of
the common glue plugins is to download pre-built single-file applications.

.. note:: If you run into any issues
          it would be really helpful if you could let us know by `opening an issue
          <https://github.com/glue-viz/glue-standalone-apps/issues/new>`_.

The plugins included by default in the standalone applications are:

* `glue-vispy-viewers <https://github.com/glue-viz/glue-vispy-viewers/>`_
* `glue-wwt <https://github.com/glue-viz/glue-wwt/>`_
* `glue-plotly <https://github.com/glue-viz/glue-plotly/>`_

With this installation method, it is not possible to install additional plugins
beyond those included by default, so if you want the ability to do this, you
should check one of the other installation methods mentioned in
:ref:`installation`.

MacOS X
-------

Donwload the :download:`glue 2023.06.4.dmg
<https://glueviz.s3.amazonaws.com/installers/2023.06.4/glue%202023.06.4.dmg>`
file, open it and copy the **glue 2023.06.4.app** application to your
**Applications** folder (or any other location you want to use). You will
likely see a dialog asking you whether to continue opening the application as it
was downloaded from the internet - if so, you can proceed (this is a standard
warning for any application not installed via the Mac App Store).

Windows
-------

Donwload the :download:`glue 2023.06.4.exe
<https://glueviz.s3.amazonaws.com/installers/2023.06.4/glue%202023.06.4.exe>` file.
Once the file has downloaded, open the **glue 2023.06.4.exe** application. You
will likely then see a dialog such as:

.. image:: images/warning1_windows.png
   :align: center
   :width: 400

Click on **More info** and you will then see:

.. image:: images/warning2_windows.png
   :align: center
   :width: 400

Click on **Run anyway** and glue should now open.

Nightly builds
--------------

The applications above are built every few months to provide stability
and are checked to make sure they all work correctly. We also provide
'nightly' builds which use the latest (released) versions of all the relevant
glue packages and plugins. These are generated automatically and are
not hand-checked, so may be unstable. The download links are:

* MacOS X: :download:`glue main.dmg <https://glueviz.s3.amazonaws.com/installers/main/glue%20main.dmg>`
* Windows: :download:`glue main.exe <https://glueviz.s3.amazonaws.com/installers/main/glue%20main.exe>`
