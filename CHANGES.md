Full changelog
==============

v0.6 (unreleased)
-----------------

* Added experimental support for PyQt5. [#663]

* Fix ``glue -t`` option. [#791]

* Updated ``glue-deps`` to show PyQt/PySide versions. [#796]

* Fix bug that caused viewers to be restored with the wrong size. [#781, #783]

* Prevent axes from moving around when data viewers are being resized, and
  instead make the absolute margins between axes and figure edge fixed. [#745]

* Fixed a bug that caused image plots to not be updated immediately when
  changing component, and fixed a bug that caused data and attribute combo
  boxes to not update correctly when showing multiple datasets in an
  ImageWidget. [#755]

* Added tests to ensure that we remain backward-compatible with old session
  files for the FITS and HDF5 factories. [#736, #748]

* When a box has been drawn to extract a spectrum from a cube, the box can
  then be moved by pressing the control key and dragging it. [#707]

* Refactored ASCII I/O to include more Astropy table formats. [#762]

* When saving a session, if no extension is specified, the .glu extension is
  added. [#729]

* Added a GUI plugin manager in the 'Plugins' menu. [#682]

* Added an option to specify whether to use an automatic aspect ratio for image
  data or whether to enforce square pixels. [#717]

* Data factories can now be given priorities to determine which ones should
  take precedence in ambiguous cases. The ``set_default_factory`` and
  ``get_default_factory`` functions are now deprecated since it is possible to
  achieve this solely with priorities. [#719]

* Improved cube slider to include editable slice value as well as
  first/previous/next/last buttons, and improved spacing of sliders for 4+
  dimensional cubes. [#690, #734]

* Registering data factories should now always be done with the
  ``@data_factory`` decorator, and not by adding functions to
  ``__factories__``, as was possible in early versions of Glue. [#724]

* Made the Excel spreadsheet reader more robust: column headers no longer have
  to be strings, and the reader no longer expects the first sheet to be called
  'Sheet1'. All sheets are now read by default. Datasets are now named as
  filename:sheetname. [#726]

* Fix compatibility with IPython 4. [#733]

* Improved reading of FITS files - all HDUs are now read by default. [#704, #732]

* Added new widget property classes, for combo boxes (based on label instead of
  data) and for tab widgets. [#752]

* Improved reading of HDF5 files - all datasets in an HDF5 file are now read by
  default. [#747]

* Fix a bug that caused images to not be shown at full resolution after resizing. [#768]

* Fix a bug that caused the color of an extracted spectrum to vary if extracted
  multiple times. [#743]

* Fixed a bug that caused compressed image HDUs to not be read correctly.
  [#767]

* Added two new settings ``settings.SUBSET_COLORS`` and ``settings.DATA_COLOR``
  that can be used to customize the default subset and data colors. [#742]

v0.5.3 (unreleased)
-------------------

* Fix selection in scatter plots when categorical data are present. [#727]

v0.5.2 (2015-08-13)
-------------------

* Fix loading of plugins with setuptools < 11.3 [#699]

* Fix loading of plugins when using glue programmatically rather than through the GUI [#698]

* Backward-compatibility fixes after refactoring data_factories [#696, #703]

v0.5.1 (2015-07-06)
-------------------

* Fixed treatment of newlines when copying detailed error. [#687]

* Fix a bug that prevented sessions from being saved with embedded files if
  component units were Astropy units. [#686]

* Users should now press 'control' to drag rather than re-define subsets. [#689]

v0.5 (2015-07-03)
-----------------

* Improvements to the PyQt/PySide wrapper module (now maintained in a separate repository). [#671]

* Fixed broken links on website. [#678]

* Added the ability to discover plugins via entry points. [#677]

* Added the ability to include float and string UI elements in custom
  viewers. [#653]

* Added an option to bundle all data in .glu session files. [#661]

* Added a ``menu_plugin`` registry to add custom tools to the registry. [#644]

* Support for 'lazy-loading' plugins which means their import is deferred
  until they are needed. [#590]

* Support for connecting custom importers. [#593]

* ``qglue`` now correctly interprets HDUList objects. [#598]

* Internal improvements to organization of domain-specific code (such as the
  Astronomy coordinate conversions and ginga data viewer). [#488, #585]

* Astronomy coordinate conversions now include more coordinate frames. [#578]

* ``load_ui`` now checks whether ``.ui`` file exists locally before
  retrieving it from the ``glue.qt.ui`` sub-package. [#599]

* Improved interface for adding new components, with syntax highlighting
  and tab-completion. [#572, #575]

* Improved error/warning messages. [#582]

* Miscellaneous bug fixes. [#637, #636, #608]

* The error console log is now available through the View menu

* Improved under-the-hood handling of categorical ROIs. [#601]

* Fixed compatibility with Python 2.6. [#540]

* Python 3.x support is now stable. [#576]

* Fixed the ability to copy detailed error messages. [#675]

* Added instructions on how to make a fully-customized Qt viewer. [#619]

* Fixes to the ginga plugin to support the latest version. [#584, #656]

* Added the ability to drag circular, rectangular, and lasso selections. [#657]

* Added the ability to reset a session. [#630]

v0.4 (Released December 22, 2015)
---------------------------------

Release Highlights:
 * Introduced custom viewers
 * Ginga-based image viewer
 * Experimental Python 3.x support

Other Notes
 * Better testing for support of optional dependencies
 * Refactored spectrum and position-velocity features from the Image widget into plugin tools
 * Adopted contracts for contracters to add optional runtime type checking
 * Added ability to export collapsed cubes as 2D fits files
 * More flexible data parsing in qglue utility
 * Numerous bugfixes
