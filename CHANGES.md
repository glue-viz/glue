Full changelog
==============

v0.9.0 (2016-10-10)
-------------------

* Fix serialization of celestial coordinate link functions. Classes 
  inheriting from MultiLink should now call MultiLink.__init__ with 
  individual components (not grouped into left/right) then the create_links 
  method with the components separated into left/right and the methods for 
  forward/backward transformation. The original behavior can be retained
  by using the ``multi_link`` function instead of the ``MultiLink`` class.
  [#1139]

* Improve support for spectral cubes. [#1075]

* Allow link functions/helpers to define a category using the ``category=``
  argument (which defaults to ``General``), and make it possible to filter
  by category in the link editor. [#1141]

* Only show the 'waiting' cursor when glue is doing something. [#1097]

* Make sure that the scatter layer artist style editor is shown when overplotting
  a scatter plot on top of an image. [#1134]

* Data viewers can now make layer_style_widget_cls a dictionary in cases
  where multiple layer artist types are supported. [#1134]

* Fix compatibility of test suite with pytest 3.x. [#1116]

* Updated bundled version of WCSAxes to v0.9. [#1089]

* Fix compatibility with pre-releases of Matplotlib 2.0. [#1115]

* Implement new widget with better control over exporting to Plotly. The
  preference pane for Plotly export has now been removed in favor of this new
  way to set the options. [#1057]

* Renamed the ``ComponentIDComboHelper`` and ``ManualDataComboHelper``
  ``append`` methods to ``append_data`` and the ``remove`` methods to
  ``remove_data``, and added a new ``ComponentIDComboHelper.set_multiple_data``
  method. [#1060]

* Fixed reading of units from FITS and VO tables, and display units in
  table viewer. [#1135, #1137]

* Make use of the QtPy package to deal with differences between PyQt4, PyQt5,
  and PySide, instead of the custom qt-helpers package. The
  ``glue.external.qt`` package is now deprecated. The ``get_qapp`` and
  ``load_ui`` functions are now available in ``glue.utils.qt``.
  [#1018, #1074, #1077, #1078, #1081]

* Avoid raising a (harmless) error when selecting a region in between two
  categorical components.

* Added a new Data method, ``update_values_from_data``, that can be used to replicate
  components from one dataset into another. [#1112]

* Refactored code related to toolbars in order to make it easier to define
  toolbars and toolbar modes that aren't Matplotlib-specific. [#1085, #1120]

* Added a new table viewer. [#1084, #1123]

* Fix saving/loading of categorical components. [#1084]

* Make it possible for tools to define a status bar message. [#1084]

* Added a command-line option, ``--no-maximized``, that prevents glue
  from opening up with the application window maximized. [#1093, #1126]

* When opening multiple files in one go, if one of the files fails to
  read, the error will now indicate which file failed. [#1104]

* Fixed a bug that caused new subset colors to incorrectly start from the start
  of the color cycle after loading a session. [#1055]

* Fixed a bug that caused the functionality to execute scripts (glue -x) to not
  work in Python 3. [#1101, #1114]

* The minimum supported version of Astropy is now 1.0, and the minimum
  supported version of IPython is now 1.0. [#1076]

* Show world coordinates and units in the cube slicer. [#1059, #1068]

* Fix errors that occurred when selecting categorical data. [#1069]

* Added experimental support for joining on multiple keys in ``join_on_key``. [#974]

* Fix compatibility with the latest version of ginga. [#1063]

v0.8.2 (2016-07-06)
-------------------

* Implement missing MaskSubsetState.copy. [#1033]

* Ensure that failing data factory identifier functions are skipped. [#1029]

* The naming of pixel axes is now more consistent between data with 3 or
  fewer dimensions, and data with more than 3 dimensions. The naming is now
  always ``Pixel Axis ?`` where ``?`` is the index of the array, and for
  datasets with 1 to 3 dimensions, we add a suffix e.g. ``[x]`` to indicate
  the traditional axes. [#1029]

* Implemented a number of performance improvements, including for: the check
  of whether points are in polygon (``points_inside_poly``), the selection of
  polygonal regions in multi-dimentional cubes when the selections are along
  pixel axes, the selection of points in scatter plots with one or two
  categorical components for rectangular, circular, and polygonal regions.
  [#1029]

* Fix a bug that caused multiple custom viewer classes to not work properly
  if the user did not override ``_custom_functions`` (which was private).
  [#810]

* Make sure histograms are updated if only the attribute changes and the
  limits and number of bins stay the same. [#1012]

* Fix a bug on Windows that caused drag and dropping files onto the glue
  application to not work. [#1007]

* Fix compatibility with PyQt5. [#1015]

* Fix a bug that caused ComponentIDComboHelper to not take into account the
  numeric and categorical options in __init__. [#1014]

* Fix a bug that caused saving of scatter plots to SVG files to crash. [#984]

v0.8.1 (2016-05-25)
-------------------

* Fixed a bug in the memoize function that caused selections using
  ElementSubsetState to fail when using views on the data. [#1004]

* Explicitly set the icon size for the slicing playback controls to avoid
  issues when using a mix of retina and non-retina displays. [#1005]

* Fixed a bug that caused add_datasets to crash if ``datasets`` was a list of
  lists of data, which is possible if a data factory returns more than one data
  object. [#1006]

v0.8 (2016-05-23)
-----------------

* Add support for circular and polygonal spectrum extraction. [#994, #1003]

* Fix compatibility with latest developer version of Numpy which does not allow
  non-integer indices for arrays. [#1002]

* Add a new method ``add_data`` to application instances. This allows for
  example additional data to be passed to glue after being launched by
  ``qglue``. [#993]

* Add playback controls to slice widget. [#971]

* Add tooltip for data labels so that long labels can be more easily
  inspected. [#912]

* Added a new helper class ``AttributeLimitsHelper`` to link widgets related to
  setting limits and handle the caching of the limits as a function of
  attribute. [#872]

* Add Quit menu item for Linux and Windows. [#926]

* Refactored the window for sending feedback to include more version
  information, and also to have a separate form for feedback and crash
  reports. [#955]

* Add log= option to ValueProperty and remove mapping= option. [#965]

* Added helper classes for ComponentID and Data combo boxes. [#891]

* Improved new component window: expressions can now include math or numpy
  functions by default, and expressions are tested on-the-fly to check that
  there are no issues with syntax or undefined variables. [#956]

* Fixed D3PO export when using Python 3. [#989]

* Fixed display of certain error messages when using Python 3. [#989]

* Add an extensible preferences window. [#988]

* Add the ability to change the foreground and background color for viewers.
  [#988]

* Fixed a bug that caused images to appear over-pixellated on the edges when
  zooming in. [#1000]

* Added an option to control whether warnings are shown when passing large data objects to viewers. [#999]

v0.7.3 (2016-05-04)
-------------------

* Remove icons for actions that appear in contextual menus, since these
  appear too large due to a Qt bug. [#911]

* Add missing find_spec for import hook, to avoid issues when trying to set
  colormap. [#930]

* Ignore extra dimensions in WCS (for instance, if the data is 3D and the
  header is 4D, ignore the 4th dimension in the WCS). [#935]

* Fix a bug that caused the merge window to appear multiple times, make sure
  that all components named PRIMARY get renamed after merging, and make sure
  that the merge mechanism is also triggered when opening datasets from the
  command-line. [#936]

* Remove the scrollbars added in v0.7.1 since they cause issues on certain
  systems. [#953]

* Fix saving of ElementSubsetState to session files. [#966]

* Fix saving of Matplotlib colormaps to session files. [#967]

* Fix the selection of the default viewer based on the data shape. [#968]

* Make sure that no combo boxes get resized based on the content (unless
  strictly needed). [#978]

v0.7.2 (2016-04-05)
-------------------

* Fix a bug that caused string columns in FITS files to not be read
  correctly, and updated coerce_numeric to give a ValueError for string
  columns that can't be convered. [#919]

* Make sure main window title is set. [#914]

* Fix issue with FITS files that are missing an END card. [#915]

* Fix a bug that caused values in exponential notation in text fields to lose
  a trailing zero (e.g. 1.000e+10 would become 1.000e+1). [#925]

v0.7.1 (2016-03-30)
-------------------

* Fix issue with small screens and layer and viewer options by adding
  scrollbars. [#902]

* Fixed a failure due to a missing Qt import in glue.core.roi. [#901]

* Fixed a bug that caused an abort trap if the filename specified on the
  command line did not exist. [#903]

* Gracefully skip vector columnns when reading in FITS files. [#896]

* Change default gray color to work on black and white backgrounds. [#906]

* Fixed a bug that caused the color in the scatter and histogram style
  editors to not show the correct initial color. [#907]

v0.7 (2016-03-10)
-----------------

* Python 2.6 is no longer supported. [#804]

* Added a generic QColorBox widget to pick colors, and an associated
  connect_color helper for callback properties. [#864]

* Added a generic QColormapCombo widget to pick colormaps.

* The ``artist_container`` argument to client classes has been renamed to
  ``layer_artist_container``. [#814]

* Added documentation about how to use layer artists in custom Qt data viewers.
  [#814]

* Fixed missing newline in Data.__str__. [#877]

* A large fraction of the code has been re-organized, which may lead to some
  imports in ``config.py`` files no longer working. However, no functionality
  has been removed, so this can be fixed by updating the imports to reflect the
  new locations.

  In particular, the following utilities have been moved:

  ``glue.qt.widget_properties``                 | ``glue.utils.qt.widget_properties``
  ``glue.qt.decorators``                        | ``glue.utils.qt.decorators``
  ``glue.qt.qtutil.mpl_to_qt4_color``           | ``glue.utils.qt.colors.mpl_to_qt4_color``
  ``glue.qt.qtutil.qt4_to_mpl_color``           | ``glue.utils.qt.colors.qt4_to_mpl_color``
  ``glue.qt.qtutil.pick_item``                  | ``glue.utils.qt.dialogs.pick_item``
  ``glue.qt.qtutil.pick_class``                 | ``glue.utils.qt.dialogs.pick_class``
  ``glue.qt.qtutil.get_text``                   | ``glue.utils.qt.dialogs.get_text``
  ``glue.qt.qtutil.tint_pixmap``                | ``glue.utils.qt.colors.tint_pixmap``
  ``glue.qt.qtutil.cmap2pixmap``                | ``glue.utils.qt.colors.cmap2pixmap``
  ``glue.qt.qtutil.pretty_number``              | ``glue.utils.qt.PropertySetMixin``
  ``glue.qt.qtutil.Worker``                     | ``glue.utils.qt.threading.Worker``
  ``glue.qt.qtutil.update_combobox``            | ``glue.utils.qt.helpers.update_combobox``
  ``glue.qt.qtutil.PythonListModel``            | ``glue.utils.qt.python_list_model.PythonListModel``
  ``glue.clients.tests.util.renderless_figure`` | ``glue.utils.matplotlib.renderless_figure``
  ``glue.core.util.CallbackMixin``              | ``glue.utils.misc.CallbackMixin``
  ``glue.core.util.Pointer``                    | ``glue.utils.misc.Pointer``
  ``glue.core.util.PropertySetMixin``           | ``glue.utils.misc.PropertySetMixin``
  ``glue.core.util.defer``                      | ``glue.utils.misc.defer``
  ``glue.qt.mime.PyMimeData``                   | ``glue.utils.qt.mime.PyMimeData``
  ``glue.qt.qtutil.GlueItemWidget``             | ``glue.utils.qt.mixins.GlueItemWidget``
  ``glue.qt.qtutil.cache_axes``                 | ``glue.utils.matplotlib.cache_axes``
  ``glue.qt.qtutil.GlueTabBar``                 | ``glue.utils.qt.helpers.GlueTabBar``

  [#827, #828, #829, #830, #831]

  ``glue.clients.histogram_client``                  | ``glue.viewers.histogram.client``
  ``glue.clients.image_client``                      | ``glue.viewers.image.client``
  ``glue.clients.scatter_client``                    | ``glue.viewers.scatter.client``
  ``glue.clients.layer_artist.LayerArtist``          | ``glue.clients.layer_artist.MatplotlibLayerArtist``
  ``glue.clients.layer_artist.ChangedTrigger``       | ``glue.clients.layer_artist.ChangedTrigger``
  ``glue.clients.layer_artist.LayerArtistContainer`` | ``glue.clients.layer_artist.LayerArtistContainer``
  ``glue.clients.ds9norm``                           | ``glue.viewers.image.ds9norm``
  ``glue.clients.profile_viewer``                    | ``glue.plugins.tools.spectrum_viewer.profile_viewer``
  ``glue.clients.util.small_view``                   | ``glue.core.util.small_view``
  ``glue.clients.util.small_view_array``             | ``glue.core.util.small_view_array``
  ``glue.clients.util.visible_limits``               | ``glue.core.util.visible_limits``
  ``glue.clients.util.tick_linker``                  | ``glue.core.util.tick_linker``
  ``glue.clients.util.update_ticks``                 | ``glue.core.util.update_ticks``
  ``glue.qt.widgets.histogram_widget``               | ``glue.viewers.histogram.qt``
  ``glue.qt.widgets.scatter_widget``                 | ``glue.viewers.scatter.qt``
  ``glue.qt.widgets.histogram_widget``               | ``glue.viewers.image.qt``
  ``glue.qt.widgets.table_widget``                   | ``glue.viewers.table.qt``
  ``glue.qt.widgets.data_viewer``                    | ``glue.viewers.common.qt.data_viewer``
  ``glue.qt.widgets.mpl_widget``                     | ``glue.viewers.common.qt.mpl_widget``
  ``glue.qt.widgets.MplWidget``                      | ``glue.viewers.common.qt.mpl_widget.MplWidget``
  ``glue.qt.glue_toolbar``                           | ``glue.viewers.common.qt.toolbar``
  ``glue.qt.custom_viewer``                          | ``glue.viewers.custom.qt``

  [#835]

  ``glue.qt.glue_application.GlueApplication``       | ``glue.app.qt.application.GlueApplication``
  ``glue.qt.plugin_manager.QtPluginManager``         | ``glue.app.qt.plugin_manager.QtPluginManager``
  ``glue.qt.feedback.FeedbackWidget``                | ``glue.app.qt.feedback.FeedbackWidget``
  ``glue.qt.widgets.glue_mdi_area``                  | ``glue.app.qt.mdi_area``

  ``glue.qt.widgets.terminal``                       | ``glue.app.qt.terminal``
  ``glue.qt.qt_roi``                                 | ``glue.core.qt.roi``
  ``glue.core.qt.simpleforms``                       | ``glue.core.qt.simpleforms``
  ``glue.qt.widgets.style_dialog``                   | ``glue.core.qt.style_dialog``
  ``glue.qt.layer_artist_model``                     | ``glue.core.qt.layer_artist_model``
  ``glue.qt.widgets.custom_component_widget``        | ``glue.dialogs.custom_component.qt``
  ``glue.qt.link_editor``                            | ``glue.dialogs.link_editor.qt``
  ``glue.qt.widgets.subset_facet``                   | ``glue.dialogs.subset_facet.qt``
  ``glue.qt.mouse_mode``                             | ``glue.viewers.common.qt.mouse_mode``
  ``glue.qt.data_slice_widget``                      | ``glue.viewers.common.qt.data_slice_widget``
  ``glue.qt.widgets.layer_tree_widget``              | ``glue.app.qt.layer_tree_widget``
  ``glue.qt.widgets.message_widget``                 | ``glue.core.qt.message_widget``
  ``glue.qt.widgets.settings_editor``                | ``glue.app.qt.settings_editor``
  ``glue.qt.qtutil.data_wizard``                     | ``glue.dialogs.data_wizard.qt.data_wizard``
  ``glue.qt.mime``                                   | ``glue.core.qt.mime``
  ``glue.qt.qtutil.ComponentIDCombo``                | ``glue.core.qt.component_id_combo``
  ``glue.qt.qtutil.RGBEdit``                         | ``glue.viewers.image.qt.rgb_edit.RGBEdit``
  ``glue.qt.qtutil.GlueListWidget``                  | ``glue.core.qt.mime.GlueMimeListWidget``
  ``glue.qt.qtutil.load_ui``                         | ``glue.utils.qt.helpers.load_ui``

  ``glue.qt.qtutil.icon_path``                       | ``glue.icons.icon_path``
  ``glue.qt.qtutil.load_icon``                       | ``glue.icons.qt.load_icon``
  ``glue.qt.qtutil.symbol_icon``                     | ``glue.icons.qt.symbol_icon``
  ``glue.qt.qtutil.layer_icon``                      | ``glue.icons.qt.layer_icon``
  ``glue.qt.qtutil.layer_artist_icon``               | ``glue.icons.qt.layer_artist_icon``
  ``glue.qt.qtutil.GlueActionButton``                | ``glue.app.qt.actions.GlueActionButton``
  ``glue.qt.qtutil.action``                          | ``glue.app.qt.actions.action``
  ``glue.qt.qt_backend.Timer``                       | ``glue.backends.QtTimer``

  [#845]

* Improved under-the-hood creation of ROIs for Scatter and Histogram Clients. [#676]

* Data viewers can now define a layer artist style editor class that appears
  under the list of layer artists. [#852]

* Properties of the VisualAttributes class are now callback properties. [#852]

* Add ``glue.utils.qt.widget_properties.connect_value`` function which can take
  an optional value_range and log option to scale the Qt values to a custom
  range of values (optionally in log space). [#852]

* Make list of data viewers sorted alphabetically. [#866]

v0.6 (2015-11-20)
-----------------

* Added experimental support for PyQt5. [#663]

* Fix ``glue -t`` option. [#791]

* Updated ``glue-deps`` to show PyQt/PySide versions. [#796]

* Fix bug that caused viewers to be restored with the wrong size. [#781, #783]

* Fixed compatibility with the latest stable version of ginga. [#797]

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
