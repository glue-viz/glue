Full changelog
==============

v0.8 (unreleased)
-----------------

* Added a new helper class ``AttributeLimitsHelper`` to link widgets related to
  setting limits and handle the caching of the limits as a function of
  attribute. [#872]

v0.7.1 (2016-03-29)
-------------------

* Fix issue with small screens and layer and viewer options by adding
  scrollbars. [#902]

* Fixed a failure due to a missing Qt import in glue.core.roi. [#901]

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
