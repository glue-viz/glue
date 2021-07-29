Full changelog
==============

v1.2.0 (unreleased)
-------------------

* Allow CategoricalComponents to be n-dimensional. [#2214]

v1.1.0 (2021-07-21)
-------------------

* Fix compatibility with xlrd 2.0 and require openpyxl for .xlsx input. [#2196]

* Dropped support for Python 3.6. [#2196]

* Fixed compatibility with recent releases of Matplotlib. [#2196]

* Avoid warnings with recent releases of astropy. [#2212]

* Fixed compatibility of auto-linking with all APE 14 WCSes. [#2209]

v1.0.1 (2020-11-10)
-------------------

* Updated supported versions of jupyter_client. [#2175]

v1.0.0 (2020-09-17)
-------------------

* Remove bundled echo package and list as a dependency. [#2125]

* Add the ability to export Python scripts for the profile viewer. [#2082]

* Add support for polar and other non-rectilinear projections in
  2-d scatter viewer [#2170]

* Add legend for matplotlib viewers (in qt and in export scripts) [#2097, #2144, #2146]

* Add new registry to apply in-place patches to the session file [#2127]

* Initial support for dask arrays. [#2137, #2149]

* Don't sync color and transparency of image layers by default. [#2116]

* Python 2.7 and 3.5 are no longer supported. [#2075]

* Always use replace mode when creating a new subset. [#2090]

* Fixed bugs in the profile viewer that occurred when using the profile viewer and
  setting reference_data to a different value than the default one. [#2078]

* Updated the data loader to be able to select directories. [#2077,#2080]

* Move spectral-cube data loader to glue-astronomy plugin package. [#2077]

* Show session filename path in window title. [#2096]

* Change ``Data.coords`` API to now rely on the API described in
  https://github.com/astropy/astropy-APEs/blob/master/APE14.rst [#2079]

* Update minimum required version of Astropy to 4.0. [#2079]

* Added a ``DataCollection.clear()`` method to remove all datasets. [#2079]

* Fixed a bug that caused profiles of subsets to not be hidden if an
  existing subset was emptied. [#2095]

* Improved ``IndexedData`` so that world coordinates can now be shown. [#2081]

* Make loading Qt plugins optional when calling ``load_plugins``. [#2112]

* Preserve ``RectangularROI`` rather than converting them to ``PolygonalROI``. [#2112]

* Significantly improve performance of ``compute_fixed_resolution_buffer`` when
  using linked datasets with some axes uncorrelated. [#2115]

* Improve performance of profile viewer when not all layers are visible. [#2115]

* Fixed missing .units on ``CoordinateComponent``. [#2117]

* Improve auto-linking of astronomical datasets with WCS information. [#2161]

* Fix bug that occurred when using the GUI link editor when links had
  been defined programmatically. [#2166]

* Add the ability to programmatically set the preferred colormap for a
  dataset using ``Data.visual.preferred_cmap`` [#2131, #2168]

* Fix bug that caused incorrect unit to be shown on slider for images. [#2159]

* Improve performance of ``Data.compute_statistic`` for subsets. [#2147]

* Fix a bug where x_att and y_att could end up being out of sync in image viewer. [#2141]

v0.15.7 (2020-03-12)
--------------------

* Fix bug that caused an infinite loop in the table viewer and caused glue to
  hang if too many incompatible subsets were in a table viewer. [#2105]

* Fixed a bug that caused an error when an invalid data was added to the table
  viewer and the table viewer was then automatically closed. [#2103]

* Fixed the dropdowns for vector and error markers to not include datetime
  components (since they represent absolute times, not deltas). [#2102]

* Fixed a bug that caused session files that were saved with LinkCollection to
  not work correctly when re-loaded. [#2100]

* Fixed a bug that caused issues when saving and re-loading sessions that were
  originally created using Excel data with string columns. [#2101]

* Avoid converting circular selections in Matplotlib plots to polygons if
  it can be avoided. [#2094]

v0.15.6 (2019-08-22)
--------------------

* Fixed bugs related to auto-linking of astronomical data with WCS information. In
  particular, links between datasets with celestial coordinates in a different order
  or links between some higher dimensional datasets with celestial axes did not
  always work correctly. [#2052]

* Fixed a bug that caused the auto-linking framework to not run when opening glue from
  the ``qglue()`` function. [#2052]

* Fixed a limitation that caused pixel selections to not propagate to other datasets.
  [#2052]

* Fixed a deprecation warning related to ``add_datasets``. [#2052]

* Fixed compatibility with Python 3.7.4. [#2060]

* Fixed performance issue with arrays that were not in native C order. [#2056]

v0.15.5 (2019-07-09)
--------------------

* Fixed bug with density map visibility when using color-coding. [#2041]

* Fixed bug with incompatible subsets and density maps. [#2041]

* Make sure that lines/errors/vectors are disabled when in density map mode. [#2041]

v0.15.4 (2019-07-08)
--------------------

* Fixed bug that occurred when trying to add a subset to a new table
  viewer (without the parent data). [#2038]

v0.15.3 (2019-06-27)
--------------------

* Fixed bugs related to the preferences dialog - first, the dialog would
  not open if no auto-linkers were present, and second, the preferences
  dialog would sometimes get sent behind the main application when editing
  the color. [#2034]

v0.15.2 (2019-06-24)
--------------------

* Fixed a bug in ``autoconnect_callbacks_to_qt`` which caused some widgets
  to not stay connect to state callback properties if a callback property
  was linked to multiple widgets. [#2032]

v0.15.1 (2019-06-24)
--------------------

* Fixed __version__ variable. [#2031]

* Fixed tox configuration. [#2031]

* Improve error message if loading a session file that uses WCS auto-linking
  but the installed version of astropy is too old. [#2031]

v0.15.0 (2019-06-23)
--------------------

* Added a new ``glue.core.coordinates.AffineCoordinates`` class for common
  affine transformations, and also added documentation on defining custom
  coordinates. [#1994]

* Rewrote ``SubsetFacet`` (now ``SubsetFacetDialog``), and updated the
  available colormaps. [#1998]

* Removed the ``ComponentSelector`` class. [#1998]

* Make it possible to view only a subset of data in the table
  viewer. [#1988]

* Show dataset name in table viewer title. [#1973]

* Expose an option (``inherit_tools``) on data viewer classes
  related to whether tools should be inherited or not from
  parent classes. [#1972]

* Added a new method ``compute_fixed_resolution_buffer`` to data
  objects (including the base data class) and use this for the
  image viewer. This improves the case where images are reprojected
  as they are now all reprojected to the screen resolution rather
  than the resolution of the reference data, and this also opens
  up the possibility of doing n-dimensional reprojection. [#1895]

* Added initial infrastructure for developing auto-linking helpers
  and implement an initial astronomy WCS auto-linker. [#1933]

* Improve the user interface for the link editor. [#1934, #1998]

* Added a new ``IndexedData`` class to represent a derived dataset produced
  by indexing a higher-dimensional dataset. [#1953]

* Removed plot.ly export plugin - this is now part of the glue-plotly
  package. [#1999]

* Fix path to data bundle when exporting Python scripts to another directory
  than the one glue was run from. [#2023]

* Added a ``--faulthandler`` command-line flag to help debug segmentation
  faults. [#1974]

* Removed ``glue.core.qt.roi`` submodule. This provided faster versions of
  the Matplotlib ROI classes in ``glue.core.roi`` but the latter are now
  efficient enough that the Qt-specific versions are no longer needed. [#1983]

* Moved ``glue.viewers.common.qt.tool`` to ``glue.viewers.common.tool``;
  ``glue.viewers.common.qt.mouse_mode`` to ``glue.viewers.matplotlib.mouse_mode``;
  and ``glue.viewers.common.qt.toolbar_mode`` to ``glue.viewers.matplotlib.toolbar_mode``
  and ``glue.viewers.matplotlib.qt.toolbar_mode``. [#1984]

* Implement a human-readable str/repr for State objects. [#2021]

* Avoiding importing Qt when using the base histogram and profile layer artists.
  [#2012]

* Fixed a bug that caused error bars to be colored incorrectly if NaN values
  were present. [#2020]

* Fixed compatibility with the latest versions of Matplotlib and Astropy. [#2020]

* No longer suggest merging data by default whenever datasets are added to
  the session. Merging different datasets should now be done manually through
  the GUI. [#2020]

* Fixed compatibility with Numpy 1.16.x. [#1989]

* Improve tab-completion of attribute names in Data to not include
  non-relevant items. [#1971]

* Fix an error that occurred if creating a new viewer was
  cancelled. [#1952]

* Fix error in image viewer when using update_values_from_data. [#1975]

* Exclude problematic versions of ipykernel from dependencies. [#1952]

* Make sure that scrolling above a viewer does not result in the
  whole canvas also scrolling. [#1919]

* Fix bug that caused date/time columns in Excel files to not be
  read in correctly.

* Improve performance when reading in large non-FITS files. [#1920]

* Fix bug that caused log parameter to be ignore for density plots. [#1963]

* Fix bug that caused an error when using the profile collapse tool and
  dragging from right to left. [#2002]

* Fix bug that caused labels on the x-axis of the histogram viewer to
  be incorrectly set to numbers instead of strings when showing a
  histogram of a string variable and saving/reloading session. [#2009]

v0.14.2 (2019-02-04)
--------------------

* Fix bug that caused demo VO Table to not be read in correctly with
  recent versions of Numpy and Astropy. [#1911]

v0.14.1 (2018-11-23)
--------------------

* Fix bug that caused the links based on ``join_on_key`` to not
  always work. [#1902]

v0.14.0 (2018-11-14)
--------------------

* Improved how we handle equal aspect ratio to not depend on
  Matplotlib. [#1894]

* Avoid showing a warning when closing an empty tab. [#1890]

* Fix bug that caused component arithmetic to not work if
  Numpy was imported in user's config.py file. [#1887]

* Added the ability to define custom layer artist makers to
  override default layer artists in viewers. [#1850]

* Fix Plot.ly exporter for categorical components and histogram
  viewer. [#1886]

* Fix issues with reading very large FITS files on some systems. [#1884]

* Added documentation about plugins. [#1837]

* Better isolate code related to pixel selection tool in image
  viewer that depended on Qt. [#1763]

* Improve handling of units in FITS files. [#1723]

* Added documentation about creating viewers for glue using the
  new state-based infrastructure. [#1740]

* Make it possible to pass the initial state of a viewer to an
  application's ``new_data_viewer`` method. [#1877]

* Ensure that glue can be imported if QtPy is installed but PyQt
  and PySide aren't. [#1865, #1836]

* Fix unit display for coordinates from WCS headers that don't have
  CTYPE but have CUNIT. [#1856]

* Enable tab completion on Data objects. [#1874]

* Automatically select datasets in link editor if there are only two. [#1837]

* Change 'Export Session' dialog to offer to save with relative paths to data
  by default instead of absolute paths. [#1803]

* Added a new method ``screenshot`` on ``GlueApplication`` to save a
  screenshot of the current view. [#1808]

* Show the active subset in the toolbar. [#1797]

* Refactored the viewer class base classes [#1746]:

  - ``glue.core.application_base.ViewerBase`` has been removed in favor of
    ``glue.viewers.common.viewer.BaseViewer`` and
    ``glue.viewers.common.viewer.Viewer``.

  - ``glue.viewers.common.viewer.Viewer`` is now where the base logic is defined
    for using state classes in viewers (instead of
    ``glue.viewers.common.qt.DataViewerWithState``).

  - ``glue.viewers.common.qt.DataViewerWithState`` is now deprecated.

* Make it so that the modest image only resamples the data when the
  mouse is no longer pressed - this avoids too many refreshes when
  panning/zooming. [#1866]

* Make it possible to unglue multiple links in one go. [#1809]

* Make it so that adding a subset to a viewer no longer adds the
  associated data, since in some cases the viewer can handle the
  subset size, but not the full data. [#1807]

* Defined a new abstract base class for all datasets, ``BaseData``,
  and a base class ``BaseCartesianData``,
  which can be used to implement interfaces to datasets that may be
  remote or may not be stored as regular cartesian data. [#1768]

* Add a new method ``Data.compute_statistic`` which can be used
  to find scalar and array statistics on the data, and use for
  the profile viewer and the state limits helpers. [#1737]

* Add a new method ``Data.compute_histogram`` which can be used
  to find histograms of specific components, with or without
  subsets applied. [#1739]

* Removed ``Data.get_pixel_component_ids`` and ``Data.get_world_component_ids``
  in favor of ``Data.pixel_component_ids`` and ``Data.world_component_ids``.
  [#1784]

* Deprecated ``Data.visible_components`` and ``Data.primary_components``. [#1788]

* Speed up histogram calculations by using the fast-histogram package instead of
  np.histogram. [#1806]

* In the case of categorical attributes, ``Data[name]`` now returns a
  ``categorical_ndarray`` object rather than the indices of the categories. You
  can access the indices with ``Data[name].codes`` and the unique categories
  with ``Data[name].categories``.  [#1784]

* Compute profiles and histograms asynchronously when dataset is large
  to avoid holding up the UI, and compute profiles in chunks to avoid
  excessive memory usage. [#1736, #1764]

* Improved naming of components when merging datasets. [#1249]

* Fixed an issue that caused residual references to viewers
  after they were closed if they were accessed through the
  IPython console. [#1770]

* Don't show layer edit options if layer is not visible. [#1805]

* Make the Matplotlib viewer code that doesn't depend on Qt accessible
  to non-Qt frontends. [#1841]

* Avoid repeated coordinate components in merged datasets. [#1792]

* Fix bug that caused new subset to be created when dragging an existing
  subset in an image viewer. [#1793]

* Better preserve data types when exporting data/subsets to FITS
  and HDF5 formats. [#1800]

v0.13.4 (2018-10-19)
--------------------

* Fix bug that caused .svg icons to not be correctly installed. [#1882]

* Fix bug that occurred in certain cases when using the state attribute limit
  helper with a state class that did not have a log attribute. [#1842]

* Fix HDF5 reader for string columns. [#1840]

* Fix visual bug in link editor in advanced mode when resizing window.

* Fixed a bug that caused custom data importers to no longer work. [#1813]

* Fixed a bug that caused ROIs to not be erased after selection if the
  active subset was not in the list of layers for the viewer. [#1801]

* Always returned to last used folder when opening/saving files. [#1794]

* Show correct dataset when using control-click to select to add
  arithmetic attributes or rename/reorder components. [#1802]

* Improve performance when updating links and changing attributes
  on subsets. [#1716]

* Fix errors that happened when clicking on the 'Export Data' and
  'Define arithmetic attributes' buttons when no data was present,
  and fixed Qt errors that happened if the data collection changed
  after the 'Export Data' dialog was opened. [#1795]

* Fixed parsing of AVM meta-data from images. [#1732]

* Fixed compatibility with Matplotlib 3.0. [#1875]

v0.13.3 (2018-05-08)
--------------------

* Fixed a bug that caused the expression for derived attributes to
  not immediately be updated in the list of derived attributes. [#1708]

* Fixed a bug that caused combo boxes with Data objects to not update
  when a data label was changed. [#1704]

* Fixed a bug related to callback functions when restoring sessions.
  [#1695]

* Fixed a bug that meant that setting Data.coords after adding
  components didn't work as expected. [#1196]

* Fixed bugs related to components with only NaN or Inf values.
  [#1712]

* Fixed a bug that caused an error when the zoom or pan tools were
  active and the viewer was closed. [#1712]

* Fixed a Qt-related segmentation fault that occurred during the
  testing process and may also affect users. [#1703]

* Show image layer attribute in list of layers. [#1706]

* Fixed a bug that caused scatter plots to not revert to fixed color
  mode after being in linear color mode. [#1705]

v0.13.2 (2018-05-01)
--------------------

* Fixed a bug that caused the EditSubsetMode toolbar to not change
  when EditSubsetMode.mode was changed programatically. [#1684]

* Fixed unintuitive behavior of single-pixel selection tool - now
  moving the crosshairs requires clicking and dragging. [#1684]

* Fixed bug that caused crosshairs to not be hidden when a layer was
  set to not be visible [#1684]

* Fixed a bug that caused viewers to be closed without warning when
  pressing delete. [#1684]

v0.13.1 (2018-04-29)
--------------------

* Fixed resetting and opening of sessions which caused Glue to quit. [#1681]

* Fixed serialization of Data.meta when non-serializable keys or values
  are present. [#1681]

v0.13.0 (2018-04-27)
--------------------

* Added new perceptually uniform Matplotlib colormaps. [#1679]

* Fixed a bug that caused vectors to not correctly be flipped when
  flipping the x/y limits of the plot. [#1677]

* Added a CSV and HDF5 data/subset exporter. [#1676]

* Started adding helpful information dialogs that can be
  hidden. [#1669]

* Make it possible to have a 'None' entry in the ComponentIDComboHelper. [#1661]

* Added a new metadata/header viewer for datasets/subsets. [#1659]

* Re-write spectrum viewer into a generic profile viewer that uses
  subsets to define the areas in which to compute profiles rather
  than custom ROIs. [#1635]

* Added support for PySide2 and remove support for PyQt4 and
  PySide. [#1662]

* Remove support for Matplotlib 1.5. [#1662]

* Renamed ``qt4_to_mpl_color`` to ``qt_to_mpl_color`` and
  ``mpl_to_qt4_color`` to ``mpl_to_qt_color``. [#1662]

* Improve performance when changing visual attributes of subsets.
  [#1617]

* Removed ``glue.core.qt.data_combo_helper`` (we now recommend using
  the GUI framework-independent equivalent in
  ``glue.core.data_combo_helper``). [#1625]

* Removed ``glue.viewers.common.qt.attribute_limits_helper`` in favor
  of ``glue.core.state_objects``. [#1625]

* Removed unused function ``glue.utils.misc.defer``. [#1625]

* Added a new ``FloodFillSubsetState`` class to represent and
  calculate subsets made by a flood-fill algorithm. [#1616]

* Added the ability to easily create viewer tools with dropdown
  menus. [#1634]

* Remove the ``MatplotlibViewerToolbar`` class as it is now no
  longer needed - instead you can just list the matplotlib tools
  directly in the ``tools`` attribute of the viewer. [#1634]

* Improve hiding/showing of side-panels. No longer hide side-panels
  when glue application goes out of focus. [#1535]

* Use memory-mapping with contiguous arrays in HDF5 files, resulting
  in improved performance for large files. [#1628]

* Deselect tools when the viewer focus changes. [#1584, #1608]

* Added support for whether symbols are shown filled or not. [#1559]

* Improved link editor to include a graph of links. [#1534]

* Improve mouse interaction with ROIs in image viewers, including
  click-and-drag relocation. Allow for more customization of mouse/toolbar
  modes. [#1515]

* Add a toolbar item to save data. [#1516, #1519, #1575]

* Give instructions for how to move selections in status tip. [#1504]

* Improve the display of data cube slice labels to include only the
  precision required given the separation of world coordinate values.
  [#1500, #1660]

* Removed the ability to edit the marker symbol in the style dialog
  since this isn't recognized by any viewer anymore. [#1560]

* Remove back/forward tools in Matplotlib viewer toolbars to
  declutter. [#1505]

* Added a new component manager that makes it possible to rename,
  reorder, and remove components, as well as better manage derived
  components, including editing previous equations. [#1479]

* Added new messages ``DataReorderComponentMessage`` and
  ``DataRenameComponentMessage`` which can be subscribed to. [#1479]

* Add support for the datetime64 dtype in Data objects, and adjust
  Matplotlib viewers to correctly show this data. [#1510]

* Make it possible to reorder components in ``Data`` using the new
  ``Data.reorder_components`` method. [#1479]

* The default order of components has changed - coordinate components
  will now always come first (rather than second). [#1479]

* Added support for scatter density maps, which is useful when making
  scatter plots of many points. [#1461]

* Improve how ComponentIDComboHelper deals with non-primary components.
  The .visible property has been removed, and a new .derived property
  has been added (to show/hide derived components). Components are now
  split up into sections in the combo boxes. [#1476]

* Fixed a bug that caused ghost components to be added when creating a
  derived component with data[...] = ... [#1476]

* Fixed a bug that caused errors when removing items from a selection
  property linked to a QComboBox. [#1476]

* Added initial support for customizing keyboard shortcuts. [#1475, #1514, #1524]

* Added support for using relative paths in session files. [#1537]

* Remember last session filename and filter used. [#1537]

* EditSubsetMode is now no longer a singleton class and is
  instead instantiated at the Application/Session level. [#1530]

* Improve performance of image viewer. [#1558]

* Added new ``Projected3dROI`` and ``RoiSubsetState3d`` classes
  to represent 3D selections made in the projection plane. [#1522]

* Fixed saving of sessions with ``BinaryComponentLink``. [#1533]

* Refactored/simplified handling of links between datasets and
  fixed performance issues when adding/removing links or loading
  data collections with many links. [#1531, #1533]

* Significantly improve performance of link computations when the
  links depend only on pixel or world coordinate components. [#1585]

* Added the ability to customize the appearance of tick and axis
  labels in Matplotlib plots. [#1511]

* Added the ability to export Python scripts from the main
  Matplotlib-based viewers. [#1511]

* Added a new selection mode that always forces the creation of a new subset.
  [#1525]

* Added a mouse over pixel selection tool, which creates a one pixel subset
  under the mouse cursor. [#1619]

* Fixed an issue that caused sliders to not be correctly updated when
  switching reference data in the image viewer. [#1665]

* Fixed a bug that caused Data.meta to not be saved/restored from session
  files. [#1668]

* Fixed an issue that caused an IndexError when quitting glue in some
  cases. [#1657]

* Fixed a bug that caused matplotlibrc files to not be ignored. [#1649]

* Fixed a non-deterministic error that happened when closing the
  TableViewer. [#7310]

* Fixed size of markers when value for size is out of vmin/vmax range. [#1609]

* Fix a bug that caused viewer limits to be calculated incorrectly if
  inf/-inf values were present in the data. [#1614]

* Fixed a bug which caused the y-axis in the PV slice viewer to be
  incorrect if the WCS could not be computed. [#1615]

* Fixed a bug that caused the WCS of a PV slice to be incorrect if the
  user has selected a custom order of the axes of a cube in the image
  viewer. [#1615]

* Fixed ticks on log x-axis in histogram viewer. [#7310]

* Fixed a bug that led to poor performance when slicing through data cubes.
  [#1554]

v0.12.5 (2018-03-10)
--------------------

* Fixed a bug which caused the current slices to be lost when adding a second
  dataset to the image viewer. [#1581]

* Fixed a bug when two datasets with a different number of dimensions
  were in an image viewer and a subset was created. [#1577]

* Fixed issues when attempting to close a viewer with the delete key. [#1574]

* Disabled default Matplotlib key bindings. [#1574]

* Fix compatibility with Matplotlib 2.2. [#1566]

* Fix compatibility with some versions of pytest. [#1520]

* Fix calculation of dependent_axes to account for cases where there
  are some non-zero non-diagonal PC values. Previously any such values
  resulted in all axes being returned as dependent axes even though this
  isn't necessary. [#1552]

* Avoid prompting users multiple times to merge data when dragging
  and dropping multiple data files onto glue. [#1564]

* Improve error message in PV slicer when _slice_index fails. [#1536]

* Fixed a bug that caused an error when trying to save a session that
  included an image viewer with an aggregated slice. [#1561]

* Fixed a bug that caused an error in the terminal if creating a data
  viewer failed properly (with a GUI error message). [#1501]

* Fixed a bug that caused performance issues when hiding all image
  layers from an image viewer. [#1557, #1562]

* Fixed a bug that caused layers to not always be properly removed
  when deleting a row from the layer list. [#1502]

* Make JSON circular reference errors more explicit. [#1529]

v0.12.4 (2018-01-09)
--------------------

* Improve plugin loading to be less sensitive to exact versions of
  installed dependencies for plugins. [#1487]

v0.12.3 (2017-11-14)
--------------------

* Fixed issues with PV slicer and spectrum viewer when changing axes
  in the parent image viewer.

v0.12.2 (2017-11-09)
--------------------

* Fix a bug when renaming tabs through the UI. [#1470]

* Fix a bug that caused the 1D and 2D viewers to not update correctly
  when the numerical values in data were changed. [#1471]

* Fix a bug that caused exporting of subsets to not work with integer
  data. [#1472]

v0.12.1 (2017-10-30)
--------------------

* Fix a bug that caused glue to crash when adding components to a dataset
  after closing a viewer that had that data. [#1460, #1464]

v0.12.0 (2017-10-25)
--------------------

* Show a GUI error message when restoring a session via drag-and-drop
  if session loading fails. [#1454]

* Don't disable layer completely if it is not enabled, just disable checkbox.
  Also show warnings instead of layer style editor. [#1451]

* Generalize registry for data/subset actions to replace the former
  single_subset_action registry (which applied only to single subset selections).
  Layer actions can now be registered with the ``@layer_action`` decorator. [#1396]

* Added support for plotting vectors in the scatter plot viewer. [#1410]

* Added glue plugins to the Version Information dialog. [#1427]

* Added the ability to create fixed layout tabs. [#1403]

* Fix selection in custom viewers. [#1453]

* Fix a bug that caused the home/reset limits button to not work correctly. [#1452]

* Fix a bug that caused the wrong layers to be enabled when mixing image and
  scatter layers and setting up links. [#1451]

* Remove 'sep' from menu on Linux. [#1394]

* Fixed bug in spectrum tool that caused the upper range in aggregations
  to be incorrectly calculated. [#1402]

* Fixed icon for scatter plot layer when a colormap is used, and fix issues with
  viewer layer icons not updating immediately. [#1425]

* Fixed dragging and dropping session files onto glue (this now loads the session
  rather than trying to load it as a dataset). Also now show a warning when
  the application is about to be reset to open a new session. [#1425, #1448]

* Make sure no errors happen if making a selection in an empty viewer. [#1425]

* Fix creating faceted subsets on Python 3.x when no dataset is selected. [#1425]

* Fix issues with overlaying a scatter layer on an image. [#1425]

* Fix issues with labels for categorical axes in the scatter and histogram
  viewers, in particular when loading viewers with categorical axes from
  session files. [#1425]

* Make sure a GUI error message is shown when adding non-1-dimensional data
  to a table viewer. [#1425]

* Fix issues when trying to launch glue multiple times from a Jupyter session.
  [#1425]

* Remove the ability to define the color of a subset differ from that of a
  subset group it belongs to - this was virtually never needed but could
  cause issues. [#1426]

* Fixed a bug that caused a previously disabled image subset layer to not
  become visible when shown again. [#1450]

* Added the ability to rename tabs programmatically. [#1405]

v0.11.1 (2017-08-25)
--------------------

* Fixed bug that caused ModestImage references to not be properly deleted, in
  turn leading to issues/crashes when removing subsets from image viewers. [#1390]

* Fixed bug with reading in old session files with a table viewer. [#1389]

v0.11.0 (2017-08-22)
--------------------

* Added splash screen. [#694]

* Make file extension check case-insensitive. [#1275]

* Fixed bug that caused table viewer to not update when adding components. [#1386]

* Fixed loading of plain (non-structured) arrays from Numpy files. [#1314, #1385]

* Disabled layer artists can no longer be selected to avoid any confusion. [#1367]

* Layer artist icons can now show colormaps when appropriate. [#1367]

* Fix behavior of data wizard so that it doesn't overwrite labels set by data
  factories. [#1367]

* Add a status tip for all ROI selection tools. [#1367]

* Fixed a bug that caused the terminal to not be available after
  resetting or opening a session. [#1366]

* If a subset's visual properties are changed, change the visual
  properties of the parent SubsetGroup. [#1365]

* Give an error if the user selects a session file when going through
  the 'Open Data Set' menu. [#1364]

* Improved scatter plot viewer to be able to show points with color or
  size based on other attributes. Also added a 'line' style to make line
  plots, and added the ability to show error bars. [#1358]

* Changed order of arguments for data exporters from (data, filename)
  to (filename, data). [#1251]

* Added registry decorators to define subset mask importers and
  exporters. [#1251]

* Get rid of QTimers for updating the data collection and layer artist
  lists, and instead refresh whenever a message is sent from the hub
  (which results in immediate changes rather than waiting up to a
   second for things to change). [#1343]

* Made it possible to delay callbacks from the Hub using the
  ``Hub.delay_callbacks`` context manager. Also fixed the Hub so that
  it uses weak references to classes and methods wherever possible. [#1339]

* Added a new method DataCollection.remove_link to match existing
  DataCollection.add_link. [#1339]

* Fix a bug that caused no messages to be emitted when components were
  removed from Data objects, and add a new DataRemoveComponentMesssage.
  [#1339]

* Fix a long-standing bug which caused performance issues after linking
  coordinate or derived components between datasets. [#1339]

* Added a function is_equivalent_cid that can be used to determine whether
  two component IDs in a dataset are equivalent. [#1339]

* The image contrast and bias can now be set with the left click as well
  as right click. [#1323]

* Updated ComponentIDComboHelper so that it can work with single datasets
  that aren't necessarily attached to a DataCollection. [#1296]

* Updated bundled version of echo to include fixes to avoid circular
  references, which in turn caused some callback functions to not be
  cleaned up. [#1281]

* Rewrote the histogram, scatter, and image viewers to use the new state
  infrastructure. This significantly simplifies the actual histogram viewer code
  both in terms of number of lines and in terms of the number of
  connections/callbacks that need to be set up manually. [#1278, #1289, #1388]

* Updated EditSubsetMode so that Data objects no longer have an edit_subset
  attribute - instead, the current list of subsets being edited is kept in
  EditSubsetMode itself. We also update the subset state only once on each
  subset group, rather than once per dataset, which avoids doing the same
  update to each dataset multiple times. [#1338]

* Remove the ability to create a new viewer by right-clicking on the canvas,
  since this causes confusion when trying to control-click to paste in the
  IPython terminal. [#1342]

* Make process_dialog more robust. [#1333]

* Fix example of setting up a custom preferences pane. [#1326]

* Fix a bug that caused links to not get removed if associated datasets
  were removed. [#1329]

* Fixed a bug that meant that the table viewer did not update when
  a ``NumericalDataChangedMessage`` message was emitted. [#1378]

* Added new combo helpers in ``glue.core.data_combo_helper`` which
  are similar to those in ``glue.core.qt.data_combo_helper`` but
  operate on ``SelectionCallbackProperty`` and are Qt-independent.
  [#1346]

* Rewrote installation instructions. [#1330]

v0.10.4 (2017-05-23)
--------------------

* Fixed a bug that caused merged datasets to crash viewers (because
  the visible_components attribute returned an empty list). [#1288]

v0.10.3 (2017-04-20)
-------------------

* Fixed bugs with saving and restoring of various types of subset states. [#1285]

* Fixed a bug that caused glue to not open when IPython 4.0 was installed. [#1287]

v0.10.2 (2017-03-22)
--------------------

* Fixed a bug that caused components that were linked to then disappear from
  drop-down lists of available components in new viewers. [#1270]

* Fixed a bug that caused Data.find_component_id to return incorrect results
  when string components were present in the data. [#1269]

* Fixed a bug that caused errors to appear in the console log after a
  table viewer was closed. [#1267]

* Fixed a bug that caused error message dialogs to not work correctly with
  Qt4. [#1262]

* Fixed a deprecation warning for pandas >= 0.19. [#1263]

* Hide common Matplotlib warnings when min/max along an axis are equal. [#1268]

* Fixed a bug that caused sessions with table viewers that had no subsets
  to not be restored correctly. [#1271]

v0.10.1 (2017-03-16)
--------------------

* Fixed compatibility with session files from before v0.8. [#1243]

* Renamed package to glue-core, since glueviz is now a meta-package (no need
  for a new major version since this change should be seamless to users).

v0.10.0 (2017-02-14)
--------------------

* The GlueApplication.add_data and load_data methods now return the
  loaded data objects. [#1239]

* Change default name of subsets to include the word 'Subset'. [#1234]

* Removed ginga plugin from core package and moved it to a separate repository
  at https://github.com/ejeschke/glue-ginga.

* Switch from using bundled WCSAxes to using the version in Astropy, and fixed
  an issue that caused the frame of the axes to be too thick. [#1231]

* Make it possible to define a default index for DataComboHelper - this makes
  it possible for viewers to have three DataComboHelper and ensure that they
  default to different attributes. [#1163]

* Deal properly with adding Subset objects to DataComboHelper. [#1163]

* Added the ability to export data and subset from the data collection view via
  contextual menus. It was previously possible to export only the mask itself,
  and only to FITS files, but the framework for exporting data/subsets has now
  been generalized.

* When hiding layers in the RGB image viewer, make sure the current layer changes
  to be a visible one if possible.

* Avoid merging IDs when creating identity links. The previous behavior of
  merging was good for performance but made it impossible to undo the linking by
  un-glueing. Derived components created by links are now hidden by default.
  Finally, ``ComponentID`` objects now hold a reference to the first parent data
  they are used in. [#1189]

* Added a decorator that can be used to avoid circular calling of methods (can
  occur when dealing with callbacks). [#1207]

* Drop support for IPython 3.x and below, and make IPython and qtconsole into
  required dependencies. [#1145]

* Added new classes that can be used to hold the state of e.g. viewers and other
  objects. These classes allow callbacks to be attached to various properties,
  and can be used to define logical relations between different attributes
  in a GUI-independent way.

* Fix selections when using Matplotlib 2.x, PyQt5 and a retina display. [#1236]

* Updated ComponentIDComboHelper to no longer store ``(cid, data)`` as the
  ``userData`` but instead just the ``cid`` (the data is now accessible via
  ``cid.parent``). [#1212]

* Avoid duplicate toolbar in dendrogram viewer. [#1213, #1237]

* Fixed bug that caused the contrast to change for the incorrect layer in the
  RGB image viewer.

* Fixed bug that caused coordinate information to be lost when merging datasets.
  The original behavior of keeping the coordinate information from the first
  dataset has been restored. [#1186]

* Fix Data.update_values_from_data to make sure that original component order
  is preserved. [#1238]

* Fix Data.components to return original component order, not alphabetical order.
  [#1238]

* Fix significant performance bottlenecks with WCS coordinate conversions. [#1185]

* Fix error when changing the contrast radio button the RGB image viewer mode,
  and also fix bugs with setting the range of values manually. [#1187]

* Fix a bug that caused coordinate axis labels to be lost during merging. [#1195]

* Fix a bug that caused tab names to not be saved and restored to/from session
  files. [#1241, #1242]

v0.9.1 (2016-11-01)
-------------------

* Fixed loading of session files made with earlier versions of glue that
  contained selections made in 3D viewers. [#1160]

* Fixed plotting of fit on spectrum to make sure that the two are properly
  aligned. [#1158]

* Made it possible to now create InequalitySubsetStates for
  categorical components using e.g. d.id['a'] == 'string'. [#1153]

* Fixed a bug that caused selections to not propagate properly between
  linked images and cubes. [#1144]

* Make last interval of faceted subsets inclusive so as to make sure all values
  in the faceted subset range end up in a subset. [#1154]

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
