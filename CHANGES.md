# Full changelog

## v1.9.1 - 2023-04-13

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### Other Changes

- Use new location for pandas.testing by @jfoster17 in https://github.com/glue-viz/glue/pull/2381
- Only force reference data change if message indicates that numerical values of data have changed by @astrofrog in https://github.com/glue-viz/glue/pull/2385

**Full Changelog**: https://github.com/glue-viz/glue/compare/v1.9.0...v1.9.1

## v1.9.0 - 2023-04-03

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### New Features

- Add a LinkSameWithUnits class that links respecting units by @astrofrog in https://github.com/glue-viz/glue/pull/2379

#### Bug Fixes

- Save legend state by @Carifio24 in https://github.com/glue-viz/glue/pull/2380
- Fix bugs related to generic BaseCartesianData subclasses by @astrofrog in https://github.com/glue-viz/glue/pull/2344

**Full Changelog**: https://github.com/glue-viz/glue/compare/v1.8.1...v1.9.0

## v1.8.1 - 2023-03-23

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### Bug Fixes

- Fixed a bug in the plugin iteration on Python 3.8 and 3.9 by @neutrinoceros in https://github.com/glue-viz/glue/pull/2377
- Avoid interference of ROI patches with legend auto-placement by @dhomeier in https://github.com/glue-viz/glue/pull/2370
- Add pretransform for full-sphere selection by @Carifio24 in https://github.com/glue-viz/glue/pull/2360
- Only recreate table model when Data layer changes by @Carifio24 in https://github.com/glue-viz/glue/pull/2372

**Full Changelog**: https://github.com/glue-viz/glue/compare/v1.8.0...v1.8.1

## v1.8.0 - 2023-03-20

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### New Features

- Relax longitude range in fullsphere projections by @Carifio24 in https://github.com/glue-viz/glue/pull/2348

#### Bug Fixes

- Update polar log transform test values to correct matplotlib 3.7 representation by @Carifio24 in https://github.com/glue-viz/glue/pull/2366
- Fix broken link editor under Qt5 by @jfoster17 in https://github.com/glue-viz/glue/pull/2375

#### Documentation

- DOC: Fix equivalent_units in examples by @pllim in https://github.com/glue-viz/glue/pull/2369

#### Other Changes

- Update link for slack invite by @astrofrog in https://github.com/glue-viz/glue/pull/2362
- Update stable version of standalone app to 2023.02.0 by @astrofrog in https://github.com/glue-viz/glue/pull/2363
- MNT: drop runtime dependency on pkg_resources (setuptools) by @neutrinoceros in https://github.com/glue-viz/glue/pull/2365
- BUG: handle deprecation warning from numpy 1.25 (np.product is deprecated in favour of np.prod) by @neutrinoceros in https://github.com/glue-viz/glue/pull/2371

**Full Changelog**: https://github.com/glue-viz/glue/compare/v1.7.0...v1.8.0

## v1.7.0 - 2023-02-02

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### New Features

- Added ability to convert units in profile viewer by @astrofrog in https://github.com/glue-viz/glue/pull/2296
- Allow specifying visual attributes and colormap when faceting subsets by @Carifio24 in https://github.com/glue-viz/glue/pull/2350

#### Bug Fixes

- Pin numpy-dev to avoid `GlueSerializer` failure on `numpy._ArrayFunctionDispatcher` by @dhomeier in https://github.com/glue-viz/glue/pull/2352
- Represent `None` `component.units` as empty strings by @dhomeier in https://github.com/glue-viz/glue/pull/2356
- Ensure unit choices are updated when x_att is changed in profile viewer by @astrofrog in https://github.com/glue-viz/glue/pull/2358

**Full Changelog**: https://github.com/glue-viz/glue/compare/v1.6.1...v1.7.0

## v1.6.1 - 2023-01-24

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### Bug Fixes

- More fixes for compatibility with Qt6 by @astrofrog in https://github.com/glue-viz/glue/pull/2349
- Fix world coordinates for 1D WCS by @astrofrog in https://github.com/glue-viz/glue/pull/2345

#### Other Changes

- API update to numpy 1.24 by @dhomeier in https://github.com/glue-viz/glue/pull/2346

**Full Changelog**: https://github.com/glue-viz/glue/compare/v1.6.0...v1.6.1

## v1.6.0 - 2022-11-03

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### New Features

- Implement linking support for all BaseCartesianData subclasses and fix bugs that caused hanging in image viewer by @astrofrog in https://github.com/glue-viz/glue/pull/2328
- Add the ability to create one-to-one 'join_on_key'-type links to the GUI link editor by @jfoster17 in https://github.com/glue-viz/glue/pull/2313
- Allow specifying visual attributes when creating subset by @Carifio24 in https://github.com/glue-viz/glue/pull/2333
- Export color settings in matplotlib viewers by @Carifio24 in https://github.com/glue-viz/glue/pull/2322
- Add support for PyQt6 and PySide6 (<6.4) to Qt backends by @astrofrog in https://github.com/glue-viz/glue/pull/2318

#### Bug Fixes

- Redraw empty histogram layer by @Carifio24 in https://github.com/glue-viz/glue/pull/2300
- Apply foreground color to minor ticks by @Carifio24 in https://github.com/glue-viz/glue/pull/2305
- Don't change labels when using log scale axes by @Carifio24 in https://github.com/glue-viz/glue/pull/2323
- Fix bugs in BaseCartesianData.get_data for pixel and world component IDs by @astrofrog in https://github.com/glue-viz/glue/pull/2327
- Don't change labels when using log x axis in histogram viewer by @Carifio24 in https://github.com/glue-viz/glue/pull/2325
- Fix issue where matplotlib viewers using `y` log axes were not initialized correctly. by @Carifio24 in https://github.com/glue-viz/glue/pull/2324
- Force line collection to always respect colormap by @Carifio24 in https://github.com/glue-viz/glue/pull/2299
- Fix export to Python script to correctly use `Coordinates` or `WCS` in `reference_data` by @dhomeier in https://github.com/glue-viz/glue/pull/2335
- Add a check for FITS files mistaken as ASCII to `astropy_table_read` for Python 3.11 compatibility by @dhomeier in https://github.com/glue-viz/glue/pull/2321

#### Documentation

- Removed mention of defunct `viz` packages in *Available plugins* by @pllim in https://github.com/glue-viz/glue/pull/2319
- Add PyQt6 and PySide6 status to `glue-deps` information by @dhomeier in https://github.com/glue-viz/glue/pull/2338

#### Other Changes

- Require Python>=3.8 and PyQt5/PySide5>=5.14 by @astrofrog in https://github.com/glue-viz/glue/pull/2334
- Performance improvements for CompositeArray by @astrofrog in https://github.com/glue-viz/glue/pull/2343

**Full Changelog**: https://github.com/glue-viz/glue/compare/v1.5.0...v1.6.0

## [v1.5.0](https://github.com/glue-viz/glue/compare/v1.4.0...v1.5.0) - 2022-06-28

### What's Changed

#### New Features

- Added rotation angle `theta` as a property to regions of interest,
- to be set on instantiation or modified using the `rotate_to` and
- `rotate_by` methods. https://github.com/glue-viz/glue/pull/2235

#### Bug Fixes

- Fixed a bug on setting a return view in `compute_statistic` when a subset
- is defined, resulting in a broadcast error in arrays large enough to need
- chunking. https://github.com/glue-viz/glue/pull/2302

## [v1.4.0](https://github.com/glue-viz/glue/compare/v1.3.0...v1.4.0) - 2022-05-31

### What's Changed

#### New Features

- Add support for specifying visual attributes when creating a subset group. https://github.com/glue-viz/glue/pull/2297
- 
- Add support for using degrees in full-sphere projections. https://github.com/glue-viz/glue/pull/2279
- 

#### Bug Fixes

- Modify profile viewer so that when in 'Sum' mode, parts of profiles
- with no valid values are NaN rather than zero. https://github.com/glue-viz/glue/pull/2298

## [v1.3.0](https://github.com/glue-viz/glue/compare/v1.2.4...v1.3.0) - 2022-04-22

### What's Changed

#### New Features

- Modify scatter viewer so that axis labels are shown in tick marks
- when in polar mode. https://github.com/glue-viz/glue/pull/2267
- 
- Support toggling plotting profile viewer layer as steps. https://github.com/glue-viz/glue/pull/2292
- 

#### Bug Fixes

- Reset limits in profile viewer when changing collapse function. https://github.com/glue-viz/glue/pull/2277
- 
- Fixed a bug that caused an error when the table viewer tried to show
- disabled layers. https://github.com/glue-viz/glue/pull/2286
- 
- Fixed an issue where glue modified global logging settings. https://github.com/glue-viz/glue/pull/2281
- 

#### Other Changes

- Remove bundled version of pvextractor and include it as a proper
- dependency. https://github.com/glue-viz/glue/pull/2252

## [v1.2.4](https://github.com/glue-viz/glue/compare/v1.2.3...v1.2.4) - 2022-01-27

### What's Changed

#### Bug Fixes

- Fixed a bug that caused selections to no longer work
- in a scatter viewer once its projection had been changed. https://github.com/glue-viz/glue/pull/2262
- 
- Fixed a bug which prevented serialization for polar plots in degree
- mode. https://github.com/glue-viz/glue/pull/2259
- 
- Fixed a bug that caused histograms and density maps to not work
- correctly with attributes linked using `join_on_key`. https://github.com/glue-viz/glue/pull/2242
- 
- Fixed issues in viewers when dask arrays were used. https://github.com/glue-viz/glue/pull/2249
- 

## [v1.2.3](https://github.com/glue-viz/glue/compare/v1.2.2...v1.2.3) - 2021-11-14

### What's Changed

#### Bug Fixes

- Fixed compatibility with Matplotlib 3.5.0. https://github.com/glue-viz/glue/pull/2250
- 
- Fixed compatibility with Python 3.10 and recent versions of PyQt. https://github.com/glue-viz/glue/pull/2258
- 

#### Other Changes

- Remove bottleneck dependency as it is no longer maintained. Certain
- array operations may be slower as a result. https://github.com/glue-viz/glue/pull/2258

## [v1.2.2](https://github.com/glue-viz/glue/compare/v1.2.1...v1.2.2) - 2021-09-16

### What's Changed

#### Bug Fixes

- Prevent resetting of image viewer limits when adding a dataset to the
- viewer. https://github.com/glue-viz/glue/pull/2232
- 
- Fixed home button for resetting limits in profile viewer. https://github.com/glue-viz/glue/pull/2233
- 
- Fixed a bug that caused the visibility checkbox of layers in the profile
- viewer to not always correctly. https://github.com/glue-viz/glue/pull/2233
- 
- Fixed a bug that caused the profile viewer limits to be unecessarily
- changed every time a subset was updated or a dataset was added. https://github.com/glue-viz/glue/pull/2233
- 

## [v1.2.1](https://github.com/glue-viz/glue/compare/v1.2.0...v1.2.1) - 2021-08-24

### What's Changed

#### Bug Fixes

- Fixed a bug that caused issues with packages depending on glue which
- defined layer artists with no zorder on the artist class. https://github.com/glue-viz/glue/pull/2226, https://github.com/glue-viz/glue/pull/2227

## [v1.2.0](https://github.com/glue-viz/glue/compare/v1.1.0...v1.2.0) - 2021-08-14

### What's Changed

#### New Features

- Added a new `WCSLink.as_affine_link` method which can be used to find
- an affine approximation to a WCSLink. https://github.com/glue-viz/glue/pull/2219
- 
- Expose `DataCollection.delay_link_manager_update` which can be used
- to delay any updating to the link tree when adding datasets. https://github.com/glue-viz/glue/pull/2225
- 
- Allow CategoricalComponents to be n-dimensional. https://github.com/glue-viz/glue/pull/2214
- 
- Improve usability of 2D scatter plot in non-cartesian projection. https://github.com/glue-viz/glue/pull/2200
- 

#### Bug Fixes

- Avoid duplicate update of layer artist when creating a new layer. https://github.com/glue-viz/glue/pull/2220

## [v1.1.0](https://github.com/glue-viz/glue/compare/v1.0.1...v1.1.0) - 2021-07-21

### What's Changed

#### Bug Fixes

- Fix compatibility with xlrd 2.0 and require openpyxl for .xlsx input. https://github.com/glue-viz/glue/pull/2196
- 
- Fixed compatibility with recent releases of Matplotlib. https://github.com/glue-viz/glue/pull/2196
- 
- Avoid warnings with recent releases of astropy. https://github.com/glue-viz/glue/pull/2212
- 
- Fixed compatibility of auto-linking with all APE 14 WCSes. https://github.com/glue-viz/glue/pull/2209
- 

#### Other Changes

- Dropped support for Python 3.6. https://github.com/glue-viz/glue/pull/2196

## [v1.0.1](https://github.com/glue-viz/glue/compare/v1.0.0...v1.0.1) - 2020-11-10

### What's Changed

- Updated supported versions of `jupyter_client`. https://github.com/glue-viz/glue/pull/2175

## [v1.0.0](https://github.com/glue-viz/glue/compare/v0.15.7...v1.0.0) - 2020-09-17

### What's Changed

- Remove bundled echo package and list as a dependency. https://github.com/glue-viz/glue/pull/2125
- 
- Add the ability to export Python scripts for the profile viewer. https://github.com/glue-viz/glue/pull/2082
- 
- Add support for polar and other non-rectilinear projections in
- 2-d scatter viewer https://github.com/glue-viz/glue/pull/2170
- 
- Add legend for matplotlib viewers (in qt and in export scripts) https://github.com/glue-viz/glue/pull/2097, https://github.com/glue-viz/glue/pull/2144, https://github.com/glue-viz/glue/pull/2146
- 
- Add new registry to apply in-place patches to the session file https://github.com/glue-viz/glue/pull/2127
- 
- Initial support for dask arrays. https://github.com/glue-viz/glue/pull/2137, https://github.com/glue-viz/glue/pull/2149
- 
- Don't sync color and transparency of image layers by default. https://github.com/glue-viz/glue/pull/2116
- 
- Python 2.7 and 3.5 are no longer supported. https://github.com/glue-viz/glue/pull/2075
- 
- Always use replace mode when creating a new subset. https://github.com/glue-viz/glue/pull/2090
- 
- Fixed bugs in the profile viewer that occurred when using the profile viewer and
- setting `reference_data` to a different value than the default one. https://github.com/glue-viz/glue/pull/2078
- 
- Updated the data loader to be able to select directories. https://github.com/glue-viz/glue/pull/2077, https://github.com/glue-viz/glue/pull/2080
- 
- Move spectral-cube data loader to glue-astronomy plugin package. https://github.com/glue-viz/glue/pull/2077
- 
- Show session filename path in window title. https://github.com/glue-viz/glue/pull/2096
- 
- Change `Data.coords` API to now rely on the API described in
- https://github.com/astropy/astropy-APEs/blob/master/APE14.rst https://github.com/glue-viz/glue/pull/2079
- 
- Update minimum required version of Astropy to 4.0. https://github.com/glue-viz/glue/pull/2079
- 
- Added a `DataCollection.clear()` method to remove all datasets. https://github.com/glue-viz/glue/pull/2079
- 
- Fixed a bug that caused profiles of subsets to not be hidden if an
- existing subset was emptied. https://github.com/glue-viz/glue/pull/2095
- 
- Improved `IndexedData` so that world coordinates can now be shown. https://github.com/glue-viz/glue/pull/2081
- 
- Make loading Qt plugins optional when calling `load_plugins`. https://github.com/glue-viz/glue/pull/2112
- 
- Preserve `RectangularROI` rather than converting them to `PolygonalROI`. https://github.com/glue-viz/glue/pull/2112
- 
- Significantly improve performance of `compute_fixed_resolution_buffer` when
- using linked datasets with some axes uncorrelated. https://github.com/glue-viz/glue/pull/2115
- 
- Improve performance of profile viewer when not all layers are visible. https://github.com/glue-viz/glue/pull/2115
- 
- Fixed missing .units on `CoordinateComponent`. https://github.com/glue-viz/glue/pull/2117
- 
- Improve auto-linking of astronomical datasets with WCS information. https://github.com/glue-viz/glue/pull/2161
- 
- Fix bug that occurred when using the GUI link editor when links had
- been defined programmatically. https://github.com/glue-viz/glue/pull/2166
- 
- Add the ability to programmatically set the preferred colormap for a
- dataset using `Data.visual.preferred_cmap` https://github.com/glue-viz/glue/pull/2131, https://github.com/glue-viz/glue/pull/2168
- 
- Fix bug that caused incorrect unit to be shown on slider for images. https://github.com/glue-viz/glue/pull/2159
- 
- Improve performance of `Data.compute_statistic` for subsets. https://github.com/glue-viz/glue/pull/2147
- 
- Fix a bug where `x_att` and `y_att` could end up being out of sync in image viewer. https://github.com/glue-viz/glue/pull/2141
- 

## [v0.15.7](https://github.com/glue-viz/glue/compare/v0.15.6...v0.15.7) - 2020-03-12

### What's Changed

- Fix bug that caused an infinite loop in the table viewer and caused glue to
- hang if too many incompatible subsets were in a table viewer. https://github.com/glue-viz/glue/pull/2105
- 
- Fixed a bug that caused an error when an invalid data was added to the table
- viewer and the table viewer was then automatically closed. https://github.com/glue-viz/glue/pull/2103
- 
- Fixed the dropdowns for vector and error markers to not include datetime
- components (since they represent absolute times, not deltas). https://github.com/glue-viz/glue/pull/2102
- 
- Fixed a bug that caused session files that were saved with LinkCollection to
- not work correctly when re-loaded. https://github.com/glue-viz/glue/pull/2100
- 
- Fixed a bug that caused issues when saving and re-loading sessions that were
- originally created using Excel data with string columns. https://github.com/glue-viz/glue/pull/2101
- 
- Avoid converting circular selections in Matplotlib plots to polygons if
- it can be avoided. https://github.com/glue-viz/glue/pull/2094
- 

## [v0.15.6](https://github.com/glue-viz/glue/compare/v0.15.5...v0.15.6) - 2019-08-22

### What's Changed

- Fixed bugs related to auto-linking of astronomical data with WCS information. In
- particular, links between datasets with celestial coordinates in a different order
- or links between some higher dimensional datasets with celestial axes did not
- always work correctly. https://github.com/glue-viz/glue/pull/2052
- 
- Fixed a bug that caused the auto-linking framework to not run when opening glue from
- the `qglue()` function. https://github.com/glue-viz/glue/pull/2052
- 
- Fixed a limitation that caused pixel selections to not propagate to other datasets.
- https://github.com/glue-viz/glue/pull/2052
- 
- Fixed a deprecation warning related to `add_datasets`. https://github.com/glue-viz/glue/pull/2052
- 
- Fixed compatibility with Python 3.7.4. https://github.com/glue-viz/glue/pull/2060
- 
- Fixed performance issue with arrays that were not in native C order. https://github.com/glue-viz/glue/pull/2056
- 

## [v0.15.5](https://github.com/glue-viz/glue/compare/v0.15.4...v0.15.5) - 2019-07-09

### What's Changed

- Fixed bug with density map visibility when using color-coding. https://github.com/glue-viz/glue/pull/2041
- 
- Fixed bug with incompatible subsets and density maps. https://github.com/glue-viz/glue/pull/2041
- 
- Make sure that lines/errors/vectors are disabled when in density map mode. https://github.com/glue-viz/glue/pull/2041
- 

## [v0.15.4](https://github.com/glue-viz/glue/compare/v0.15.3...v0.15.4) - 2019-07-08

### What's Changed

- Fixed bug that occurred when trying to add a subset to a new table
- viewer (without the parent data). https://github.com/glue-viz/glue/pull/2038

## [v0.15.3](https://github.com/glue-viz/glue/compare/v0.15.2...v0.15.3) - 2019-06-27

### What's Changed

- Fixed bugs related to the preferences dialog - first, the dialog would
- not open if no auto-linkers were present, and second, the preferences
- dialog would sometimes get sent behind the main application when editing
- the color. https://github.com/glue-viz/glue/pull/2034

## [v0.15.2](https://github.com/glue-viz/glue/compare/v0.15.1...v0.15.2) - 2019-06-24

### What's Changed

- Fixed a bug in `autoconnect_callbacks_to_qt` which caused some widgets
- to not stay connect to state callback properties if a callback property
- was linked to multiple widgets. https://github.com/glue-viz/glue/pull/2032

## [v0.15.1](https://github.com/glue-viz/glue/compare/v0.15.0...v0.15.1) - 2019-06-24

### What's Changed

- Fixed `__version__` variable. https://github.com/glue-viz/glue/pull/2031
- 
- Fixed tox configuration. https://github.com/glue-viz/glue/pull/2031
- 
- Improve error message if loading a session file that uses WCS auto-linking
- but the installed version of astropy is too old. https://github.com/glue-viz/glue/pull/2031
- 

## [v0.15.0](https://github.com/glue-viz/glue/compare/v0.14.2...v0.15.0) - 2019-06-23

### What's Changed

- Added a new `glue.core.coordinates.AffineCoordinates` class for common
- affine transformations, and also added documentation on defining custom
- coordinates. https://github.com/glue-viz/glue/pull/1994
- 
- Rewrote `SubsetFacet` (now `SubsetFacetDialog`), and updated the
- available colormaps. https://github.com/glue-viz/glue/pull/1998
- 
- Removed the `ComponentSelector` class. https://github.com/glue-viz/glue/pull/1998
- 
- Make it possible to view only a subset of data in the table
- viewer. https://github.com/glue-viz/glue/pull/1988
- 
- Show dataset name in table viewer title. https://github.com/glue-viz/glue/pull/1973
- 
- Expose an option (`inherit_tools`) on data viewer classes
- related to whether tools should be inherited or not from
- parent classes. https://github.com/glue-viz/glue/pull/1972
- 
- Added a new method `compute_fixed_resolution_buffer` to data
- objects (including the base data class) and use this for the
- image viewer. This improves the case where images are reprojected
- as they are now all reprojected to the screen resolution rather
- than the resolution of the reference data, and this also opens
- up the possibility of doing n-dimensional reprojection. https://github.com/glue-viz/glue/pull/1895
- 
- Added initial infrastructure for developing auto-linking helpers
- and implement an initial astronomy WCS auto-linker. https://github.com/glue-viz/glue/pull/1933
- 
- Improve the user interface for the link editor. [#1934, #1998]
- 
- Added a new `IndexedData` class to represent a derived dataset produced
- by indexing a higher-dimensional dataset. https://github.com/glue-viz/glue/pull/1953
- 
- Removed plot.ly export plugin - this is now part of the glue-plotly
- package. https://github.com/glue-viz/glue/pull/1999
- 
- Fix path to data bundle when exporting Python scripts to another directory
- than the one glue was run from. https://github.com/glue-viz/glue/pull/2023
- 
- Added a `--faulthandler` command-line flag to help debug segmentation
- faults. https://github.com/glue-viz/glue/pull/1974
- 
- Removed `glue.core.qt.roi` submodule. This provided faster versions of
- the Matplotlib ROI classes in `glue.core.roi` but the latter are now
- efficient enough that the Qt-specific versions are no longer needed. https://github.com/glue-viz/glue/pull/1983
- 
- Moved `glue.viewers.common.qt.tool` to `glue.viewers.common.tool`;
- `glue.viewers.common.qt.mouse_mode` to `glue.viewers.matplotlib.mouse_mode`;
- and `glue.viewers.common.qt.toolbar_mode` to `glue.viewers.matplotlib.toolbar_mode`
- and `glue.viewers.matplotlib.qt.toolbar_mode`. https://github.com/glue-viz/glue/pull/1984
- 
- Implement a human-readable str/repr for State objects. https://github.com/glue-viz/glue/pull/2021
- 
- Avoiding importing Qt when using the base histogram and profile layer artists.
- https://github.com/glue-viz/glue/pull/2012
- 
- Fixed a bug that caused error bars to be colored incorrectly if NaN values
- were present. https://github.com/glue-viz/glue/pull/2020
- 
- Fixed compatibility with the latest versions of Matplotlib and Astropy. https://github.com/glue-viz/glue/pull/2020
- 
- No longer suggest merging data by default whenever datasets are added to
- the session. Merging different datasets should now be done manually through
- the GUI. https://github.com/glue-viz/glue/pull/2020
- 
- Fixed compatibility with Numpy 1.16.x. https://github.com/glue-viz/glue/pull/1989
- 
- Improve tab-completion of attribute names in Data to not include
- non-relevant items. https://github.com/glue-viz/glue/pull/1971
- 
- Fix an error that occurred if creating a new viewer was
- cancelled. https://github.com/glue-viz/glue/pull/1952
- 
- Fix error in image viewer when using `update_values_from_data`. https://github.com/glue-viz/glue/pull/1975
- 
- Exclude problematic versions of ipykernel from dependencies. https://github.com/glue-viz/glue/pull/1952
- 
- Make sure that scrolling above a viewer does not result in the
- whole canvas also scrolling. https://github.com/glue-viz/glue/pull/1919
- 
- Fix bug that caused date/time columns in Excel files to not be
- read in correctly.
- 
- Improve performance when reading in large non-FITS files. https://github.com/glue-viz/glue/pull/1920
- 
- Fix bug that caused log parameter to be ignore for density plots. https://github.com/glue-viz/glue/pull/1963
- 
- Fix bug that caused an error when using the profile collapse tool and
- dragging from right to left. https://github.com/glue-viz/glue/pull/2002
- 
- Fix bug that caused labels on the x-axis of the histogram viewer to
- be incorrectly set to numbers instead of strings when showing a
- histogram of a string variable and saving/reloading session. https://github.com/glue-viz/glue/pull/2009
- 

## [v0.14.2](https://github.com/glue-viz/glue/compare/v0.14.2...v0.14.1) - 2019-02-04

### What's Changed

- Fix bug that caused demo VO Table to not be read in correctly with
- recent versions of Numpy and Astropy. https://github.com/glue-viz/glue/pull/1911

## [v0.14.1](https://github.com/glue-viz/glue/compare/v0.14.0...v0.14.1) - 2018-11-23

### What's Changed

- Fix bug that caused the links based on `join_on_key` to not
- always work. https://github.com/glue-viz/glue/pull/1902

## [v0.14.0](https://github.com/glue-viz/glue/compare/v0.13.4...v0.14.0) - 2018-11-14

### What's Changed

- Improved how we handle equal aspect ratio to not depend on
- Matplotlib. https://github.com/glue-viz/glue/pull/1894
- 
- Avoid showing a warning when closing an empty tab. https://github.com/glue-viz/glue/pull/1890
- 
- Fix bug that caused component arithmetic to not work if
- Numpy was imported in user's config.py file. https://github.com/glue-viz/glue/pull/1887
- 
- Added the ability to define custom layer artist makers to
- override default layer artists in viewers. https://github.com/glue-viz/glue/pull/1850
- 
- Fix Plot.ly exporter for categorical components and histogram
- viewer. https://github.com/glue-viz/glue/pull/1886
- 
- Fix issues with reading very large FITS files on some systems. https://github.com/glue-viz/glue/pull/1884
- 
- Added documentation about plugins. https://github.com/glue-viz/glue/pull/1837
- 
- Better isolate code related to pixel selection tool in image
- viewer that depended on Qt. https://github.com/glue-viz/glue/pull/1763
- 
- Improve handling of units in FITS files. https://github.com/glue-viz/glue/pull/1723
- 
- Added documentation about creating viewers for glue using the
- new state-based infrastructure. https://github.com/glue-viz/glue/pull/1740
- 
- Make it possible to pass the initial state of a viewer to an
- application's `new_data_viewer` method. https://github.com/glue-viz/glue/pull/1877
- 
- Ensure that glue can be imported if QtPy is installed but PyQt
- and PySide aren't. [#1865, #1836]
- 
- Fix unit display for coordinates from WCS headers that don't have
- CTYPE but have CUNIT. https://github.com/glue-viz/glue/pull/1856
- 
- Enable tab completion on Data objects. https://github.com/glue-viz/glue/pull/1874
- 
- Automatically select datasets in link editor if there are only two. https://github.com/glue-viz/glue/pull/1837
- 
- Change 'Export Session' dialog to offer to save with relative paths to data
- by default instead of absolute paths. https://github.com/glue-viz/glue/pull/1803
- 
- Added a new method `screenshot` on `GlueApplication` to save a
- screenshot of the current view. https://github.com/glue-viz/glue/pull/1808
- 
- Show the active subset in the toolbar. https://github.com/glue-viz/glue/pull/1797
- 
- Refactored the viewer class base classes https://github.com/glue-viz/glue/pull/1746:
- 
- - `glue.core.application_base.ViewerBase` has been removed in favor of
- 
- 
- 
- 
- 
- 
- 
- - `glue.viewers.common.viewer.BaseViewer` and
- 
- 
- 
- 
- 
- 
- 
- - `glue.viewers.common.viewer.Viewer`.
- 
- 
- 
- 
- 
- 
- 
- - 
- 
- 
- 
- 
- 
- 
- 
- - `glue.viewers.common.viewer.Viewer` is now where the base logic is defined
- 
- 
- 
- 
- 
- 
- 
- - for using state classes in viewers (instead of
- 
- 
- 
- 
- 
- 
- 
- - `glue.viewers.common.qt.DataViewerWithState`).
- 
- 
- 
- 
- 
- 
- 
- - 
- 
- 
- 
- 
- 
- 
- 
- - `glue.viewers.common.qt.DataViewerWithState` is now deprecated.
- 
- 
- 
- 
- 
- 
- 
- - 
- 
- 
- 
- 
- 
- 
- 
- 
- Make it so that the modest image only resamples the data when the
- mouse is no longer pressed - this avoids too many refreshes when
- panning/zooming. https://github.com/glue-viz/glue/pull/1866
- 
- Make it possible to unglue multiple links in one go. https://github.com/glue-viz/glue/pull/1809
- 
- Make it so that adding a subset to a viewer no longer adds the
- associated data, since in some cases the viewer can handle the
- subset size, but not the full data. https://github.com/glue-viz/glue/pull/1807
- 
- Defined a new abstract base class for all datasets, `BaseData`,
- and a base class `BaseCartesianData`,
- which can be used to implement interfaces to datasets that may be
- remote or may not be stored as regular cartesian data. https://github.com/glue-viz/glue/pull/1768
- 
- Add a new method `Data.compute_statistic` which can be used
- to find scalar and array statistics on the data, and use for
- the profile viewer and the state limits helpers. https://github.com/glue-viz/glue/pull/1737
- 
- Add a new method `Data.compute_histogram` which can be used
- to find histograms of specific components, with or without
- subsets applied. https://github.com/glue-viz/glue/pull/1739
- 
- Removed `Data.get_pixel_component_ids` and `Data.get_world_component_ids`
- in favor of `Data.pixel_component_ids` and `Data.world_component_ids`.
- https://github.com/glue-viz/glue/pull/1784
- 
- Deprecated `Data.visible_components` and `Data.primary_components`. https://github.com/glue-viz/glue/pull/1788
- 
- Speed up histogram calculations by using the fast-histogram package instead of
- np.histogram. https://github.com/glue-viz/glue/pull/1806
- 
- In the case of categorical attributes, `Data[name]` now returns a
- `categorical_ndarray` object rather than the indices of the categories. You
- can access the indices with `Data[name].codes` and the unique categories
- with `Data[name].categories`.  https://github.com/glue-viz/glue/pull/1784
- 
- Compute profiles and histograms asynchronously when dataset is large
- to avoid holding up the UI, and compute profiles in chunks to avoid
- excessive memory usage. [#1736, #1764]
- 
- Improved naming of components when merging datasets. https://github.com/glue-viz/glue/pull/1249
- 
- Fixed an issue that caused residual references to viewers
- after they were closed if they were accessed through the
- IPython console. https://github.com/glue-viz/glue/pull/1770
- 
- Don't show layer edit options if layer is not visible. https://github.com/glue-viz/glue/pull/1805
- 
- Make the Matplotlib viewer code that doesn't depend on Qt accessible
- to non-Qt frontends. https://github.com/glue-viz/glue/pull/1841
- 
- Avoid repeated coordinate components in merged datasets. https://github.com/glue-viz/glue/pull/1792
- 
- Fix bug that caused new subset to be created when dragging an existing
- subset in an image viewer. https://github.com/glue-viz/glue/pull/1793
- 
- Better preserve data types when exporting data/subsets to FITS
- and HDF5 formats. https://github.com/glue-viz/glue/pull/1800
- 

## [v0.13.4](https://github.com/glue-viz/glue/compare/v0.13.3...v0.13.4) - 2018-10-19

### What's Changed

- Fix bug that caused .svg icons to not be correctly installed. https://github.com/glue-viz/glue/pull/1882
- 
- Fix bug that occurred in certain cases when using the state attribute limit
- helper with a state class that did not have a log attribute. https://github.com/glue-viz/glue/pull/1842
- 
- Fix HDF5 reader for string columns. https://github.com/glue-viz/glue/pull/1840
- 
- Fix visual bug in link editor in advanced mode when resizing window.
- 
- Fixed a bug that caused custom data importers to no longer work. https://github.com/glue-viz/glue/pull/1813
- 
- Fixed a bug that caused ROIs to not be erased after selection if the
- active subset was not in the list of layers for the viewer. https://github.com/glue-viz/glue/pull/1801
- 
- Always returned to last used folder when opening/saving files. https://github.com/glue-viz/glue/pull/1794
- 
- Show correct dataset when using control-click to select to add
- arithmetic attributes or rename/reorder components. https://github.com/glue-viz/glue/pull/1802
- 
- Improve performance when updating links and changing attributes
- on subsets. https://github.com/glue-viz/glue/pull/1716
- 
- Fix errors that happened when clicking on the 'Export Data' and
- 'Define arithmetic attributes' buttons when no data was present,
- and fixed Qt errors that happened if the data collection changed
- after the 'Export Data' dialog was opened. https://github.com/glue-viz/glue/pull/1795
- 
- Fixed parsing of AVM meta-data from images. https://github.com/glue-viz/glue/pull/1732
- 
- Fixed compatibility with Matplotlib 3.0. https://github.com/glue-viz/glue/pull/1875
- 

## [v0.13.3](https://github.com/glue-viz/glue/compare/v0.13.2...v0.13.3) - 2018-05-08

### What's Changed

- Fixed a bug that caused the expression for derived attributes to
- not immediately be updated in the list of derived attributes. https://github.com/glue-viz/glue/pull/1708
- 
- Fixed a bug that caused combo boxes with Data objects to not update
- when a data label was changed. https://github.com/glue-viz/glue/pull/1704
- 
- Fixed a bug related to callback functions when restoring sessions.
- https://github.com/glue-viz/glue/pull/1695
- 
- Fixed a bug that meant that setting Data.coords after adding
- components didn't work as expected. https://github.com/glue-viz/glue/pull/1196
- 
- Fixed bugs related to components with only NaN or Inf values.
- https://github.com/glue-viz/glue/pull/1712
- 
- Fixed a bug that caused an error when the zoom or pan tools were
- active and the viewer was closed. https://github.com/glue-viz/glue/pull/1712
- 
- Fixed a Qt-related segmentation fault that occurred during the
- testing process and may also affect users. https://github.com/glue-viz/glue/pull/1703
- 
- Show image layer attribute in list of layers. https://github.com/glue-viz/glue/pull/1706
- 
- Fixed a bug that caused scatter plots to not revert to fixed color
- mode after being in linear color mode. https://github.com/glue-viz/glue/pull/1705
- 

## [v0.13.2](https://github.com/glue-viz/glue/compare/v0.13.1...v0.13.2) - 2018-05-01

### What's Changed

- Fixed a bug that caused the EditSubsetMode toolbar to not change
- when EditSubsetMode.mode was changed programatically. https://github.com/glue-viz/glue/pull/1684
- 
- Fixed unintuitive behavior of single-pixel selection tool - now
- moving the crosshairs requires clicking and dragging. https://github.com/glue-viz/glue/pull/1684
- 
- Fixed bug that caused crosshairs to not be hidden when a layer was
- set to not be visible https://github.com/glue-viz/glue/pull/1684
- 
- Fixed a bug that caused viewers to be closed without warning when
- pressing delete. https://github.com/glue-viz/glue/pull/1684
- 

## [v0.13.1](https://github.com/glue-viz/glue/compare/v0.13.0...v0.13.1) - 2018-04-29

### What's Changed

- Fixed resetting and opening of sessions which caused Glue to quit. https://github.com/glue-viz/glue/pull/1681
- 
- Fixed serialization of Data.meta when non-serializable keys or values
- are present. https://github.com/glue-viz/glue/pull/1681
- 

## [v0.13.0](https://github.com/glue-viz/glue/compare/v0.12.5...v0.13.0) - 2018-04-27

### What's Changed

- Added new perceptually uniform Matplotlib colormaps. https://github.com/glue-viz/glue/pull/1679
- 
- Fixed a bug that caused vectors to not correctly be flipped when
- flipping the x/y limits of the plot. https://github.com/glue-viz/glue/pull/1677
- 
- Added a CSV and HDF5 data/subset exporter. https://github.com/glue-viz/glue/pull/1676
- 
- Started adding helpful information dialogs that can be
- hidden. https://github.com/glue-viz/glue/pull/1669
- 
- Make it possible to have a 'None' entry in the ComponentIDComboHelper. https://github.com/glue-viz/glue/pull/1661
- 
- Added a new metadata/header viewer for datasets/subsets. https://github.com/glue-viz/glue/pull/1659
- 
- Re-write spectrum viewer into a generic profile viewer that uses
- subsets to define the areas in which to compute profiles rather
- than custom ROIs. https://github.com/glue-viz/glue/pull/1635
- 
- Added support for PySide2 and remove support for PyQt4 and
- PySide. https://github.com/glue-viz/glue/pull/1662
- 
- Remove support for Matplotlib 1.5. https://github.com/glue-viz/glue/pull/1662
- 
- Renamed `qt4_to_mpl_color` to `qt_to_mpl_color` and
- `mpl_to_qt4_color` to `mpl_to_qt_color`. https://github.com/glue-viz/glue/pull/1662
- 
- Improve performance when changing visual attributes of subsets.
- https://github.com/glue-viz/glue/pull/1617
- 
- Removed `glue.core.qt.data_combo_helper` (we now recommend using
- the GUI framework-independent equivalent in
- `glue.core.data_combo_helper`). https://github.com/glue-viz/glue/pull/1625
- 
- Removed `glue.viewers.common.qt.attribute_limits_helper` in favor
- of `glue.core.state_objects`. https://github.com/glue-viz/glue/pull/1625
- 
- Removed unused function `glue.utils.misc.defer`. https://github.com/glue-viz/glue/pull/1625
- 
- Added a new `FloodFillSubsetState` class to represent and
- calculate subsets made by a flood-fill algorithm. https://github.com/glue-viz/glue/pull/1616
- 
- Added the ability to easily create viewer tools with dropdown
- menus. https://github.com/glue-viz/glue/pull/1634
- 
- Remove the `MatplotlibViewerToolbar` class as it is now no
- longer needed - instead you can just list the matplotlib tools
- directly in the `tools` attribute of the viewer. https://github.com/glue-viz/glue/pull/1634
- 
- Improve hiding/showing of side-panels. No longer hide side-panels
- when glue application goes out of focus. https://github.com/glue-viz/glue/pull/1535
- 
- Use memory-mapping with contiguous arrays in HDF5 files, resulting
- in improved performance for large files. https://github.com/glue-viz/glue/pull/1628
- 
- Deselect tools when the viewer focus changes. [#1584, #1608]
- 
- Added support for whether symbols are shown filled or not. https://github.com/glue-viz/glue/pull/1559
- 
- Improved link editor to include a graph of links. https://github.com/glue-viz/glue/pull/1534
- 
- Improve mouse interaction with ROIs in image viewers, including
- click-and-drag relocation. Allow for more customization of mouse/toolbar
- modes. https://github.com/glue-viz/glue/pull/1515
- 
- Add a toolbar item to save data. [#1516, #1519, #1575]
- 
- Give instructions for how to move selections in status tip. https://github.com/glue-viz/glue/pull/1504
- 
- Improve the display of data cube slice labels to include only the
- precision required given the separation of world coordinate values.
- [#1500, #1660]
- 
- Removed the ability to edit the marker symbol in the style dialog
- since this isn't recognized by any viewer anymore. https://github.com/glue-viz/glue/pull/1560
- 
- Remove back/forward tools in Matplotlib viewer toolbars to
- declutter. https://github.com/glue-viz/glue/pull/1505
- 
- Added a new component manager that makes it possible to rename,
- reorder, and remove components, as well as better manage derived
- components, including editing previous equations. https://github.com/glue-viz/glue/pull/1479
- 
- Added new messages `DataReorderComponentMessage` and
- `DataRenameComponentMessage` which can be subscribed to. https://github.com/glue-viz/glue/pull/1479
- 
- Add support for the datetime64 dtype in Data objects, and adjust
- Matplotlib viewers to correctly show this data. https://github.com/glue-viz/glue/pull/1510
- 
- Make it possible to reorder components in `Data` using the new
- `Data.reorder_components` method. https://github.com/glue-viz/glue/pull/1479
- 
- The default order of components has changed - coordinate components
- will now always come first (rather than second). https://github.com/glue-viz/glue/pull/1479
- 
- Added support for scatter density maps, which is useful when making
- scatter plots of many points. https://github.com/glue-viz/glue/pull/1461
- 
- Improve how ComponentIDComboHelper deals with non-primary components.
- The .visible property has been removed, and a new .derived property
- has been added (to show/hide derived components). Components are now
- split up into sections in the combo boxes. https://github.com/glue-viz/glue/pull/1476
- 
- Fixed a bug that caused ghost components to be added when creating a
- derived component with data[...] = ... https://github.com/glue-viz/glue/pull/1476
- 
- Fixed a bug that caused errors when removing items from a selection
- property linked to a QComboBox. https://github.com/glue-viz/glue/pull/1476
- 
- Added initial support for customizing keyboard shortcuts. [#1475, #1514, #1524]
- 
- Added support for using relative paths in session files. https://github.com/glue-viz/glue/pull/1537
- 
- Remember last session filename and filter used. https://github.com/glue-viz/glue/pull/1537
- 
- EditSubsetMode is now no longer a singleton class and is
- instead instantiated at the Application/Session level. https://github.com/glue-viz/glue/pull/1530
- 
- Improve performance of image viewer. https://github.com/glue-viz/glue/pull/1558
- 
- Added new `Projected3dROI` and `RoiSubsetState3d` classes
- to represent 3D selections made in the projection plane. https://github.com/glue-viz/glue/pull/1522
- 
- Fixed saving of sessions with `BinaryComponentLink`. https://github.com/glue-viz/glue/pull/1533
- 
- Refactored/simplified handling of links between datasets and
- fixed performance issues when adding/removing links or loading
- data collections with many links. [#1531, #1533]
- 
- Significantly improve performance of link computations when the
- links depend only on pixel or world coordinate components. https://github.com/glue-viz/glue/pull/1585
- 
- Added the ability to customize the appearance of tick and axis
- labels in Matplotlib plots. https://github.com/glue-viz/glue/pull/1511
- 
- Added the ability to export Python scripts from the main
- Matplotlib-based viewers. https://github.com/glue-viz/glue/pull/1511
- 
- Added a new selection mode that always forces the creation of a new subset.
- https://github.com/glue-viz/glue/pull/1525
- 
- Added a mouse over pixel selection tool, which creates a one pixel subset
- under the mouse cursor. https://github.com/glue-viz/glue/pull/1619
- 
- Fixed an issue that caused sliders to not be correctly updated when
- switching reference data in the image viewer. https://github.com/glue-viz/glue/pull/1665
- 
- Fixed a bug that caused Data.meta to not be saved/restored from session
- files. https://github.com/glue-viz/glue/pull/1668
- 
- Fixed an issue that caused an IndexError when quitting glue in some
- cases. https://github.com/glue-viz/glue/pull/1657
- 
- Fixed a bug that caused matplotlibrc files to not be ignored. https://github.com/glue-viz/glue/pull/1649
- 
- Fixed a non-deterministic error that happened when closing the
- TableViewer. https://github.com/glue-viz/glue/pull/7310
- 
- Fixed size of markers when value for size is out of vmin/vmax range. https://github.com/glue-viz/glue/pull/1609
- 
- Fix a bug that caused viewer limits to be calculated incorrectly if
- inf/-inf values were present in the data. https://github.com/glue-viz/glue/pull/1614
- 
- Fixed a bug which caused the y-axis in the PV slice viewer to be
- incorrect if the WCS could not be computed. https://github.com/glue-viz/glue/pull/1615
- 
- Fixed a bug that caused the WCS of a PV slice to be incorrect if the
- user has selected a custom order of the axes of a cube in the image
- viewer. https://github.com/glue-viz/glue/pull/1615
- 
- Fixed ticks on log x-axis in histogram viewer. https://github.com/glue-viz/glue/pull/7310
- 
- Fixed a bug that led to poor performance when slicing through data cubes.
- https://github.com/glue-viz/glue/pull/1554
- 

## [v0.12.5](https://github.com/glue-viz/glue/compare/v0.12.4...v0.12.5) - 2018-03-10

### What's Changed

- Fixed a bug which caused the current slices to be lost when adding a second
- dataset to the image viewer. https://github.com/glue-viz/glue/pull/1581
- 
- Fixed a bug when two datasets with a different number of dimensions
- were in an image viewer and a subset was created. https://github.com/glue-viz/glue/pull/1577
- 
- Fixed issues when attempting to close a viewer with the delete key. https://github.com/glue-viz/glue/pull/1574
- 
- Disabled default Matplotlib key bindings. https://github.com/glue-viz/glue/pull/1574
- 
- Fix compatibility with Matplotlib 2.2. https://github.com/glue-viz/glue/pull/1566
- 
- Fix compatibility with some versions of pytest. https://github.com/glue-viz/glue/pull/1520
- 
- Fix calculation of `dependent_axes` to account for cases where there
- are some non-zero non-diagonal PC values. Previously any such values
- resulted in all axes being returned as dependent axes even though this
- isn't necessary. https://github.com/glue-viz/glue/pull/1552
- 
- Avoid prompting users multiple times to merge data when dragging
- and dropping multiple data files onto glue. https://github.com/glue-viz/glue/pull/1564
- 
- Improve error message in PV slicer when `_slice_index` fails. https://github.com/glue-viz/glue/pull/1536
- 
- Fixed a bug that caused an error when trying to save a session that
- included an image viewer with an aggregated slice. https://github.com/glue-viz/glue/pull/1561
- 
- Fixed a bug that caused an error in the terminal if creating a data
- viewer failed properly (with a GUI error message). https://github.com/glue-viz/glue/pull/1501
- 
- Fixed a bug that caused performance issues when hiding all image
- layers from an image viewer. [#1557, #1562]
- 
- Fixed a bug that caused layers to not always be properly removed
- when deleting a row from the layer list. https://github.com/glue-viz/glue/pull/1502
- 
- Make JSON circular reference errors more explicit. https://github.com/glue-viz/glue/pull/1529
- 

## [v0.12.4](https://github.com/glue-viz/glue/compare/v0.12.3...v0.12.4) - 2018-01-09

### What's Changed

- Improve plugin loading to be less sensitive to exact versions of
- installed dependencies for plugins. https://github.com/glue-viz/glue/pull/1487

## [v0.12.3](https://github.com/glue-viz/glue/compare/v0.12.3...v0.12.2) - 2017-11-14

### What's Changed

- Fixed issues with PV slicer and spectrum viewer when changing axes
- in the parent image viewer.

## [v0.12.2](https://github.com/glue-viz/glue/compare/v0.12.1...v0.12.3) - 2017-11-09

### What's Changed

- Fix a bug when renaming tabs through the UI. https://github.com/glue-viz/glue/pull/1470
- 
- Fix a bug that caused the 1D and 2D viewers to not update correctly
- when the numerical values in data were changed. https://github.com/glue-viz/glue/pull/1471
- 
- Fix a bug that caused exporting of subsets to not work with integer
- data. https://github.com/glue-viz/glue/pull/1472
- 

## [v0.12.1](https://github.com/glue-viz/glue/compare/v0.12.0...v0.12.1) - 2017-10-30

### What's Changed

- Fix a bug that caused glue to crash when adding components to a dataset
- after closing a viewer that had that data. [#1460, #1464]

## [v0.12.0](https://github.com/glue-viz/glue/compare/v0.11.1...v0.12.0) - 2017-10-25

### What's Changed

- Show a GUI error message when restoring a session via drag-and-drop
- if session loading fails. https://github.com/glue-viz/glue/pull/1454
- 
- Don't disable layer completely if it is not enabled, just disable checkbox.
- Also show warnings instead of layer style editor. https://github.com/glue-viz/glue/pull/1451
- 
- Generalize registry for data/subset actions to replace the former
- `single_subset_action` registry (which applied only to single subset selections).
- Layer actions can now be registered with the `@layer_action` decorator. https://github.com/glue-viz/glue/pull/1396
- 
- Added support for plotting vectors in the scatter plot viewer. https://github.com/glue-viz/glue/pull/1410
- 
- Added glue plugins to the Version Information dialog. https://github.com/glue-viz/glue/pull/1427
- 
- Added the ability to create fixed layout tabs. https://github.com/glue-viz/glue/pull/1403
- 
- Fix selection in custom viewers. https://github.com/glue-viz/glue/pull/1453
- 
- Fix a bug that caused the home/reset limits button to not work correctly. https://github.com/glue-viz/glue/pull/1452
- 
- Fix a bug that caused the wrong layers to be enabled when mixing image and
- scatter layers and setting up links. https://github.com/glue-viz/glue/pull/1451
- 
- Remove 'sep' from menu on Linux. https://github.com/glue-viz/glue/pull/1394
- 
- Fixed bug in spectrum tool that caused the upper range in aggregations
- to be incorrectly calculated. https://github.com/glue-viz/glue/pull/1402
- 
- Fixed icon for scatter plot layer when a colormap is used, and fix issues with
- viewer layer icons not updating immediately. https://github.com/glue-viz/glue/pull/1425
- 
- Fixed dragging and dropping session files onto glue (this now loads the session
- rather than trying to load it as a dataset). Also now show a warning when
- the application is about to be reset to open a new session. [#1425, #1448]
- 
- Make sure no errors happen if making a selection in an empty viewer. https://github.com/glue-viz/glue/pull/1425
- 
- Fix creating faceted subsets on Python 3.x when no dataset is selected. https://github.com/glue-viz/glue/pull/1425
- 
- Fix issues with overlaying a scatter layer on an image. https://github.com/glue-viz/glue/pull/1425
- 
- Fix issues with labels for categorical axes in the scatter and histogram
- viewers, in particular when loading viewers with categorical axes from
- session files. https://github.com/glue-viz/glue/pull/1425
- 
- Make sure a GUI error message is shown when adding non-1-dimensional data
- to a table viewer. https://github.com/glue-viz/glue/pull/1425
- 
- Fix issues when trying to launch glue multiple times from a Jupyter session.
- https://github.com/glue-viz/glue/pull/1425
- 
- Remove the ability to define the color of a subset differ from that of a
- subset group it belongs to - this was virtually never needed but could
- cause issues. https://github.com/glue-viz/glue/pull/1426
- 
- Fixed a bug that caused a previously disabled image subset layer to not
- become visible when shown again. https://github.com/glue-viz/glue/pull/1450
- 
- Added the ability to rename tabs programmatically. https://github.com/glue-viz/glue/pull/1405
- 

## [v0.11.1](https://github.com/glue-viz/glue/compare/v0.11.0...v0.11.1) - 2017-08-25

### What's Changed

- Fixed bug that caused ModestImage references to not be properly deleted, in
- turn leading to issues/crashes when removing subsets from image viewers. https://github.com/glue-viz/glue/pull/1390
- 
- Fixed bug with reading in old session files with a table viewer. https://github.com/glue-viz/glue/pull/1389
- 

## [v0.11.0](https://github.com/glue-viz/glue/compare/v0.10.4...v0.11.0) - 2017-08-22

### What's Changed

- Added splash screen. https://github.com/glue-viz/glue/pull/694
- 
- Make file extension check case-insensitive. https://github.com/glue-viz/glue/pull/1275
- 
- Fixed bug that caused table viewer to not update when adding components. https://github.com/glue-viz/glue/pull/1386
- 
- Fixed loading of plain (non-structured) arrays from Numpy files. [#1314, #1385]
- 
- Disabled layer artists can no longer be selected to avoid any confusion. https://github.com/glue-viz/glue/pull/1367
- 
- Layer artist icons can now show colormaps when appropriate. https://github.com/glue-viz/glue/pull/1367
- 
- Fix behavior of data wizard so that it doesn't overwrite labels set by data
- factories. https://github.com/glue-viz/glue/pull/1367
- 
- Add a status tip for all ROI selection tools. https://github.com/glue-viz/glue/pull/1367
- 
- Fixed a bug that caused the terminal to not be available after
- resetting or opening a session. https://github.com/glue-viz/glue/pull/1366
- 
- If a subset's visual properties are changed, change the visual
- properties of the parent SubsetGroup. https://github.com/glue-viz/glue/pull/1365
- 
- Give an error if the user selects a session file when going through
- the 'Open Data Set' menu. https://github.com/glue-viz/glue/pull/1364
- 
- Improved scatter plot viewer to be able to show points with color or
- size based on other attributes. Also added a 'line' style to make line
- plots, and added the ability to show error bars. https://github.com/glue-viz/glue/pull/1358
- 
- Changed order of arguments for data exporters from (data, filename)
- to (filename, data). https://github.com/glue-viz/glue/pull/1251
- 
- Added registry decorators to define subset mask importers and
- exporters. https://github.com/glue-viz/glue/pull/1251
- 
- Get rid of QTimers for updating the data collection and layer artist
- lists, and instead refresh whenever a message is sent from the hub
- (which results in immediate changes rather than waiting up to a
- second for things to change). https://github.com/glue-viz/glue/pull/1343
- 
- Made it possible to delay callbacks from the Hub using the
- `Hub.delay_callbacks` context manager. Also fixed the Hub so that
- it uses weak references to classes and methods wherever possible. https://github.com/glue-viz/glue/pull/1339
- 
- Added a new method `DataCollection.remove_link` to match existing
- `DataCollection.add_link`. https://github.com/glue-viz/glue/pull/1339
- 
- Fix a bug that caused no messages to be emitted when components were
- removed from Data objects, and add a new DataRemoveComponentMesssage.
- https://github.com/glue-viz/glue/pull/1339
- 
- Fix a long-standing bug which caused performance issues after linking
- coordinate or derived components between datasets. https://github.com/glue-viz/glue/pull/1339
- 
- Added a function `is_equivalent_cid` that can be used to determine whether
- two component IDs in a dataset are equivalent. https://github.com/glue-viz/glue/pull/1339
- 
- The image contrast and bias can now be set with the left click as well
- as right click. https://github.com/glue-viz/glue/pull/1323
- 
- Updated ComponentIDComboHelper so that it can work with single datasets
- that aren't necessarily attached to a DataCollection. https://github.com/glue-viz/glue/pull/1296
- 
- Updated bundled version of echo to include fixes to avoid circular
- references, which in turn caused some callback functions to not be
- cleaned up. https://github.com/glue-viz/glue/pull/1281
- 
- Rewrote the histogram, scatter, and image viewers to use the new state
- infrastructure. This significantly simplifies the actual histogram viewer code
- both in terms of number of lines and in terms of the number of
- connections/callbacks that need to be set up manually. [#1278, #1289, #1388]
- 
- Updated EditSubsetMode so that Data objects no longer have an `edit_subset`
- attribute - instead, the current list of subsets being edited is kept in
- EditSubsetMode itself. We also update the subset state only once on each
- subset group, rather than once per dataset, which avoids doing the same
- update to each dataset multiple times. https://github.com/glue-viz/glue/pull/1338
- 
- Remove the ability to create a new viewer by right-clicking on the canvas,
- since this causes confusion when trying to control-click to paste in the
- IPython terminal. https://github.com/glue-viz/glue/pull/1342
- 
- Make `process_dialog` more robust. https://github.com/glue-viz/glue/pull/1333
- 
- Fix example of setting up a custom preferences pane. https://github.com/glue-viz/glue/pull/1326
- 
- Fix a bug that caused links to not get removed if associated datasets
- were removed. https://github.com/glue-viz/glue/pull/1329
- 
- Fixed a bug that meant that the table viewer did not update when
- a `NumericalDataChangedMessage` message was emitted. https://github.com/glue-viz/glue/pull/1378
- 
- Added new combo helpers in `glue.core.data_combo_helper` which
- are similar to those in `glue.core.qt.data_combo_helper` but
- operate on `SelectionCallbackProperty` and are Qt-independent.
- https://github.com/glue-viz/glue/pull/1346
- 
- Rewrote installation instructions. https://github.com/glue-viz/glue/pull/1330
- 

## [v0.10.4](https://github.com/glue-viz/glue/compare/v0.10.3...v0.10.4) - 2017-05-23

### What's Changed

- Fixed a bug that caused merged datasets to crash viewers (because
- the `visible_components` attribute returned an empty list). https://github.com/glue-viz/glue/pull/1288

## [v0.10.3](https://github.com/glue-viz/glue/compare/v0.10.2...v0.10.3) - 2017-04-20

### What's Changed

- Fixed bugs with saving and restoring of various types of subset states. https://github.com/glue-viz/glue/pull/1285
- 
- Fixed a bug that caused glue to not open when IPython 4.0 was installed. https://github.com/glue-viz/glue/pull/1287
- 

## [v0.10.2](https://github.com/glue-viz/glue/compare/v0.10.1...v0.10.2) - 2017-03-22

### What's Changed

- Fixed a bug that caused components that were linked to then disappear from
- drop-down lists of available components in new viewers. https://github.com/glue-viz/glue/pull/1270
- 
- Fixed a bug that caused `Data.find_component_id` to return incorrect results
- when string components were present in the data. https://github.com/glue-viz/glue/pull/1269
- 
- Fixed a bug that caused errors to appear in the console log after a
- table viewer was closed. https://github.com/glue-viz/glue/pull/1267
- 
- Fixed a bug that caused error message dialogs to not work correctly with
- Qt4. https://github.com/glue-viz/glue/pull/1262
- 
- Fixed a deprecation warning for pandas >= 0.19. https://github.com/glue-viz/glue/pull/1263
- 
- Hide common Matplotlib warnings when min/max along an axis are equal. https://github.com/glue-viz/glue/pull/1268
- 
- Fixed a bug that caused sessions with table viewers that had no subsets
- to not be restored correctly. https://github.com/glue-viz/glue/pull/1271
- 

## [v0.10.1](https://github.com/glue-viz/glue/compare/v0.10.0...v0.10.1) - 2017-03-16

### What's Changed

- Fixed compatibility with session files from before v0.8. https://github.com/glue-viz/glue/pull/1243
- 
- Renamed package to glue-core, since glueviz is now a meta-package (no need
- for a new major version since this change should be seamless to users).
- 

## [v0.10.0](https://github.com/glue-viz/glue/compare/v0.9.1...v0.10.0) - 2017-02-14

### What's Changed

- The `GlueApplication.add_data` and `load_data` methods now return the
- loaded data objects. https://github.com/glue-viz/glue/pull/1239
- 
- Change default name of subsets to include the word 'Subset'. https://github.com/glue-viz/glue/pull/1234
- 
- Removed ginga plugin from core package and moved it to a separate repository
- at https://github.com/ejeschke/glue-ginga.
- 
- Switch from using bundled WCSAxes to using the version in Astropy, and fixed
- an issue that caused the frame of the axes to be too thick. https://github.com/glue-viz/glue/pull/1231
- 
- Make it possible to define a default index for DataComboHelper - this makes
- it possible for viewers to have three DataComboHelper and ensure that they
- default to different attributes. https://github.com/glue-viz/glue/pull/1163
- 
- Deal properly with adding Subset objects to DataComboHelper. https://github.com/glue-viz/glue/pull/1163
- 
- Added the ability to export data and subset from the data collection view via
- contextual menus. It was previously possible to export only the mask itself,
- and only to FITS files, but the framework for exporting data/subsets has now
- been generalized.
- 
- When hiding layers in the RGB image viewer, make sure the current layer changes
- to be a visible one if possible.
- 
- Avoid merging IDs when creating identity links. The previous behavior of
- merging was good for performance but made it impossible to undo the linking by
- un-glueing. Derived components created by links are now hidden by default.
- Finally, `ComponentID` objects now hold a reference to the first parent data
- they are used in. https://github.com/glue-viz/glue/pull/1189
- 
- Added a decorator that can be used to avoid circular calling of methods (can
- occur when dealing with callbacks). https://github.com/glue-viz/glue/pull/1207
- 
- Drop support for IPython 3.x and below, and make IPython and qtconsole into
- required dependencies. https://github.com/glue-viz/glue/pull/1145
- 
- Added new classes that can be used to hold the state of e.g. viewers and other
- objects. These classes allow callbacks to be attached to various properties,
- and can be used to define logical relations between different attributes
- in a GUI-independent way.
- 
- Fix selections when using Matplotlib 2.x, PyQt5 and a retina display. https://github.com/glue-viz/glue/pull/1236
- 
- Updated ComponentIDComboHelper to no longer store `(cid, data)` as the
- `userData` but instead just the `cid` (the data is now accessible via
- `cid.parent`). https://github.com/glue-viz/glue/pull/1212
- 
- Avoid duplicate toolbar in dendrogram viewer. [#1213, #1237]
- 
- Fixed bug that caused the contrast to change for the incorrect layer in the
- RGB image viewer.
- 
- Fixed bug that caused coordinate information to be lost when merging datasets.
- The original behavior of keeping the coordinate information from the first
- dataset has been restored. https://github.com/glue-viz/glue/pull/1186
- 
- Fix `Data.update_values_from_data` to make sure that original component order
- is preserved. https://github.com/glue-viz/glue/pull/1238
- 
- Fix Data.components to return original component order, not alphabetical order.
- https://github.com/glue-viz/glue/pull/1238
- 
- Fix significant performance bottlenecks with WCS coordinate conversions. https://github.com/glue-viz/glue/pull/1185
- 
- Fix error when changing the contrast radio button the RGB image viewer mode,
- and also fix bugs with setting the range of values manually. https://github.com/glue-viz/glue/pull/1187
- 
- Fix a bug that caused coordinate axis labels to be lost during merging. https://github.com/glue-viz/glue/pull/1195
- 
- Fix a bug that caused tab names to not be saved and restored to/from session
- files. [#1241, #1242]
- 

## [v0.9.1](https://github.com/glue-viz/glue/compare/v0.9.0...v0.9.1) - 2016-11-01

### What's Changed

- Fixed loading of session files made with earlier versions of glue that
- contained selections made in 3D viewers. https://github.com/glue-viz/glue/pull/1160
- 
- Fixed plotting of fit on spectrum to make sure that the two are properly
- aligned. https://github.com/glue-viz/glue/pull/1158
- 
- Made it possible to now create InequalitySubsetStates for
- categorical components using e.g. d.id['a'] == 'string'. https://github.com/glue-viz/glue/pull/1153
- 
- Fixed a bug that caused selections to not propagate properly between
- linked images and cubes. https://github.com/glue-viz/glue/pull/1144
- 
- Make last interval of faceted subsets inclusive so as to make sure all values
- in the faceted subset range end up in a subset. https://github.com/glue-viz/glue/pull/1154
- 

## [v0.9.0](https://github.com/glue-viz/glue/compare/v0.8.2...v0.9.0) - 2016-10-10

### What's Changed

- Fix serialization of celestial coordinate link functions. Classes
- inheriting from MultiLink should now call `MultiLink.__init__` with
- individual components (not grouped into left/right) then the `create_links`
- method with the components separated into left/right and the methods for
- forward/backward transformation. The original behavior can be retained
- by using the `multi_link` function instead of the `MultiLink` class.
- https://github.com/glue-viz/glue/pull/1139
- 
- Improve support for spectral cubes. https://github.com/glue-viz/glue/pull/1075
- 
- Allow link functions/helpers to define a category using the `category=`
- argument (which defaults to `General`), and make it possible to filter
- by category in the link editor. https://github.com/glue-viz/glue/pull/1141
- 
- Only show the 'waiting' cursor when glue is doing something. https://github.com/glue-viz/glue/pull/1097
- 
- Make sure that the scatter layer artist style editor is shown when overplotting
- a scatter plot on top of an image. https://github.com/glue-viz/glue/pull/1134
- 
- Data viewers can now make `layer_style_widget_cls` a dictionary in cases
- where multiple layer artist types are supported. https://github.com/glue-viz/glue/pull/1134
- 
- Fix compatibility of test suite with pytest 3.x. https://github.com/glue-viz/glue/pull/1116
- 
- Updated bundled version of WCSAxes to v0.9. https://github.com/glue-viz/glue/pull/1089
- 
- Fix compatibility with pre-releases of Matplotlib 2.0. https://github.com/glue-viz/glue/pull/1115
- 
- Implement new widget with better control over exporting to Plotly. The
- preference pane for Plotly export has now been removed in favor of this new
- way to set the options. https://github.com/glue-viz/glue/pull/1057
- 
- Renamed the `ComponentIDComboHelper` and `ManualDataComboHelper`
- `append` methods to `append_data` and the `remove` methods to
- `remove_data`, and added a new `ComponentIDComboHelper.set_multiple_data`
- method. https://github.com/glue-viz/glue/pull/1060
- 
- Fixed reading of units from FITS and VO tables, and display units in
- table viewer. [#1135, #1137]
- 
- Make use of the QtPy package to deal with differences between PyQt4, PyQt5,
- and PySide, instead of the custom qt-helpers package. The
- `glue.external.qt` package is now deprecated. The `get_qapp` and
- `load_ui` functions are now available in `glue.utils.qt`.
- [#1018, #1074, #1077, #1078, #1081]
- 
- Avoid raising a (harmless) error when selecting a region in between two
- categorical components.
- 
- Added a new Data method, `update_values_from_data`, that can be used to replicate
- components from one dataset into another. https://github.com/glue-viz/glue/pull/1112
- 
- Refactored code related to toolbars in order to make it easier to define
- toolbars and toolbar modes that aren't Matplotlib-specific. [#1085, #1120]
- 
- Added a new table viewer. [#1084, #1123]
- 
- Fix saving/loading of categorical components. https://github.com/glue-viz/glue/pull/1084
- 
- Make it possible for tools to define a status bar message. https://github.com/glue-viz/glue/pull/1084
- 
- Added a command-line option, `--no-maximized`, that prevents glue
- from opening up with the application window maximized. [#1093, #1126]
- 
- When opening multiple files in one go, if one of the files fails to
- read, the error will now indicate which file failed. https://github.com/glue-viz/glue/pull/1104
- 
- Fixed a bug that caused new subset colors to incorrectly start from the start
- of the color cycle after loading a session. https://github.com/glue-viz/glue/pull/1055
- 
- Fixed a bug that caused the functionality to execute scripts (glue -x) to not
- work in Python 3. [#1101, #1114]
- 
- The minimum supported version of Astropy is now 1.0, and the minimum
- supported version of IPython is now 1.0. https://github.com/glue-viz/glue/pull/1076
- 
- Show world coordinates and units in the cube slicer. [#1059, #1068]
- 
- Fix errors that occurred when selecting categorical data. https://github.com/glue-viz/glue/pull/1069
- 
- Added experimental support for joining on multiple keys in `join_on_key`. https://github.com/glue-viz/glue/pull/974
- 
- Fix compatibility with the latest version of ginga. https://github.com/glue-viz/glue/pull/1063
- 

## [v0.8.2](https://github.com/glue-viz/glue/compare/v0.8.1...v0.8.2) - 2016-07-06

### What's Changed

- Implement missing MaskSubsetState.copy. https://github.com/glue-viz/glue/pull/1033
- 
- Ensure that failing data factory identifier functions are skipped. https://github.com/glue-viz/glue/pull/1029
- 
- The naming of pixel axes is now more consistent between data with 3 or
- fewer dimensions, and data with more than 3 dimensions. The naming is now
- always `Pixel Axis ?` where `?` is the index of the array, and for
- datasets with 1 to 3 dimensions, we add a suffix e.g. `[x]` to indicate
- the traditional axes. https://github.com/glue-viz/glue/pull/1029
- 
- Implemented a number of performance improvements, including for: the check
- of whether points are in polygon (`points_inside_poly`), the selection of
- polygonal regions in multi-dimentional cubes when the selections are along
- pixel axes, the selection of points in scatter plots with one or two
- categorical components for rectangular, circular, and polygonal regions.
- https://github.com/glue-viz/glue/pull/1029
- 
- Fix a bug that caused multiple custom viewer classes to not work properly
- if the user did not override `_custom_functions` (which was private).
- https://github.com/glue-viz/glue/pull/810
- 
- Make sure histograms are updated if only the attribute changes and the
- limits and number of bins stay the same. https://github.com/glue-viz/glue/pull/1012
- 
- Fix a bug on Windows that caused drag and dropping files onto the glue
- application to not work. https://github.com/glue-viz/glue/pull/1007
- 
- Fix compatibility with PyQt5. https://github.com/glue-viz/glue/pull/1015
- 
- Fix a bug that caused ComponentIDComboHelper to not take into account the
- numeric and categorical options in `__init__`. https://github.com/glue-viz/glue/pull/1014
- 
- Fix a bug that caused saving of scatter plots to SVG files to crash. https://github.com/glue-viz/glue/pull/984
- 

## [v0.8.1](https://github.com/glue-viz/glue/compare/v0.8.0...v0.8.1) - 2016-05-25

### What's Changed

- Fixed a bug in the memoize function that caused selections using
- ElementSubsetState to fail when using views on the data. https://github.com/glue-viz/glue/pull/1004
- 
- Explicitly set the icon size for the slicing playback controls to avoid
- issues when using a mix of retina and non-retina displays. https://github.com/glue-viz/glue/pull/1005
- 
- Fixed a bug that caused `add_datasets` to crash if `datasets` was a list of
- lists of data, which is possible if a data factory returns more than one data
- object. https://github.com/glue-viz/glue/pull/1006
- 

## [v0.8.0](https://github.com/glue-viz/glue/compare/v0.7.3...v0.8.0) - 2016-05-23

### What's Changed

- Add support for circular and polygonal spectrum extraction. [#994, #1003]
- 
- Fix compatibility with latest developer version of Numpy which does not allow
- non-integer indices for arrays. https://github.com/glue-viz/glue/pull/1002
- 
- Add a new method `add_data` to application instances. This allows for
- example additional data to be passed to glue after being launched by
- `qglue`. https://github.com/glue-viz/glue/pull/993
- 
- Add playback controls to slice widget. https://github.com/glue-viz/glue/pull/971
- 
- Add tooltip for data labels so that long labels can be more easily
- inspected. https://github.com/glue-viz/glue/pull/912
- 
- Added a new helper class `AttributeLimitsHelper` to link widgets related to
- setting limits and handle the caching of the limits as a function of
- attribute. https://github.com/glue-viz/glue/pull/872
- 
- Add Quit menu item for Linux and Windows. https://github.com/glue-viz/glue/pull/926
- 
- Refactored the window for sending feedback to include more version
- information, and also to have a separate form for feedback and crash
- reports. https://github.com/glue-viz/glue/pull/955
- 
- Add log= option to ValueProperty and remove mapping= option. https://github.com/glue-viz/glue/pull/965
- 
- Added helper classes for ComponentID and Data combo boxes. https://github.com/glue-viz/glue/pull/891
- 
- Improved new component window: expressions can now include math or numpy
- functions by default, and expressions are tested on-the-fly to check that
- there are no issues with syntax or undefined variables. https://github.com/glue-viz/glue/pull/956
- 
- Fixed D3PO export when using Python 3. https://github.com/glue-viz/glue/pull/989
- 
- Fixed display of certain error messages when using Python 3. https://github.com/glue-viz/glue/pull/989
- 
- Add an extensible preferences window. https://github.com/glue-viz/glue/pull/988
- 
- Add the ability to change the foreground and background color for viewers.
- https://github.com/glue-viz/glue/pull/988
- 
- Fixed a bug that caused images to appear over-pixellated on the edges when
- zooming in. https://github.com/glue-viz/glue/pull/1000
- 
- Added an option to control whether warnings are shown when passing large data objects to viewers. https://github.com/glue-viz/glue/pull/999
- 

## [v0.7.3](https://github.com/glue-viz/glue/compare/v0.7.2...v0.7.3) - 2016-05-04

### What's Changed

- Remove icons for actions that appear in contextual menus, since these
- appear too large due to a Qt bug. https://github.com/glue-viz/glue/pull/911
- 
- Add missing `find_spec` for import hook, to avoid issues when trying to set
- colormap. https://github.com/glue-viz/glue/pull/930
- 
- Ignore extra dimensions in WCS (for instance, if the data is 3D and the
- header is 4D, ignore the 4th dimension in the WCS). https://github.com/glue-viz/glue/pull/935
- 
- Fix a bug that caused the merge window to appear multiple times, make sure
- that all components named PRIMARY get renamed after merging, and make sure
- that the merge mechanism is also triggered when opening datasets from the
- command-line. https://github.com/glue-viz/glue/pull/936
- 
- Remove the scrollbars added in v0.7.1 since they cause issues on certain
- systems. https://github.com/glue-viz/glue/pull/953
- 
- Fix saving of ElementSubsetState to session files. https://github.com/glue-viz/glue/pull/966
- 
- Fix saving of Matplotlib colormaps to session files. https://github.com/glue-viz/glue/pull/967
- 
- Fix the selection of the default viewer based on the data shape. https://github.com/glue-viz/glue/pull/968
- 
- Make sure that no combo boxes get resized based on the content (unless
- strictly needed). https://github.com/glue-viz/glue/pull/978
- 

## [v0.7.2](https://github.com/glue-viz/glue/compare/v0.7.1...v0.7.2) - 2016-04-05

### What's Changed

- Fix a bug that caused string columns in FITS files to not be read
- correctly, and updated `coerce_numeric` to give a `ValueError` for string
- columns that can't be convered. https://github.com/glue-viz/glue/pull/919
- 
- Make sure main window title is set. https://github.com/glue-viz/glue/pull/914
- 
- Fix issue with FITS files that are missing an END card. https://github.com/glue-viz/glue/pull/915
- 
- Fix a bug that caused values in exponential notation in text fields to lose
- a trailing zero (e.g. 1.000e+10 would become 1.000e+1). https://github.com/glue-viz/glue/pull/925
- 

## [v0.7.1](https://github.com/glue-viz/glue/compare/v0.7.0...v0.7.1) - 2016-03-30

### What's Changed

- Fix issue with small screens and layer and viewer options by adding
- scrollbars. https://github.com/glue-viz/glue/pull/902
- 
- Fixed a failure due to a missing Qt import in glue.core.roi. https://github.com/glue-viz/glue/pull/901
- 
- Fixed a bug that caused an abort trap if the filename specified on the
- command line did not exist. https://github.com/glue-viz/glue/pull/903
- 
- Gracefully skip vector columnns when reading in FITS files. https://github.com/glue-viz/glue/pull/896
- 
- Change default gray color to work on black and white backgrounds. https://github.com/glue-viz/glue/pull/906
- 
- Fixed a bug that caused the color in the scatter and histogram style
- editors to not show the correct initial color. https://github.com/glue-viz/glue/pull/907
- 

## [v0.7.0](https://github.com/glue-viz/glue/compare/v0.6.0...v0.7.0) - 2016-03-10

### What's Changed

- Python 2.6 is no longer supported. https://github.com/glue-viz/glue/pull/804
- 
- Added a generic QColorBox widget to pick colors, and an associated
- `connect_color` helper for callback properties. https://github.com/glue-viz/glue/pull/864
- 
- Added a generic QColormapCombo widget to pick colormaps.
- 
- The `artist_container` argument to client classes has been renamed to
- `layer_artist_container`. https://github.com/glue-viz/glue/pull/814
- 
- Added documentation about how to use layer artists in custom Qt data viewers.
- https://github.com/glue-viz/glue/pull/814
- 
- Fixed missing newline in `Data.__str__`. https://github.com/glue-viz/glue/pull/877
- 
- A large fraction of the code has been re-organized, which may lead to some
- imports in `config.py` files no longer working. However, no functionality
- has been removed, so this can be fixed by updating the imports to reflect the
- new locations.
- 
- In particular, the following utilities have been moved:
- 
- `glue.qt.widget_properties`                 | `glue.utils.qt.widget_properties`
- `glue.qt.decorators`                        | `glue.utils.qt.decorators`
- `glue.qt.qtutil.mpl_to_qt4_color`           | `glue.utils.qt.colors.mpl_to_qt4_color`
- `glue.qt.qtutil.qt4_to_mpl_color`           | `glue.utils.qt.colors.qt4_to_mpl_color`
- `glue.qt.qtutil.pick_item`                  | `glue.utils.qt.dialogs.pick_item`
- `glue.qt.qtutil.pick_class`                 | `glue.utils.qt.dialogs.pick_class`
- `glue.qt.qtutil.get_text`                   | `glue.utils.qt.dialogs.get_text`
- `glue.qt.qtutil.tint_pixmap`                | `glue.utils.qt.colors.tint_pixmap`
- `glue.qt.qtutil.cmap2pixmap`                | `glue.utils.qt.colors.cmap2pixmap`
- `glue.qt.qtutil.pretty_number`              | `glue.utils.qt.PropertySetMixin`
- `glue.qt.qtutil.Worker`                     | `glue.utils.qt.threading.Worker`
- `glue.qt.qtutil.update_combobox`            | `glue.utils.qt.helpers.update_combobox`
- `glue.qt.qtutil.PythonListModel`            | `glue.utils.qt.python_list_model.PythonListModel`
- `glue.clients.tests.util.renderless_figure` | `glue.utils.matplotlib.renderless_figure`
- `glue.core.util.CallbackMixin`              | `glue.utils.misc.CallbackMixin`
- `glue.core.util.Pointer`                    | `glue.utils.misc.Pointer`
- `glue.core.util.PropertySetMixin`           | `glue.utils.misc.PropertySetMixin`
- `glue.core.util.defer`                      | `glue.utils.misc.defer`
- `glue.qt.mime.PyMimeData`                   | `glue.utils.qt.mime.PyMimeData`
- `glue.qt.qtutil.GlueItemWidget`             | `glue.utils.qt.mixins.GlueItemWidget`
- `glue.qt.qtutil.cache_axes`                 | `glue.utils.matplotlib.cache_axes`
- `glue.qt.qtutil.GlueTabBar`                 | `glue.utils.qt.helpers.GlueTabBar`
- 
- [#827, #828, #829, #830, #831]
- 
- `glue.clients.histogram_client`                  | `glue.viewers.histogram.client`
- `glue.clients.image_client`                      | `glue.viewers.image.client`
- `glue.clients.scatter_client`                    | `glue.viewers.scatter.client`
- `glue.clients.layer_artist.LayerArtist`          | `glue.clients.layer_artist.MatplotlibLayerArtist`
- `glue.clients.layer_artist.ChangedTrigger`       | `glue.clients.layer_artist.ChangedTrigger`
- `glue.clients.layer_artist.LayerArtistContainer` | `glue.clients.layer_artist.LayerArtistContainer`
- `glue.clients.ds9norm`                           | `glue.viewers.image.ds9norm`
- `glue.clients.profile_viewer`                    | `glue.plugins.tools.spectrum_viewer.profile_viewer`
- `glue.clients.util.small_view`                   | `glue.core.util.small_view`
- `glue.clients.util.small_view_array`             | `glue.core.util.small_view_array`
- `glue.clients.util.visible_limits`               | `glue.core.util.visible_limits`
- `glue.clients.util.tick_linker`                  | `glue.core.util.tick_linker`
- `glue.clients.util.update_ticks`                 | `glue.core.util.update_ticks`
- `glue.qt.widgets.histogram_widget`               | `glue.viewers.histogram.qt`
- `glue.qt.widgets.scatter_widget`                 | `glue.viewers.scatter.qt`
- `glue.qt.widgets.histogram_widget`               | `glue.viewers.image.qt`
- `glue.qt.widgets.table_widget`                   | `glue.viewers.table.qt`
- `glue.qt.widgets.data_viewer`                    | `glue.viewers.common.qt.data_viewer`
- `glue.qt.widgets.mpl_widget`                     | `glue.viewers.common.qt.mpl_widget`
- `glue.qt.widgets.MplWidget`                      | `glue.viewers.common.qt.mpl_widget.MplWidget`
- `glue.qt.glue_toolbar`                           | `glue.viewers.common.qt.toolbar`
- `glue.qt.custom_viewer`                          | `glue.viewers.custom.qt`
- 
- https://github.com/glue-viz/glue/pull/835
- 
- `glue.qt.glue_application.GlueApplication`       | `glue.app.qt.application.GlueApplication`
- `glue.qt.plugin_manager.QtPluginManager`         | `glue.app.qt.plugin_manager.QtPluginManager`
- `glue.qt.feedback.FeedbackWidget`                | `glue.app.qt.feedback.FeedbackWidget`
- `glue.qt.widgets.glue_mdi_area`                  | `glue.app.qt.mdi_area`
- 
- `glue.qt.widgets.terminal`                       | `glue.app.qt.terminal`
- `glue.qt.qt_roi`                                 | `glue.core.qt.roi`
- `glue.core.qt.simpleforms`                       | `glue.core.qt.simpleforms`
- `glue.qt.widgets.style_dialog`                   | `glue.core.qt.style_dialog`
- `glue.qt.layer_artist_model`                     | `glue.core.qt.layer_artist_model`
- `glue.qt.widgets.custom_component_widget`        | `glue.dialogs.custom_component.qt`
- `glue.qt.link_editor`                            | `glue.dialogs.link_editor.qt`
- `glue.qt.widgets.subset_facet`                   | `glue.dialogs.subset_facet.qt`
- `glue.qt.mouse_mode`                             | `glue.viewers.common.qt.mouse_mode`
- `glue.qt.data_slice_widget`                      | `glue.viewers.common.qt.data_slice_widget`
- `glue.qt.widgets.layer_tree_widget`              | `glue.app.qt.layer_tree_widget`
- `glue.qt.widgets.message_widget`                 | `glue.core.qt.message_widget`
- `glue.qt.widgets.settings_editor`                | `glue.app.qt.settings_editor`
- `glue.qt.qtutil.data_wizard`                     | `glue.dialogs.data_wizard.qt.data_wizard`
- `glue.qt.mime`                                   | `glue.core.qt.mime`
- `glue.qt.qtutil.ComponentIDCombo`                | `glue.core.qt.component_id_combo`
- `glue.qt.qtutil.RGBEdit`                         | `glue.viewers.image.qt.rgb_edit.RGBEdit`
- `glue.qt.qtutil.GlueListWidget`                  | `glue.core.qt.mime.GlueMimeListWidget`
- `glue.qt.qtutil.load_ui`                         | `glue.utils.qt.helpers.load_ui`
- 
- `glue.qt.qtutil.icon_path`                       | `glue.icons.icon_path`
- `glue.qt.qtutil.load_icon`                       | `glue.icons.qt.load_icon`
- `glue.qt.qtutil.symbol_icon`                     | `glue.icons.qt.symbol_icon`
- `glue.qt.qtutil.layer_icon`                      | `glue.icons.qt.layer_icon`
- `glue.qt.qtutil.layer_artist_icon`               | `glue.icons.qt.layer_artist_icon`
- `glue.qt.qtutil.GlueActionButton`                | `glue.app.qt.actions.GlueActionButton`
- `glue.qt.qtutil.action`                          | `glue.app.qt.actions.action`
- `glue.qt.qt_backend.Timer`                       | `glue.backends.QtTimer`
- 
- https://github.com/glue-viz/glue/pull/845
- 
- Improved under-the-hood creation of ROIs for Scatter and Histogram Clients. https://github.com/glue-viz/glue/pull/676
- 
- Data viewers can now define a layer artist style editor class that appears
- under the list of layer artists. https://github.com/glue-viz/glue/pull/852
- 
- Properties of the VisualAttributes class are now callback properties. https://github.com/glue-viz/glue/pull/852
- 
- Add `glue.utils.qt.widget_properties.connect_value` function which can take
- an optional `value_range` and log option to scale the Qt values to a custom
- range of values (optionally in log space). https://github.com/glue-viz/glue/pull/852
- 
- Make list of data viewers sorted alphabetically. https://github.com/glue-viz/glue/pull/866
- 

## [v0.6.0](https://github.com/glue-viz/glue/compare/v0.5.2...v0.6.0) - 2015-11-20

### What's Changed

- Added experimental support for PyQt5. https://github.com/glue-viz/glue/pull/663
- 
- Fix `glue -t` option. https://github.com/glue-viz/glue/pull/791
- 
- Updated `glue-deps` to show PyQt/PySide versions. https://github.com/glue-viz/glue/pull/796
- 
- Fix bug that caused viewers to be restored with the wrong size. [#781, #783]
- 
- Fixed compatibility with the latest stable version of ginga. https://github.com/glue-viz/glue/pull/797
- 
- Prevent axes from moving around when data viewers are being resized, and
- instead make the absolute margins between axes and figure edge fixed. https://github.com/glue-viz/glue/pull/745
- 
- Fixed a bug that caused image plots to not be updated immediately when
- changing component, and fixed a bug that caused data and attribute combo
- boxes to not update correctly when showing multiple datasets in an
- ImageWidget. https://github.com/glue-viz/glue/pull/755
- 
- Added tests to ensure that we remain backward-compatible with old session
- files for the FITS and HDF5 factories. [#736, #748]
- 
- When a box has been drawn to extract a spectrum from a cube, the box can
- then be moved by pressing the control key and dragging it. https://github.com/glue-viz/glue/pull/707
- 
- Refactored ASCII I/O to include more Astropy table formats. https://github.com/glue-viz/glue/pull/762
- 
- When saving a session, if no extension is specified, the .glu extension is
- added. https://github.com/glue-viz/glue/pull/729
- 
- Added a GUI plugin manager in the 'Plugins' menu. https://github.com/glue-viz/glue/pull/682
- 
- Added an option to specify whether to use an automatic aspect ratio for image
- data or whether to enforce square pixels. https://github.com/glue-viz/glue/pull/717
- 
- Data factories can now be given priorities to determine which ones should
- take precedence in ambiguous cases. The `set_default_factory` and
- `get_default_factory` functions are now deprecated since it is possible to
- achieve this solely with priorities. https://github.com/glue-viz/glue/pull/719
- 
- Improved cube slider to include editable slice value as well as
- first/previous/next/last buttons, and improved spacing of sliders for 4+
- dimensional cubes. [#690, #734]
- 
- Registering data factories should now always be done with the
- `@data_factory` decorator, and not by adding functions to
- `__factories__`, as was possible in early versions of Glue. https://github.com/glue-viz/glue/pull/724
- 
- Made the Excel spreadsheet reader more robust: column headers no longer have
- to be strings, and the reader no longer expects the first sheet to be called
- 'Sheet1'. All sheets are now read by default. Datasets are now named as
- filename:sheetname. https://github.com/glue-viz/glue/pull/726
- 
- Fix compatibility with IPython 4. https://github.com/glue-viz/glue/pull/733
- 
- Improved reading of FITS files - all HDUs are now read by default. [#704, #732]
- 
- Added new widget property classes, for combo boxes (based on label instead of
- data) and for tab widgets. https://github.com/glue-viz/glue/pull/752
- 
- Improved reading of HDF5 files - all datasets in an HDF5 file are now read by
- default. https://github.com/glue-viz/glue/pull/747
- 
- Fix a bug that caused images to not be shown at full resolution after resizing. https://github.com/glue-viz/glue/pull/768
- 
- Fix a bug that caused the color of an extracted spectrum to vary if extracted
- multiple times. https://github.com/glue-viz/glue/pull/743
- 
- Fixed a bug that caused compressed image HDUs to not be read correctly.
- https://github.com/glue-viz/glue/pull/767
- 
- Added two new settings `settings.SUBSET_COLORS` and `settings.DATA_COLOR`
- that can be used to customize the default subset and data colors. https://github.com/glue-viz/glue/pull/742
- 

## v0.5.3 (unreleased)

### What's Changed

- Fix selection in scatter plots when categorical data are present. https://github.com/glue-viz/glue/pull/727

## [v0.5.2](https://github.com/glue-viz/glue/compare/v0.5.1...v0.5.2) - 2015-08-13

### What's Changed

- Fix loading of plugins with setuptools < 11.3 https://github.com/glue-viz/glue/pull/699
- 
- Fix loading of plugins when using glue programmatically rather than through the GUI https://github.com/glue-viz/glue/pull/698
- 
- Backward-compatibility fixes after refactoring `data_factories` [#696, #703]
- 

## [v0.5.1](https://github.com/glue-viz/glue/compare/v0.5.0...v0.5.1) - 2015-07-06

### What's Changed

- Fixed treatment of newlines when copying detailed error. https://github.com/glue-viz/glue/pull/687
- 
- Fix a bug that prevented sessions from being saved with embedded files if
- component units were Astropy units. https://github.com/glue-viz/glue/pull/686
- 
- Users should now press 'control' to drag rather than re-define subsets. https://github.com/glue-viz/glue/pull/689
- 

## [v0.5.0](https://github.com/glue-viz/glue/compare/v0.4.0...v0.5.0) - 2015-07-03

### What's Changed

- Improvements to the PyQt/PySide wrapper module (now maintained in a separate repository). https://github.com/glue-viz/glue/pull/671
- 
- Fixed broken links on website. https://github.com/glue-viz/glue/pull/678
- 
- Added the ability to discover plugins via entry points. https://github.com/glue-viz/glue/pull/677
- 
- Added the ability to include float and string UI elements in custom
- viewers. https://github.com/glue-viz/glue/pull/653
- 
- Added an option to bundle all data in .glu session files. https://github.com/glue-viz/glue/pull/661
- 
- Added a `menu_plugin` registry to add custom tools to the registry. https://github.com/glue-viz/glue/pull/644
- 
- Support for 'lazy-loading' plugins which means their import is deferred
- until they are needed. https://github.com/glue-viz/glue/pull/590
- 
- Support for connecting custom importers. https://github.com/glue-viz/glue/pull/593
- 
- `qglue` now correctly interprets HDUList objects. https://github.com/glue-viz/glue/pull/598
- 
- Internal improvements to organization of domain-specific code (such as the
- Astronomy coordinate conversions and ginga data viewer). [#488, #585]
- 
- Astronomy coordinate conversions now include more coordinate frames. https://github.com/glue-viz/glue/pull/578
- 
- `load_ui` now checks whether `.ui` file exists locally before
- retrieving it from the `glue.qt.ui` sub-package. https://github.com/glue-viz/glue/pull/599
- 
- Improved interface for adding new components, with syntax highlighting
- and tab-completion. [#572, #575]
- 
- Improved error/warning messages. https://github.com/glue-viz/glue/pull/582
- 
- Miscellaneous bug fixes. [#637, #636, #608]
- 
- The error console log is now available through the View menu
- 
- Improved under-the-hood handling of categorical ROIs. https://github.com/glue-viz/glue/pull/601
- 
- Fixed compatibility with Python 2.6. https://github.com/glue-viz/glue/pull/540
- 
- Python 3.x support is now stable. https://github.com/glue-viz/glue/pull/576
- 
- Fixed the ability to copy detailed error messages. https://github.com/glue-viz/glue/pull/675
- 
- Added instructions on how to make a fully-customized Qt viewer. https://github.com/glue-viz/glue/pull/619
- 
- Fixes to the ginga plugin to support the latest version. [#584, #656]
- 
- Added the ability to drag circular, rectangular, and lasso selections. https://github.com/glue-viz/glue/pull/657
- 
- Added the ability to reset a session. https://github.com/glue-viz/glue/pull/630
- 

## [v0.4.0](https://github.com/glue-viz/glue/releases/tag/v0.4.0) (Released December 22, 2015)

### Release Highlights:

- Introduced custom viewers
- Ginga-based image viewer
- Experimental Python 3.x support

### Other Notes

- Better testing for support of optional dependencies
- Refactored spectrum and position-velocity features from the Image widget into plugin tools
- Adopted contracts for contracters to add optional runtime type checking
- Added ability to export collapsed cubes as 2D fits files
- More flexible data parsing in qglue utility
- Numerous bugfixes
