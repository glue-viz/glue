Changelog
=========

v0.5 (Unreleased)
-----------------

* Added an option to bundle all data in .glu session files
* Added a ``menu_plugin`` registry to add custom tools to the registry
* Support for 'lazy-loading' plugins which means their import is deferred until they are needed
* Support for connecting custom importers
* ``qglue`` now correctly interprets HDUList objects 
* Internal improvements to organization of domain-specific code (such as the Astronomy coordinate conversions and ginga data viewer)
* ``load_ui`` now checks whether ``.ui`` file exists locally before retrieving it from the ``glue.qt.ui`` sub-package
* Improved interface for adding new components, with syntax highlighting and tab-completion
* Improved error/warning messages and miscellaneous bug fixes
* The error console log is now availble through the View menu
* Fixed compatibility with Python 2.6
* Python 3.x support is now stable
* Fixed the ability to copy detailed error messages

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
