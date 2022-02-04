Code organization
=================

The Glue code base is intended to be organized in a modular way, such that you
will never need to understand *all* the code in Glue, and the aim is for it to
be easy for you to identify where to make specific changes to implement the
functionality you need or fix issues.

Glue sub-packages
-----------------

The code is organized into the following
top-level sub-packages (starting with some of the easy ones):

:mod:`!glue.external`
^^^^^^^^^^^^^^^^^^^^^

This is a sub-package that you should never have to edit directly. It contains
files and modules edited in other repositories that have been bundled with
Glue. If you do need to make any changes to them, you should first edit those
other repositories, and then port over the changes to Glue. In general, it's
useful to know these bundled modules are available, but you will likely not
need to edit them.

:mod:`!glue.utils`
^^^^^^^^^^^^^^^^^^

This is a sub-package that contains various Python, Matplotlib, and Qt-related
utilities that do not depend on any other parts of Glue. These utilities don't
know about Glue data objects, subsets, or specific data viewers. Instead, this
sub-package includes utilities such as :func:`~glue.utils.geometry.points_inside_poly`,
a function to find whether points are inside a polygon, or
:func:`~glue.utils.qt.cmap2pixmap`, a function to convert a Matplotlib colormap
into a Qt ``QPixmap`` instance. This is one of the easiest sub-packages to
approach -- it is just a collection of small helper functions and classes and
doesn't require understanding any other parts of Glue.

:mod:`!glue.core`
^^^^^^^^^^^^^^^^^

As its name describes, this is the most important part of the Glue package.
This defines the general classes for datasets, subsets, data collections,
messages, layer artists, and other Glue concepts. On the other hand it does
*not* define specific viewers or data readers. The code in this sub-package is
not concerned with specific graphical user interface (GUI) representations, and
you could in principle develop a completely different GUI than the main Glue
one making use of the Glue core code. You could even use :mod:`!glue.core` to
give glue-like functionality to other existing applications.

:mod:`!glue.viewers`
^^^^^^^^^^^^^^^^^^^^

This sub-package contains the code for all the built-in viewers in glue, such
as the scatter plot and image viewers. Each viewer is contained in a
sub-package of :mod:`!glue.viewers`, such as :mod:`!glue.viewers.scatter`. A
:mod:`!glue.viewers.common` sub-package is also provided, with utilities and
base classes that might be useful for various viewers. For instance, the
:mod:`!glue.viewers.common.qt.toolbar_mode` sub-module contains code related to
defining toolbar mouse 'modes' for selection.

:mod:`!glue.dialogs`
^^^^^^^^^^^^^^^^^^^^

This sub-package contains implementations of various common dialogs, each
organized into sub-packages. For instance, :mod:`!glue.dialogs.custom_component`
contains the implementation of the dialog used to add new components to
datasets in the Glue application. The implementation for these dialogs only
uses the framework from the :mod:`!glue.core` package and the dialogs don't need
to know anything about the rest of the state of the design of the Glue
application.

.. :mod:`glue.core.data_factories`
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
..
.. While the core package defines the basic infrastructure for reading/writing
.. files, specific implementations of readers/writers live in
.. :mod:`glue.core.data_factories`. If you want to add a new reader or writer, this is
.. the place to put it!

:mod:`!glue.app`
^^^^^^^^^^^^^^^^

This package defines the Glue *Application*, that is the default GUI that users
interact with if they launch the Glue Application. This essentially pulls
together all the components from other sub-packages into a single application.
However, it would be entirely possible to develop other applications using the
available components - for instance, one could build an application with fixed
data viewers for a specific purpose.

:mod:`!glue.plugins`
^^^^^^^^^^^^^^^^^^^^

This package features more specialized tools/viewers for Glue, and in the long
term some of these will be moved into top-level sub-packages such as
``glue.viewers`` as they are made more general.

:mod:`!glue.icons`
^^^^^^^^^^^^^^^^^^

This contains various icons used in Glue, both in the vector SVG form, and in
rasterized PNG format.

.. _qt_code:

Qt-specific code
----------------

Glue currently uses the Qt GUI framework. However, this does not mean that you
need to know Qt to understand all of the code in Glue. Instead, we have taken
care to isolate all Qt-specific code into directories called ``qt/``. For
instance, the ``glue/utils/qt`` directory contains Qt-related utilities, and
any other code in ``glue/utils`` is not allowed to import Qt. We enforce this
while testing by making sure that all the tests in Glue run if all the ``qt/``
directories are removed, and no Qt implementation is installed.

Another example is that the ``glue/viewers/scatter/qt`` directory contains code
for the scatter plot viewer that is Qt-specific, but any other code in
``glue/viewers/scatter`` is Qt-agnostic. As a result, if you are trying to fix
something that is not related to the GUI, but to e.g. the data structures in
Glue, or the specific way in which e.g. Matplotlib displays something, you
shouldn't have to go into any of the ``qt`` sub-directories.

Another consequence of this is that if you or anyone else is interested in
developing a GUI front-end for Glue that is not based on Qt, you can re-use a
lot of the existing code that is not in the Qt directories. If we were to add
the code for another GUI framework into the Glue package, we could simply
create directories parallel to the ``qt`` directories but for the new framework.
