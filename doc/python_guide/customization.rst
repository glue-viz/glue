.. _customization:

Customizing Glue
================

There are a few ways to customize the Glue UI with configuration files and
plugins.

Configuration Files
-------------------

Each time Glue starts, it looks for and executes a configuration file. This
is a normal python script into which users can define or import new functions
to link data, plug in their own visualization modules, set up logging, etc.

The glue configuration file is called ``config.py``. Glue looks for this file
in the following locations, in order:

 * The current working directory
 * The path specified in the ``GLUERC`` environment variable, if present
 * The path ``.glue/config.py`` within the user's home directory

Registries
----------

Glue is written so as to allow users to easily register new plug-in data
viewers, tools, exporters, and more. Registering such plug-ins can be done
via *registries* located in the ``glue.config`` sub-package. Registries
include for example ``link_function``, ``data_factory``, ``colormaps``, and
so on. As demonstrated below, some registries can be used as decorators (see
e.g. `Adding Custom Link Functions`_) and for others you can add items using
the ``add`` method (see e.g. `Custom Colormaps`_).

In the following sections, we show a few examples of registering new
functionality, and a full list of available registries is given in `Complete
list of registries`_.

Adding Custom Link Functions
----------------------------
.. _custom_links:

From the :ref:`Link Data Dialog <getting_started_link>`, you inform Glue how
to convert between quantities among different data sets. You do this by
selecting a translation function, and specifying which data attributes should
be treated as inputs and outputs. You can use the configuration file to
specify custom translation functions. Here's how:

.. literalinclude:: scripts/config_link_example.py

Some remarks about this code:
 #. ``link_function`` is used as a `decorator <http://stackoverflow.com/questions/739654/understanding-python-decorators/1594484#1594484>`_. The decorator adds the function to Glue's list of link functions
 #. We provide a short summary of the function in the ``info`` keyword, and a list of ``output_labels``. Usually, only one quantity is returned, so ``output_labels`` has one element.
 #. Glue will always pass numpy arrays as inputs to a link function, and expects a numpy array (or a tuple of numpy arrays) as output

With this code in your configuration file, the ``deg_to_rad`` function is
available in the ``Link Data`` dialog:

.. figure:: images/custom_link.png
   :align: center
   :width: 200px

This would allow you to link between two datasets with different conventions
for specifying angles.

Custom Data Loaders
-------------------
.. _custom_data_factory:

Glue lets you create custom data loader functions,
to use from within the GUI.

Here's a quick example: the default image loader in Glue reads each color in
an RGB image into 3 two-dimensional components. Perhaps you want to be able
to load these images into a single 3-dimensional component called ``cube``.
Here's how you could do this::

 from glue.config import data_factory
 from glue.core import Data
 from skimage.io import imread

 def is_jpeg(filename, **kwargs):
     return filename.endswith('.jpeg')

 @data_factory('3D image loader', is_jpeg)
 def read_jpeg(file_name):
     im = imread(file_name)
     return Data(cube=im)

Let's look at this line-by-line:

* The `is_jpeg` function takes a filename and keywords as input,
  and returns True if a data factory can handle this file

* The ``@data_factory`` decorator is how Glue "finds" this function. Its two
  arguments are a label, and the `is_jpeg` identifier function

* The first line in ``read_jpeg`` uses scikit-image to load an image file
  into a NumPy array.

* The second line :ref:`constructs a Data object <data_creation>` from this
  array, and returns the result.

If you put this in your ``config.py`` file, you will see a new
file type when loading data:

  .. figure:: images/custom_data.png
     :align: center
     :width: 50%

If you open a file using this file type selection, Glue will pass the path of
this file to your function, and use the resulting Data object.

For more examples of custom data loaders, see the `example repository
<https://github.com/glue-viz/glue-data-loaders>`_.

Custom importers
----------------

The `Custom Data Loaders`_ described above allow Glue to recognize more file
formats than originally implemented, but it is also possible to write entire
new ways of importing data, including new GUI dialogs. An example would be a
dialog that allows the user to query and download online data.

Currently, an importer should be defined as a function that returns a list of
:class:`~glue.core.data.Data` objects. In future we may relax this latter
requirement and allow existing tools in Glue to interpret the data.

An importer can be defined using the ``@importer`` decorator::

    from glue.config import importer
    from glue.core import Data

    @importer("Import from custom source")
    def my_importer():
        # Main code here
        return [Data(...), Data(...)]

The label in the ``@importer`` decorator is the text that will appear in the
``Import`` menu in Glue.

Custom menubar tools
--------------------

In some cases, it might be desirable to add tools to Glue that can operate on
any aspects of the data or subsets, and can be accessed from the menubar. To
do this, you can define a function that takes two arguments (the session
object, and the data collection object), and decorate it with the
``@menubar_plugin`` decorator, giving it the label that will appear in the
**Tools** menubar::

    from glue.config import menubar_plugin

    @menubar_plugin("Do something")
    def my_plugin(session, data_collection):
        # do anything here
        return

The function can do anything, such as launch a QWidget, or anything else
(such as a web browser, etc.), and does not need to return anything (instead
it can operate by directly modifying the data collection or subsets).

Custom Colormaps
----------------

You can add additional matplotlib colormaps to Glue's image viewer by adding
the following code into ``config.py``::

    from glue.config import colormaps
    from matplotlib.cm import Paired
    colormaps.add('Paired', Paired)

Custom Subset Actions
---------------------

You can add menu items to run custom functions on subsets. Use the following
pattern in ``config..py``::

    from glue.config import single_subset_action

    def callback(subset, data_collection):
        print "Called with %s, %s" % (subset, data_collection)

    single_subset_action('Menu title', callback)

This menu item is available by right clicking on a subset when a single
subset is selected in the Data Collection window. Note that you must select
the subset specific to a particular Data set, and not the parent Subset Group.

Complete list of registries
---------------------------

A few registries have been demonstrated above, and a complete list of main
registries are listed below. All can be imported from ``glue.config`` - each
registry is an instance of a class, given in the second column, and which
provides more information about what the registry is and how it can be used.

========================== =======================================================
Registry name                  Registry class
========================== =======================================================
``qt_client``                :class:`glue.config.QtClientRegistry`
``tool_registry``            :class:`glue.config.QtToolRegistry`
``data_factory``             :class:`glue.config.DataFactoryRegistry`
``link_function``            :class:`glue.config.LinkFunctionRegistry`
``link_helper``              :class:`glue.config.LinkHelperRegistry`
``colormaps``                :class:`glue.config.ColormapRegistry`
``exporters``                :class:`glue.config.ExporterRegistry`
``settings``                 :class:`glue.config.SettingRegistry`
``fit_plugin``               :class:`glue.config.ProfileFitterRegistry`
``single_subset_action``     :class:`glue.config.SingleSubsetLayerActionRegistry`
========================== =======================================================

Deferring loading of plug-in functionality (advanced)
-----------------------------------------------------

In some cases, you may want to defer the loading of your plugin until it is
actually needed. To do this:

* Place the code for your plugin in a file or package that could be imported
  from the ``config.py`` (but don't import it directly - it just has to be
  importable)

* Include a function called ``setup`` alongside the plugin, and this function
  should contain code to actually add your custom tools to the appropriate
  registries.

* In ``config.py``, you can then add the plugin file or package to a registry
  by using the ``lazy_add`` method and pass a string giving the name of the
  package or sub-package containing the plugin.

Imagine that you have created a data viewer ``MyQtViewer``. You could
directly register it using::

    from glue.config import qt_client
    qt_client.add(MyQtViewer)

but if you want to defer the loading of the ``MyQtViewer`` class, you can
place the definition of ``MyQtViewer`` in a file called e.g.
``my_qt_viewer.py`` that is located in the same directory as your
``config.py`` file. This file should look something like::

    class MyQtViewer(...):
        ...

    def setup():
        from glue.config import qt_client
        qt_client.add(MyQtViewer)

then in ``config.py``, you can do::

    from glue.config import qt_client
    qt_client.lazy_add('my_qt_viewer')

With this in place, the ``setup`` in your plugin will only get called if the
Qt data viewers are needed, but you will avoid unecessarily importing Qt if
you only want to access ``glue.core``.

