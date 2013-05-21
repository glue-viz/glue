.. _scripting_interface

Scripting Glue
==============

Glue is written in Python, and can be extended by a variety of mechanisms. If you use Python in other contexts, many of these concepts will be familiar to you.

Configuration Files
-------------------
Each time Glue starts, it looks for an executes a configuration file. This is a normal python script into which users can define or import new functions to link data, plug in their own visualization modules, set up logging, etc.

The glue configuration file is called ``config.py``. Glue looks for this file in the following locations, in order:

 * The current working directory
 * The path specified in the ``GLUERC`` environment variable, if present
 * The path``glue/config.py`` within the user's home directory

Adding Custom Link Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the Link Data dialog, you inform Glue how to convert between quantities within or among different data sets. You do this by selecting a translation function, and specifying which data attributes should be treated as inputs and outputs. You can use the configuration file to specify custom translation functions. Here's how:

.. literalinclude:: config_link_example.py

Some remarks about this code:
 #. ``link_function`` is used as a `decorator <http://stackoverflow.com/questions/739654/understanding-python-decorators/1594484#1594484>`_. The decorator adds the function to Glue's list of link functions
 #. We provide a short summary of the function in the ``info`` keyword, and a list of ``output_labels``. Usually, only one quantity is returned, so ``output_labels`` has one element.
 #. Glue will always pass numpy arrays as inputs to a link function, and expects a numpy array (or a tuple of numpy arrays) as output

With this code in your configuration file, the ``deg_to_rad`` function is available in the ``Link Data`` dialog:

.. figure:: custom_link.png
   :align: center
   :width: 200px

This would allow you to link between two datasets with different conventions for specifying angles.

.. todo:: Add descriptions for adding custom clients and data factories


Starting Glue from a script
---------------------------
Startup scripts provide a particularly powerful way to start Glue. Such scripts would load data files, define the logical connections,
and send this information to Glue. This has several benefits:

 #. In a research project, the same files are visualized several times. The startup script saves you from having to load and link data every time
 #. A script allows you to perform data pre-processing or cleaning before visualization.
 #. It provides a human-readable description of how the data have been loaded and linked. Startup scripts are better documentation than saved sessions
 #. Unlike a saved session, the startup script can evolve over time.

Startup scripts are normal python files that import the Glue module. They can be started from the command line::

 python startup_script.py
 glue startup_script.py

Or, if using a pre-built application, by right-clicking on the script icon and telling the operating system to Open the file with Glue.

Here's a simple script to load data and pass it to Glue:

.. literalinclude:: w5.py

Some remarks:
 * The ``load_data`` function constructs Glue Data objects from files. It uses the file extension as a hit for file type
 * Individual data objects are bundled inside a ``DataCollection``
 * The ``LinkSame`` function indicates that two attributes in different data sets descirbe the same quantity
 * ``GlueApplication`` takes a ``DataCollection`` as input, and starts the GUI via ``start()``

The Terminal Window
-------------------

The terminal window provides an IPython shell with access to the
variables used by Glue. You can use this to inspect the state of Glue
as it runs, analyze data in place, etc.

The main variables available by default form the termainal
window are ``data_collection``, ``dc`` (an alias to
``data_collection``), and ``hub``.

In addition, you can drag individual datasets or subsets into the terminal,
and assign them to local variables.

Essential API
-------------
Here are some of the most common parts of the Glue API, organized by task.

Loading Data
^^^^^^^^^^^^
The easiest way to construct glue Data objects is with the :func:`~glue.core.data_factories.load_data` function::

 from glue.core.data_factories import load_data
 data = load_data('path_to_file')

:func:`~glue.core.data_factories.load_data` takes a path to a file, and tries to parse its contents (including metadata) into a data object. It uses the file extension to guess how to load the file. If the file extension is non-standard or ambiguous, you can pass a specific data factory as a second argument. For example, :func:`~glue.core.data_factories.load_data` assumes that files ending in ``.fits`` are images, but they may be tables. This code uses Glue's table parser explicitly::

  from glue.core.data_factories import load_data, tabular_data
  data = load_data('table.fits', tabular_data)

Creating Data
^^^^^^^^^^^^^

The next easiest way to create a Data object is to pass a set arrays to the constructor to :class:`~glue.core.data.Data`::

   from glue.core import Data
   d = Data(label='Custom data', x=[1, 2, 3], y=[2, 3, 4], z=[3, 4, 5])

This creates a data object with 3 components, labeled x, y, and z::

   >>> d['x']
   [1, 2, 3]

.. note:: All components within a :class:`~glue.core.data.Data` instance must have the same shape


If you need to add components to a data set after it is initialized, use the ``add_component`` method::

   d = Data(label='Custom Data', x=[1, 2, 3])
   d.add_component([3, 4, 5], label='z')


.. todo:: Describe creating Coordinate objects


Custom Data Loaders
^^^^^^^^^^^^^^^^^^^
.. _custom_data_factory:
Glue let's you create custom data loader functions,
to use from within the GUI.

Here's a quick example: the default image loader in Glue
reads each color in an RGB image into 3 two-dimensional components.
Perhaps you want to be able to load these images into a single 3-dimensional
component called ``cube``. Here's how you could do this::

 from glue.config import data_factory
 from glue.core import Data
 from skimage.io import imread

 @data_factory('3D image loader', '*.jpg *.bmp *.tiff')
 def read_jpeg(file_name):
     im = imread(file_name)
     return Data(cube=im)

Let's look at this line-by-line:

 * The ``@data_factory`` decorator is how Glue "finds" this function.
   Its two arguments are a label, and the file formats it can open.

 * The first line in ``read_jpeg`` uses scikit-image to load an image
   file into a NumPy array.

 * The second line constructs a glue Data object from this array (using
   the syntax described in the previous section), and returns the result.

If you put this in your ``config.py`` file, you will see a new
file type when loading data:

  .. figure:: custom_data.png
     :align: center
     :width: 50%



Linking Data
^^^^^^^^^^^^

Creating Subsets
^^^^^^^^^^^^^^^^^

Custom File Formats
^^^^^^^^^^^^^^^^^^^^
