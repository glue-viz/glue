.. _customization:

Customizing Glue
================
There are a few ways to customize the Glue UI
with configuration files and plugins.


Configuration Files
-------------------

Each time Glue starts, it looks for an executes a configuration file. This is a normal python script into which users can define or import new functions to link data, plug in their own visualization modules, set up logging, etc.

The glue configuration file is called ``config.py``. Glue looks for this file in the following locations, in order:

 * The current working directory
 * The path specified in the ``GLUERC`` environment variable, if present
 * The path ``.glue/config.py`` within the user's home directory

Adding Custom Link Functions
----------------------------
.. _custom_links:

From the :ref:`Link Data Dialog <getting_started_link>`, you inform Glue how to convert between quantities among different data sets. You do this by selecting a translation function, and specifying which data attributes should be treated as inputs and outputs. You can use the configuration file to specify custom translation functions. Here's how:

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


Custom Data Loaders
-------------------
.. _custom_data_factory:

Glue lets you create custom data loader functions,
to use from within the GUI.

Here's a quick example: the default image loader in Glue
reads each color in an RGB image into 3 two-dimensional components.
Perhaps you want to be able to load these images into a single 3-dimensional
component called ``cube``. Here's how you could do this::

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

* The ``@data_factory`` decorator is how Glue "finds" this function.
   Its two arguments are a label, and the `is_jpeg` identifier function

* The first line in ``read_jpeg`` uses scikit-image to load an image file into a NumPy array.

* The second line :ref:`constructs a Data object <data_creation>` from this array, and returns the result.

If you put this in your ``config.py`` file, you will see a new
file type when loading data:

  .. figure:: custom_data.png
     :align: center
     :width: 50%

If you open a file using this file type selection, Glue will pass
the path of this file to your function, and use the resulting Data
object.

For more examples of custom data loaders, see the `example repository <https://github.com/glue-viz/glue-data-loaders>`_.

Custom Colormaps
----------------
You can add additional matplotlib colormaps to Glue's image viewer by adding the following code into ``config.py``::
    
    from glue.config import colormaps
    from matplotlib.cm import Paired
    colormaps.add('Paired', Paired)
