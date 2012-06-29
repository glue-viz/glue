Working with Data Objects
==========================

.. currentmodule:: glue.data

If you would like to extend or customize Glue's functionality, you will
likely need to deal with managing the :class:`Data` object. This document walks through handling data in Glue.

The basic hierarchy of data objects in Glue looks like this:

.. image:: glue_hierarchy.png
   :width: 300
   :alt: Glue Hierarchy


* :class:`Component`
    Each :class:`Component` object stores a numpy
    array -- this is where the actual, numerical information resides.

* :class:`Data`
    The :class:`Data` object stores (among other things) one or more
    components. Each Component stores the information about one
    quantity. In a catalog, each column is a distinct Component. All
    of the Components stored with a Data object must have numpy arrays
    with the same shape. Components are referenced in the Data via
    :class:`ComponentID` objects.

* :class:`~glue.DataCollection`
    The DataCollection stores one or more Data sets. If you want to
    examine multiple files in a Glue session, you load each file into
    a different Data object and store them in a DataCollection


.. note::

 It's somewhat unfortunate that the actual "data" (i.e. the numpy
 array in a component object) is so deeply buried. This structure
 helps Glue keep track of other information more easily.


Creating objects
----------------

If you need to create your own data objects, the code looks something
like this::

   import glue
   import numpy as np
   data = glue.Data(label="first dataset")
   x = glue.Component( np.array([1, 2, 3]))
   y = glue.Component( np.array([4, 5, 6]))
   x_id = data.add_component(x, label = 'X')
   y_id = data.add_component(y, label = 'Y')
   collection = glue.DataCollection()
   collection.append(data)

Registering with a Hub
----------------------

Simply register the :class:`~glue.DataCollection` to the hub -- all the child objects will be auto-subscribed::

   collection.register_to_hub(hub)


Retrieving Values
-----------------

Components are extracted from data by their componentIDs::

    comp = data.get_component(x_id)
    comp == x  # True
    comp.data  # array([1, 2, 3])

To save typing, ``data[component_id]`` fetches the numerical data directly::

   print data[y_id]  # array([4, 5, 6])

To see what ComponentIDs are stored with a data set::

   print data.components

To search by label::

   data.find_component_id('X')   # [component_id]
   data.find_componenet_id('Z')  # []

To fetch the data stored in the collection::

   print collection.data
   for d in collection:
       print d

Working with Files
------------------

There are a few classes to help you create :class:`Data` objects from files::

   catalog_data = glue.TabularData()
   catalog_data.read_data(catalog_filename, *args, **kwargs)

This creates a data object from a catalog, using `ATpy <http://atpy.github.com/>`_ for table parsing. Extra arguments to ``_read_data`` are passed to ATpy's ``read`` method


For loading FITS and HDF5 images or cubes::

   image_data = glue.GriddedData()
   image_data.read_data(file_name, format=['hdf5' | 'fits'])


