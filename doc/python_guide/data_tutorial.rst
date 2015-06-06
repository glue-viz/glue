.. _data_tutorial:
.. currentmodule:: glue.core.data

Working with Data
=================

If you are writing Python code that uses Glue, you will probably want work
with data. The hierarchy of data objects in Glue looks like this:

.. image:: images/glue_hierarchy.png
   :width: 300
   :alt: Glue Hierarchy


* :class:`Component`
    Each :class:`Component` object stores a numpy
    array -- this is where the actual, numerical information resides.

* :class:`Data`
    The :class:`Data` object stores (among other things) one or more
    components. It is dictionary-like, and it maps
    :class:`ComponentIDs <ComponentID>` to :class:`Components <Component>`.
    It also stores references to the :class:`Subsets <glue.core.subset.Subset>` of the data via ``data.subsets``.

* :class:`~glue.core.data_collection.DataCollection`
    The DataCollection stores one or more Data sets. If you want to
    examine multiple files in a Glue session, you load each file into
    a different Data object and store them in a DataCollection


.. _data_creation:

Building :class:`Data` objects
------------------------------

Building from scratch::

   from glue.core import Data, DataCollection
   import numpy as np

   # pass components in as keywords...
   data = Data(x=[1, 2, 3], label="first dataset")

   # or add as components after creation...
   y_id = data.add_component([4, 5, 6], label = 'Y')

   collection = DataCollection([data])

The functions in :mod:`glue.core.data_factories` create
:class:`Data` objects from files::

    from glue.core.data_factores import *
    load_data('image.fits', factory=gridded_data)  # reads a fits image
    load_data('catalog.csv', factory=tabular_data) # reads a catalog
    load_data('catalog.csv')  # guesses factory

If these functions do not fit your needs, you can also :ref:`write your own
data loader <custom_data_factory>`, and use it from the Glue GUI.


.. _data_access_api:
.. currentmodule :: glue.core

Using :class:`~data.Data` and :class:`~data_collection.DataCollection`
----------------------------------------------------------------------

:class:`~data_collection.DataCollection` behaves like a list -- you can access :class:`~data.Data` objects by indexing into it::

    In [1]:  dc
    Out[1]:
    DataCollection (2 data sets)
          0: w5
          1: w5_psc

    In [2]: dc[0]
    Out[2]: Data (label: w5)

This DataCollection has two data sets. Let's grab the first one::

    In [3]: data = dc[0]

    In [4]: data.components
    Out[4]: [PRIMARY, Pixel y, Pixel x, World y: DEC--TAN, World x: RA---TAN]

:class:`~data.Data` objects behave like dictionaries: you can retrieve the numerical data associated with each one with bracket-syntax::

    In [5]: data['PRIMARY']
    ... a numpy array ...

Numpy-style fancy-indexing is also supported::

    In  [6]: data['PRIMARY', 0:3, 0:2]
    Out [6]:
    array([[ 454.47747803,  454.18780518],
           [ 452.36376953,  452.8883667 ],
           [ 451.77172852,  453.42767334]], dtype=float32)

Note that this syntax gives you the numpy array, and not the Component object itself. This is usually what you are interested in. However, you can retrieve the Component object if you like with ``get_component``::

    In [6]: primary_id = data.components[0]

    In [7]: print primary_id, type(primary_id)
    Out[7]: PRIMARY <class 'glue.core.data.ComponentID'>

    In [8]: component = data.get_component(primary_id)  #component object
    In [9]: component.data   # numpy array

.. note:: The item access syntax (square brackets) will not work if component labels are not unique. In this case, you must first retrieve the wanted ComponentID (In[6]) and use it to get the component object (In[8]).
