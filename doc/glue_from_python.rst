.. _qglue:

Starting Glue from Python
=========================

In addition to using Glue as a standalone program, you can import glue
as a library from Python. There are (at least) two good reasons to do this:

 #. You are working with multidimensional data in python, and want to use Glue for quick interactive visualization.
 #. You find yourself repeatedly loading the same sets of data each time you run Glue. You want to write a startup script to automate this process.

Quickly send data to Glue with ``qglue``
----------------------------------------

The easiest way to send python variables to Glue is to use ``qglue``::

    from glue import qglue

For example, say you are working with a `Pandas <http://pandas.pydata.org/>`_ DataFrame::

    In [13]: df
    Out[13]:
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 500 entries, 0 to 499
    Data columns (total 3 columns):
    x    500  non-null values
    y    500  non-null values
    z    500  non-null values
    dtypes: float64(3)
    In [14]: qglue(xyz=df)

This will send this data to Glue, label it ``xyz`` and start the UI.

``qglue`` accepts many data types as inputs. Let's see some examples::

    import numpy as np
    import pandas as pd
    from astropy.table import Table

    x = [1, 2, 3]
    y = [2, 3, 4]

    u = [10, 20, 30, 40]
    v = [20, 40, 60, 80]

    pandas_data = pd.DataFrame({'x': x, 'y': y})
    dict_data = {'u': u, 'v': v}
    recarray_data = np.rec.array([(0, 1), (2, 3)],
                                 dtype=[('a', 'i'), ('b', 'i')])
    astropy_table = Table({'x': x, 'y': y})
    bad_data = {'x': x, 'u':u}


.. note::

   Terminology reminder: In Glue, *Data* sets are collections of one
   or more *components*. Components in a dataset are bascially arrays
   of the same shape. For more information, see the :ref:`Data API
   Tutorial <data_tutorial>`


* ``qglue(xy=pandas_data)``
 constructs a dataset labeled ``xy``, with two components (``x`` and ``y``)

* ``qglue(uv=dict_data)``
   construct a dataset labeled ``uv``, with two components (``u`` and ``v``)

* ``qglue(xy=pandas_data, uv=dict_data)``
 constructs both of the previous two data sets.

* ``qglue(rec=recarray_data, astro=astropy_table)``
 constructs two datasets: ``rec`` (components ``a`` and ``b``), and ``astro`` (components ``x`` and ``y``)

* ``qglue(bad=bad_data)``
 doesn't work, because the two components ``x`` and ``u`` have
 different shapes.

.. note:: Datasets cannot be given the label ``links``.

Linking data with ``qglue``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`Data Linking <linking>` tutorial discusses how Glue uses the
concept of links to compare different datasets. From the GUI, links
are defined using the :ref:`Link Manager <getting_started_link>`. It is
also possible to define some of these links with ``qglue``.

The ``links`` keyword for ``qglue`` accepts a list of link descriptions. Each link description has the following format::

    (component_list_a, component_set_b, forward_func, back_func)

* ``component_list_a`` and ``component_list_b`` are lists of component names. In the first example above, the ``x`` component in the ``xyz`` dataset is named ``'xyz.x'``.

* ``forward_func`` is a function which accepts one or more numpy arrays as input, and returns one or more numpy arrays as output. It computes the quantities in ``component_set_b``, given the quantities in ``component_list_a``.
* ``back_func`` performs the reverse operastion.

Here's an example::

   def pounds_to_kilos(lbs):
       return lbs / 2.2

   def kilos_to_pounds(kilos):
       return kilos * 2.2

   def lengths_to_area(width, height):
       return width * height

   link1 = (['data1.m_lb'], ['data_2.m_kg'], pounds_to_kilos, kilos_to_pounds)
   link2 = (['data1.width', 'data1.height'], ['data2.area'], lengths_to_area)
   qglue(data1=data1, data2=data2, links=[link1, link2])

The first link converts between the masses in two different data sets,
recorded in different units. The seonc link is a 1-way link that computes
an the area of items in dataset 1, based on their width and height (there is
no way to compute the width and height from the area measurements in dataset 2,
so the reverse function is not provided). These links would enable the following interaction, for example:
 #. Overplot histograms of the mass distribution of both datasets
 #. Define a region in a plot of mass vs area for data 2, and apply that filter to dataset 1


Manual data construction
------------------------
If ``qglue`` is not flexible enough for your needs, you can
build data objects using the general Glue data API.

Here's a simple script to load data and pass it to Glue:

.. literalinclude:: w5.py

Some remarks:
 * The ``load_data`` function constructs Glue Data objects from files. It uses the file extension as a hit for file type
 * Individual data objects are bundled inside a ``DataCollection``
 * The ``LinkSame`` function indicates that two attributes in different data sets descirbe the same quantity
 * ``GlueApplication`` takes a ``DataCollection`` as input, and starts the GUI via ``start()``

For more details on using Glue's data API, see the :ref:`Data Tutorial <data_tutorial>`

Starting Glue from a script
---------------------------
.. _startup_scripts:

If you call glue with a python script as input, Glue will simply
run that script::

 glue startup_script.py

Likewise, if you are using a pre-built application, you can right-click on a scipt and open the file with Glue.
