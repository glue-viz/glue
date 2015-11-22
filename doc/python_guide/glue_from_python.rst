.. _qglue:

Starting Glue with Python
=========================

In addition to using Glue as a standalone program, you can import glue
as a library from Python. There are (at least) two good reasons to do this:

 #. You are working with multidimensional data in python, and want to use Glue for quick interactive visualization.
 #. You find yourself repeatedly loading the same sets of data each time you run Glue. You want to write a startup script to automate this process.

Quickly send data to Glue with :func:`~glue.qglue.qglue`
--------------------------------------------------------

The easiest way to send python variables to Glue is to use
:func:`~glue.qglue.qglue`::

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

:func:`~glue.qglue.qglue` accepts many data types as inputs. Let's see some examples::

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

   Terminology reminder: In Glue, :class:`~glue.core.data.Data` sets are collections of one or more :class:`~glue.core.component.Component` objects.
   Components in a dataset are bascially arrays of the same shape. For more information, see :ref:`data_tutorial`


* ``qglue(xy=pandas_data)``:
   constructs a dataset labeled ``xy``, with two components (``x`` and ``y``)

* ``qglue(uv=dict_data)``:
   construct a dataset labeled ``uv``, with two components (``u`` and ``v``)

* ``qglue(xy=pandas_data, uv=dict_data)``:
   constructs both of the previous two data sets.

* ``qglue(rec=recarray_data, astro=astropy_table)``:
   constructs two datasets: ``rec`` (components ``a`` and ``b``), and
   ``astro`` (components ``x`` and ``y``)

* ``qglue(bad=bad_data)``:
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
recorded in different units. The second link is a 1-way link that computes
the area of items in dataset 1, based on their width and height (there is
no way to compute the width and height from the area measurements in dataset 2,
so the reverse function is not provided). These links would enable the following interaction, for example:

 #. Overplot histograms of the mass distribution of both datasets
 #. Define a region in a plot of mass vs area for data 2, and apply that filter to dataset 1

Using qglue with the IPython Notebook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can call :func:`~glue.qglue.qglue` from the IPython notebook normally. However, the default behavior is for Glue to block the execution of the
notebook while the UI is running. If you would like to be able to use the notebook and Glue at the same time, run this cell before starting glue::

    %gui qt

This must be executed in a separate cell, before starting Glue.

Manual data construction
------------------------
If ``qglue`` is not flexible enough for your needs, you can
build data objects using the general Glue data API.

Here's a simple script to load data and pass it to Glue:

.. literalinclude:: scripts/w5.py

Some remarks:
 * :func:`~glue.core.data_factories.load_data` constructs Glue Data objects from files. It uses the file extension as a hint for file type
 * Individual data objects are bundled inside a
   :class:`~glue.core.data_collection.DataCollection`
 * The :class:`~glue.core.link_helpers.LinkSame` function indicates that two attributes in different data sets descirbe the same quantity
 * ``GlueApplication`` takes a ``DataCollection`` as input, and starts the GUI via ``start()``

For more details on using Glue's data API, see the :ref:`Data Tutorial <data_tutorial>`

Starting Glue from a script
---------------------------
.. _startup_scripts:

If you call glue with a python script as input, Glue will simply
run that script::

 glue startup_script.py

Likewise, if you are using a pre-built application, you can right-click on a scipt and open the file with Glue.

Interacting with Glue using Python
==================================

There are two ways to programmatically interact with an active Glue session. We outline each option below, and then describe some
useful ways to interact with Glue using Python.

The Glue-IPython terminal
-------------------------

Glue includes a button to open an IPython terminal window. This gives
you programmatic access to Glue data. A number of variables are available
by default (these are also listed when you open the terminal):

  * ``dc`` / ``data_collection`` refer to the central
    :class:`~glue.core.data_collection.DataCollection`, which holds all of the
    datasets, subsets, and data links

  * ``hub`` is the communication :ref:`hub <hub>` object.

  * ``application`` is the top level :class:`~glue.qt.glue_application.GlueApplication`, which has access to plot windows (among other things)

Additionally, you can drag datasets and subsets into the terminal
window, to easily assign them new variable names.

.. note::

    If you start Glue from a non-notebook IPython session, you will
    encounter an error like ``Multiple incompatible subclass instances of IPKernelApp are being created``. The
    solution to this is to start Glue from a non-IPython shell,
    or from the notebook (see above).


Notebook integration
--------------------
As described above, the IPython notebook can be configured so that
Glue runs without blocking. When launched via :func:`~glue.qglue.qglue`,
that function immediately returns a reference to the
:class:`~glue.qt.glue_application.GlueApplication` object.

Usage Examples
--------------

Adding new attributes to datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A common task is to combine two or more attributes in a dataset,
and store the result as a new attribute to visualize. Let's use
the catalog data from the :ref:`Getting Started <getting_started>` as an example.

First, we need to grab the relevant dataset from the data collection.
From the Glue-IPython window, we can do this simply by dragging the
dataset onto the window. If we want to do this from what is
returned by :func:`~glue.qglue.qglue`, it would look like this::

    print app.data_collection  # look at each dataset
    catalog = data_collection[2]  # or whichever entry it is

To examine the attributes in this dataset::

    In [1]: print catalog
    Data Set: catalogNumber of dimensions: 1
    Shape: 17771
    Components:
     0) ID
     1) Pixel Axis 0
     2) World 0
     3) Jmag
     4) Hmag
     5) Ksmag
     ...

Datasets behave like dictionaries mapping component names to numpy arrays. So one way to define a new component is like this::

    In [2]: j_minus_h = catalog['Jmag'] - catalog['Hmag']

To add this back to the dataset::

    In [3]: catalog['jmh'] = j_minus_h

This new attribute is now available for visualizing in the GUI

Adding lazy attributes
^^^^^^^^^^^^^^^^^^^^^^

In the procedure above, the `j_minus_h` array was precomputed.
An alternative approach is to define a new attribute on the fly.
While ``data[attribute_name]`` returns a numpy array, ``data.id[attribute_name]`` returns a lightweight proxy object that you can
use to build simple arithmetic expressions::

    In [4]: jmh_lazy = catalog.id['Jmag'] - catalog.id['Hmag']
    In [5]: jmh_lazy
    <BinaryComponentLink: (Jmag - Hmag)>
    In [6]: catalog['jmh2'] = jmh_lazy

This new component is computed as needed on the fly, and can
be more memory efficient for particular applications.


Defining new subsets
^^^^^^^^^^^^^^^^^^^^

You can define new subsets from Python. An example might look like::

    state = catalog.id['j'] > catalog.id['h']
    label = 'J > H'
    sg = data_collection.new_subset_group(label, state)
    sg.style.color = '#00ff00'

This can be a powerful technique. For a demo of using sending
Scikit-learn-identified clusters back into Glue as subsets, see `this
notebook <http://nbviewer.ipython.org/github/ChrisBeaumont/crime/blob/master/glue_startup.ipynb>`_.

The following example demonstrates how to use it to build custom subsets that
would be cumbersome to define manually. We will be using the W5 Point Source
catalog from the :ref:`tutorial <getting_started>`.

We also define a few subsets to play with. Our setup looks like this.

.. image:: images/subset_01.png
   :width: 60%

Click the terminal button next to the link data button to open the terminal window.

Assign variables to the two subsets defined for this data collection::

    >>> red, faint_h = data_collection.subset_groups

Let's also grab a component in the data::

    >>> catalog = data_collection[0]
    >>> hmag = catalog.id['Hmag']

To find the intersection of the two subsets we have already defined
(i.e., red sources with faint H band magnitudes)::

   >>> new_state = red & faint_h
   >>> label = "Red and faint"
   >>> data_collection.new_subset_group(label=label, subset_state=new_state)

The resulting intersection is shown in blue here:

.. image:: images/subset_02.png
   :width: 60%

The boolean operators ``&``, ``^``, ``|``, and ``~`` act on subsets to
define new subsets represented by the intersection, exclusive
union, union, and inverse, respectively.

You can also build subsets out of inequality constraints on component IDs::

   >>> mid_mag = (hmag > 10) & (hmag < 15)
   >>> data_collection.new_subset_group(subset_state=mid_mag)

This selects objects with H band magnitudes between 10 and 15:

.. image:: images/subset_03.png
   :width: 60%

