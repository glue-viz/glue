.. _qglue:

Starting Glue from Python
=========================

In addition to using Glue as a standalone program, you can import glue
as a library from Python. There are (at least) two good reasons to do this:

#. You are working with multidimensional data in python, and want to use Glue
   for quick interactive visualization.
#. You find yourself repeatedly loading the same sets of data each time you
   run Glue. You want to write a startup script to automate this process.

Quickly send data to Glue with qglue
------------------------------------

The easiest way to send python variables to Glue is to use
:func:`~glue.qglue.qglue`::

    from glue import qglue

For example, say you are working with a `Pandas <http://pandas.pydata.org/>`_ DataFrame::

    >>> df
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 500 entries, 0 to 499
    Data columns (total 3 columns):
    x    500  non-null values
    y    500  non-null values
    z    500  non-null values
    dtypes: float64(3)

You can easily start up Glue with this data using::

    >>> app = qglue(xyz=df)

This will send this data to Glue, and label it ``xyz``.

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

.. note:: Reminder: in Glue, :class:`~glue.core.data.Data` sets are collections
          of one or more :class:`~glue.core.component.Component` objects.
          Components in a dataset are bascially arrays of the same shape. For
          more information, see :ref:`data_tutorial`

.. note:: Datasets cannot be given the label ``links``.

Linking data with ``qglue``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`Data Linking <linking>` tutorial discusses how Glue uses the
concept of links to compare different datasets. From the GUI, links
are defined using the :ref:`Link Manager <getting_started_link>`. It is
also possible to define some of these links with ``qglue``.

The ``links`` keyword for ``qglue`` accepts a list of link descriptions. Each link description has the following format::

    (component_list_a, component_set_b, forward_func, back_func)

* ``component_list_a`` and ``component_list_b`` are lists of component names.
  In the first example above, the ``x`` component in the ``xyz`` dataset is
  named ``'xyz.x'``.

* ``forward_func`` is a function which accepts one or more numpy arrays as
  input, and returns one or more numpy arrays as output. It computes the
  quantities in ``component_set_b``, given the quantities in
  ``component_list_a``.

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

The first link converts between the masses in two different data sets, recorded
in different units. The second link is a 1-way link that computes the area of
items in dataset 1, based on their width and height (there is no way to compute
the width and height from the area measurements in dataset 2, so the reverse
function is not provided). These links would enable the following interaction,
for example:

#. Overplot histograms of the mass distribution of both datasets
#. Define a region in a plot of mass vs area for data 2, and apply that filter
   to dataset 1

.. note:: If you start Glue from a non-notebook IPython session, you will
          encounter an error like ``Multiple incompatible subclass instances of
          IPKernelApp are being created``. The solution to this is to start
          Glue from a non-IPython shell, or from the notebook (see next
          section).

.. _notebook:

Using qglue with the IPython/Jupyter Notebook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can call :func:`~glue.qglue.qglue` from the IPython/Jupyter notebook
normally. However, the default behavior is for Glue to block the execution of
the notebook while the UI is running. If you would like to be able to use the
notebook and Glue at the same time, run this cell before starting glue::

    %gui qt

This must be executed in a separate cell, before starting Glue.

.. _add_data_qglue:

Adding data to glue when started using qglue
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once glue has been launched, you can continue to add data to it using the
:meth:`~glue.core.application_base.Application.add_data` method::

    >>> app = qglue(data1=array1)
    >>> app.add_data(data2=array2)

You can also pass filenames to :meth:`~glue.core.application_base.Application.add_data`::

    >>> app.add_data('myimage.fits')

Manual data construction
------------------------

If ``qglue`` is not flexible enough for your needs, you can build data objects
using the general Glue data API described in :ref:`data_tutorial`.

Here's a simple script to load data and pass it to Glue:

.. literalinclude:: scripts/w5.py

Some remarks:

 * :func:`~glue.core.data_factories.load_data` constructs Glue Data objects
   from files. It uses the file extension as a hint for file type
 * Individual data objects are bundled inside a
   :class:`~glue.core.data_collection.DataCollection`
 * The :class:`~glue.core.link_helpers.LinkSame` function indicates that two
   attributes in different data sets descirbe the same quantity
 * ``GlueApplication`` takes a ``DataCollection`` as input, and starts the GUI
   via ``start()``

Starting Glue from a script
---------------------------
.. _startup_scripts:

If you call glue with a python script as input, Glue will simply
run that script::

    $ glue startup_script.py

Likewise, if you are using the pre-built Mac application, you can right-click
on a script and open the file with Glue.
