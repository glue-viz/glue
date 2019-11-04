Working with non-glue data objects
==================================

The main type of data object in glue are represented by the
:class:`~glue.core.data.Data` class. In some cases, you may however want to be
able to convert between these objects and other data classes (such as pandas
DataFrames). In addition, you may want to be able to convert selections in glue
to other kinds of data objects. Glue now includes a command-line interface to
make such translations as seamless as possible.

The functionality shown below is only possible for a limited set of non-glue
data classes in the core glue package, but glue plugins may add support for
other kinds of data objects. If you want to develop your own translation
functions, you can do this as described in :ref:`custom-data-translation`
and :ref:`custom-subset-translation`.

Converting data objects
-----------------------

There are several ways of getting non-glue data objects into glue. To illustrate
this, we use a pandas :class:`~pandas.DataFrame` as an example::

    >>> from pandas import DataFrame
    >>> df1 = DataFrame()
    >>> df1['a'] = [1.2, 3.4, 2.9]
    >>> df1['g'] = ['r', 'q', 's']
    >>> df1
         a  g
    0  1.2  r
    1  3.4  q
    2  2.9  s

We now create a glue :class:`~glue.core.data_collection.DataCollection` object (note that if you
are already using an active glue application, you do not need to do this, and
instead you can access the data collection using the ``.data_collection``
attribute of the application::

    >>> from glue.core import DataCollection
    >>> dc = DataCollection()
    >>> dc
    DataCollection (0 data sets)

At this point, you can add the data to the data collection by assigning it to an
item with the label you want to give the dataset::

    >>> dc['dataframe'] = df1
    >>> dc
    DataCollection (1 data set)
    	  0: dataframe

Note that ``dc.append`` won't work here because it only works with glue
:class:`~glue.core.data.Data` objects. We can see that the data collection now has
one dataset. We can access this data either by index::

    >>> dc[0]
    Data (label: dataframe)

or by label::

    >>> dc['dataframe']
    Data (label: dataframe)

Note that in both cases, this returns a glue :class:`~glue.core.data.Data` object::

    >>> print(dc['dataframe'])
    Data Set: dataframe
    Number of dimensions: 1
    Shape: 3
    Main components:
     - a
     - g
    Coordinate components:
     - Pixel Axis 0 [x]

To get back the kind of object that you put in, you need to call the
:meth:`~glue.core.data.BaseData.get_object` method::

    >>> df2 = dc['dataframe'].get_object()
    >>> type(df2)
    <class 'pandas.core.frame.DataFrame'>
    >>> df2
         a  g
    0  1.2  r
    1  3.4  q
    2  2.9  s

In this case, glue knew to return a :class:`~pandas.DataFrame` object by default
because this is what was used to initialize the data object. However, you can
also specify this explicitly, either to convert to a different kind of object,
or to convert a glue data object that was not initially created from a
:class:`~pandas.DataFrame` to a :class:`~pandas.DataFrame`::

    >>> from glue.core import Data
    >>> data = Data(label='simple')
    >>> data['f'] = [21, 45, 56]
    >>> df3 = data.get_object(cls=DataFrame)
    >>> type(df3)
    <class 'pandas.core.frame.DataFrame'>
    >>> df3
        f
    0  21
    1  45
    2  56

To see what data classes are currently supported for the translation, you can
call :meth:`~glue.core.data.BaseData.get_object` with no arguments:

    >>> data.get_object()
    Traceback (most recent call last):
    ...
    ValueError: Specify the object class to use with cls= - supported classes are:
    <BLANKLINE>
    * pandas.core.frame.DataFrame

The core glue application only supports translations with :class:`~pandas.DataFrame`
for now, but plugin packages may define translations to other domain-specific
data objects.

Working with subsets
--------------------

In the examples above, we saw how to translate certain kinds of non-glue objects
to glue objects, and translate these back. In some cases, you may want to
translate not the full dataset but a subset of the data back to a non-glue object.
For example, you may have passed a :class:`~pandas.DataFrame` to glue, made
a series of selections, and want to get  the subset of points in the selection to
a :class:`~pandas.DataFrame`. Continuing from the prevous example where
the data collection contains a single dataset created from a :class:`~pandas.DataFrame`::

    >>> dc
    DataCollection (1 data set)
    	  0: dataframe

We now make a selection based on the data values (here we make the selection
programmatically, but often you may be making it interatively in the data
viewers)::

    >>> dc.new_subset_group(subset_state=dc['dataframe'].id['a'] < 3,
    ...                     label='my subset')
    <glue.core.subset_group.SubsetGroup object at ...>

Now that the subset has been created, you can retrieve it as a :class:`~pandas.DataFrame`
using the :meth:`~glue.core.data.BaseData.get_subset_object` method::

    >>> dfsub1 = dc['dataframe'].get_subset_object()
    >>> type(dfsub1)
    <class 'pandas.core.frame.DataFrame'>
    >>> dfsub1
         a  g
    0  1.2  r
    1  2.9  s

Generally speaking, for datasets with 1-d fields, the translation functions will create
an object which has a subset of the original rows. For datasets with 2 or more dimensions,
the final dataset may have the same shape but with the values outside of the subset masked,
e.g. by NaN values. This behavior is left up to the individual translation functions.

If multiple subsets are present, you can specify which one to retrieve using the ``subset_id``
keyword argument::

    >>> dc.new_subset_group(subset_state=dc['dataframe'].id['a'] > 2,
    ...                     label='my other subset')
    <glue.core.subset_group.SubsetGroup object at ...>
    >>> dfsub2 = dc['dataframe'].get_subset_object(subset_id='my other subset')
    >>> dfsub2
         a  g
    0  3.4  q
    1  2.9  s

or you can also not set ``subset_id`` to see a list of available subsets::

    >>> dc['dataframe'].get_subset_object()
    Traceback (most recent call last):
    ...
    ValueError: Several subsets are present, specify which one to retrieve with subset_id= - valid options are:
    <BLANKLINE>
    * 0 or 'my subset'
    * 1 or 'my other subset'

Translating the definition of the subsets
-----------------------------------------

In the previous section on `Working with subsets`_, we translated the data in the glue
subsets to non-glue objects, but it is also possible to translate not the data values
but the more abtract representation of the selection. The core glue package does not
include any formats to translate these selections to currently, but if you have a
glue plugin installed that does, or if you have defined one yourself as described
in :ref:`custom-subset-translation`, you should be able to call the
:meth:`~glue.core.data.BaseData.get_selection_definition` method which takes a ``format=``
keyword argument that specifies the translator to use (leaving this out will
show a list of available of translation functions available as for subsets)::

    >>> dc['dataframe'].get_selection_definition(subset_id='my subset',
    ...                                          format='simple-string')  # doctest: +SKIP
    "a > 2"

