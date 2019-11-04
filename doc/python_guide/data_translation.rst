Working with non-glue data objects
==================================

The main type of data object in glue are represented by the
:class:`~glue.core.data.Data` class. In some cases, you may however want to be
able to convert between these objects and other data classes (such as pandas
DataFrames). In addition, you may want to be able to convert selections in glue
to other kinds of data objects. Glue now includes a command-line interface to
make such translations as seamless as possible.

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

We now create a glue :class:`~glue.core.DataCollection` object (note that if you
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
:class:`~glue.core.Data` objects. We can see that the data collection now has
one dataset. We can access this data either by index::

    >>> dc[0]
    Data (label: dataframe)

or by label::

    >>> dc['dataframe']
    Data (label: dataframe)

Note that in both cases, this returns a glue :class:`~glue.core.Data` object::

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
:meth:`~glue.core.Data.to_object` method::

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
call :meth:`~glue.core.Data.to_object` with no arguments:

    >>> data.get_object()
    Traceback (most recent call last):
    ...
    ValueError: Specify the object class to use with cls= - supported classes are:
    <BLANKLINE>
    * pandas.core.frame.DataFrame

The core glue application only supports translations with :class:`~pandas.DataFrame`
for now, but plugin packages may define translations to other domain-specific
data objects.