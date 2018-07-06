.. _dev_selection:

The selection/subset framework
==============================

One of the central concepts in Glue is that of subsets, which are typically
created as a result of the user selecting data in a viewer or creating the
subset from the command-line. In order to go from a selection on the screen to
defining a subset from a dataset, Glue includes the following concepts:

* **Region of interests** (ROIs), which are an abstract representation of a
  geometrical region or selection.

* **Subset states**, which is a descriptions of the subset selection.

* Data **Subsets**, which are the result of applying a subset state/selection
  to a specific dataset.

When a user makes a selection in a data viewer in the Glue application, the
selection is first translated into a ROI, after which the ROI is converted to a
subset state, then applied to the data collection to produce subsets in each
dataset. These three concepts are described in more detail below.

Regions of interest
-------------------

The easiest way to think of regions of interest is as geometrical regions.
Basic classes for common types of ROIs are included in the :mod:`glue.core.roi`
sub-module. For example, the :mod:`~glue.core.roi.RectangularROI` class
describes a rectangular region using the lower and upper values in two
dimensions::

    >>> from glue.core.roi import RectangularROI
    >>> roi = RectangularROI(xmin=1, xmax=3, ymin=2, ymax=5)

Note that this is not related to any particular dataset -- it is an abstract
representation of a rectangular region. It also doesn't specify which
components the rectangle is drawn in. All ROIs have a
:meth:`glue.core.roi.RectangularROI.contains` method that can be used to check
if a point or a set of points lies inside the region::

    >>> roi.contains(0, 3)
    False
    >>> roi.contains(2, 3)
    True
    >>> import numpy as np
    >>> x = np.array([0, 2, 4])
    >>> y = np.array([3, 3, 2])
    >>> roi.contains(x, y)
    array([False,  True, False], dtype=bool)

Subset states
-------------

While regions of interest define geometrical regions, subset states, which are
sub-classes of :class:`~glue.core.subset.SubsetState`, describe a selection as
a function of Glue :class:`~glue.core.component_id.ComponentID` objects. Note
that this is different from :class:`~glue.core.subset.Subset` instances, which
describe the subset *resulting* from the selection (see `Subsets`_). The
following simple example shows how to easily create a
:class:`~glue.core.subset.SubsetState`::


    >>> from glue.core import Data
    >>> data = Data(x=[1,2,3], y=[2,3,4])
    >>> state = data.id['x'] > 1.5
    >>> state
    <InequalitySubsetState: (x > 1.5)>

Note that ``state`` is not the subset of values in ``data`` that are greater
than 1.5 -- instead, it is a representation of the inequality, the *concept* of
selecting all values of x greater than 1.5. This distinction is important,
because if another dataset defines a link between one of its components and the
``x`` component of ``data``, then the inequality can be used for that other
component too.

While the above syntax is convenient for using Glue via the command-line, in the
case of data viewers, we actually want to translate ROIs into subset states. To
do this, we can use the :func:`~glue.core.subset.roi_to_subset_state` function
that takes a ROI and returns a subset state. At the moment this method works for
1- and 2-d ROIs. In more complex cases, you can also define your own logic for
converting ROIs into subset states. See the documentation of
:func:`~glue.core.subset.roi_to_subset_state` for more details.

Subset states can be combined using logical operations:

>>> state1 = data.id['x'] > 1.5
>>> state2 = data.id['y'] < 4
>>> state1 & state2
<glue.core.subset.AndState at 0x10ebd0160>
>>> state1 | state2
<glue.core.subset.OrState at 0x10ebd00f0>
>>> ~state1
<glue.core.subset.InvertState at 0x10ebd03c8>

Note that you should use ``&``, ``|``, and ``~`` as opposed to ``and``, ``or``,
and ``not``.

Subsets
-------

A subset is what we normally think of as sub-part of a dataset. Subsets are
typically created by making `Subset states`_ first. There are then different
ways of applying this subset state to a :class:`~glue.core.data.Data` object to actually create a subset. The
easiest way of doing this is to simply call the
:meth:`~glue.core.data.BaseData.new_subset` method with the
:class:`~glue.core.subset.SubsetState` and optionally a label describing that
subset::

   >>> subset = data.new_subset(state, label='x > 1.5')
   >>> subset
   Subset: x > 1.5 (data: )

The resulting subset can then be used in a similar way to a
:class:`~glue.core.data.Data` object, but it will return only the values in the
subset::

    >>> subset['x']
    array([2, 3])

    >>> subset['y']
    array([3, 4])

Finally, you can also get the mask from a subset::

    >>> subset.to_mask()
    array([False,  True,  True], dtype=bool)

One of the benefits of subset states is that they can be applied to multiple
data objects, and if the different data objects have linked components (as described in :doc:`linking`), this
may produce several valid subsets in different datasets. We can apply a :class:`~glue.core.subset.SubsetState` to all datasets in a data collection by using the  :meth:`~glue.core.data_collection.DataCollection.new_subset_group` method with
the :class:`~glue.core.subset.SubsetState` and a label describing that subset, similarly to :meth:`~glue.core.data.BaseData.new_subset`

    >>> from glue.core import DataCollection
    >>> data_collection = DataCollection([data])
    >>> subset_group = data_collection.new_subset_group('x > 1.5', state)

This creates a :class:`~glue.core.subset_group.SubsetGroup` which represents a group of subsets, with the individual subsets accessible via the ``subsets`` attribute::

    >>> subset = subset_group.subsets[0]
    >>> subset
    Subset: x > 1.5 (data: )
