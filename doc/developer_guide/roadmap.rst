Development Roadmap
===================

This page provides a high-level overview of some of the directions in which we
want to push development in future. We are very interested in hearing from
people who are interested in contributing to any of the ideas below - if you
are, please join the friendly
`glue-viz-dev <https://groups.google.com/forum/#!forum/glue-viz-dev>`_ list and
let us know!

There are many more ways you can contribute to glue that are not mentioned
below - these are just the tip of the iceberg, but are here to give you an idea
of places you might be able to contribute. You can also search the issue
tracker on glue for all issues related to `enhancements
<https://github.com/glue-viz/glue/issues?q=is%3Aopen+is%3Aissue+label%3Aenhancement>`_
for example.

Support for big/complex data: an abstract data and computation interface
------------------------------------------------------------------------

Glue currently provides ways of importing data from different sources, but
ultimately, the data is essentially loaded in memory. In principle, the
:class:`~glue.core.data.Data` class can be subclassed in order to provide for
example a object where data is only accessed on-the-fly as needed (in fact,
this also happens if a memory-mapped Numpy array is passed to
:class:`~glue.core.data.Data`). Computations such as calculating histograms or
selections is left up to the rest of the glue, and viewers are responsible
for figuring out which subsets of data to access, if needed, and how to stride
over the data when only a subset is needed. In addition, a lot of the glue code
assumes regularly gridded datasets - which makes it difficult to apply to e.g.
simulations with adaptive grids.

It would be nice to have a much better separation between data
representation/access/computation and the rest of the interactive glue
environment, including the viewers. The idea would be to develop an abstract
base class for data objects which defines ways to access the data values and
subsets, including for example ways of computing fixed resolution buffers for
both the data and subsets. The Data object would be responsible for storing the
data as well as information about the subsets in that dataset, in an efficient
way. In fact, glue could then function entirely in world coordinates and not
even have to worry about the concept of 'pixels' in the data.

A consequence of this is that image viewers for example would simply request
fixed resolution buffers at the screen resolution, for both data and subsets,
and would then be able to display them. Behind the scenes, the user could be
using e.g. a package such as `yt <http://yt-project.org/>`_ to access a 3Tb
simulation file with adaptive/nested grids, but this would be seamless to the
user (except of course that the speed would be limited by the computational
requirements of the data object).

In fact, we could even provide a way to transfer API calls to this standard
Data API over the network, which would open up the possibility of using glue to
explore datasets hosted on computer clusters. Of course, there would be some
network latency during operations, but *some* latency would be expected anyway
for very large datasets, which would still benefit hugely from this.

Things that would need to be done in order to achieve this:

#. Define what belongs inside the Data abstraction and what doesn't
#. Define an API for data/subset access and computations
#. Refactor glue to use this data access API with the built-in
   :class:`~glue.core.data.Data` objects.
#. Develop new data objects based e.g. on yt
#. Develop a way for the data API calls to be passed over the network

Related GitHub issues: `#708 <https://github.com/glue-viz/glue/issues/708>`_

Support for big data: more efficient viewers
--------------------------------------------

`Matplotlib <http://matplotlib.org/>`_ and `VisPy <http://vispy.org/>`_ both
start becoming slow when the limit of a million points/markers is reached. This
severely limits the size of the largest datasets that can be visualized in the
scatter plot viewers, because the visualization will be slow even if the data
contains only two components of a million elements each. In addition to large
tables, this can easily happen if the user makes a scatter plot of one image
versus another.

We therefore need to work on more efficient ways to show scatter plot data. In
particular, we could explore methods that rasterize the points extremely
efficiently, or methods that sub-sample the points in smart ways (for example,
neighbouring points could be replaced by a slightly larger point).

Related GitHub issues: `#722 <https://github.com/glue-viz/glue/issues/722>`_

Glue in the browser
-------------------

It is currently already possible to :ref:`launch glue from an IPython/Jupyter notebook <notebook>` and
access the data and viewers using the returned application object. However, the
next step would be to implement actual viewers that are not based on Qt, but
instead can be used inside the notebook directly. One promising avenue would be
to explore the use of `bokeh <http://bokeh.pydata.org>`_.

The glue code base is designed so that the core representation of data objects,
subsets, and so on in glue.core is completely independent of the visualization
framework. Therefore, this would just require developing new viewers, not
re-writing large sections of already existing code.

Related GitHub issues: `#801 <https://github.com/glue-viz/glue/issues/801>`_





