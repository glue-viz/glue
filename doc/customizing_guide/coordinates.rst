.. _coordinates:

Customizing the coordinate system of a data object
==================================================

Background
----------

Data objects represented by the :class:`~glue.core.data.Data` class can have
a coordinate system defined, for display and/or linking purposes. This
coordinate system is defined in the ``.coords`` attribute of data objects.
By default the ``coords`` object for :class:`~glue.core.data.Data` objects
created manually is :obj:`None` unless you explicitly specify ``coords=`` when
creating the data object. For data objects returned by data loaders, whether
``coords`` is set or not will depend on the particular file format. For example
if we use the glue loaders to read in an example file:

    >>> from glue.core.data_factories import load_data
    >>> from glue.utils import require_data
    >>> require_data('Astronomy/W5/w5.fits')
    Successfully downloaded data file to w5.fits
    >>> data = load_data('w5.fits')

the resulting ``coords`` object has methods to convert between pixel coordinates
and so-called 'world' or 'physical' coordinates:

    >>> data.coords.pixel_to_world_values(2, 3)  # doctest: +FLOAT_CMP
    (array(46.34527244), array(58.85867558))
    >>> data.coords.world_to_pixel_values(46.3, 58.9)  # doctest: +FLOAT_CMP
    (array(10.39880029), array(16.44193896))

If not :obj:`None`, the ``coords`` attribute will be an object exposing the two
above methods as well as other useful methods and properties related to
coordinate transformations. The programmatic interface we have adopted for
``coords`` objects is described in `A shared Python interface for World Coordinate Systems
<https://github.com/astropy/astropy-APEs/blob/master/APE14.rst>`_ (while originally
defined by the Astropy project, this is very general and not astronomy-specific).
Any object implementing that API can be used as a coordinate object and will
integrate with the rest of glue.

A number of convenience coordinate classes are available in glue for common
cases, and it is also possible to define your own (both options are described in
the next sections).

.. _affine-coordinates:

Affine coordinates
------------------

The most common cases of transformation between pixel and world coordinates are
`affine transformations <https://en.wikipedia.org/wiki/Affine_transformation>`_,
which can represent combinations of e.g. reflections, scaling, translations,
rotations, and shear. A common way of representing an affine transformations is
through an `augmented
matrix <https://en.wikipedia.org/wiki/Affine_transformation>`_, which has shape
N+1 x N+1, where N is the number of pixel and world dimensions.

Glue provides an :class:`~glue.core.coordinates.AffineCoordinates` class for
representing arbitrary affine transformations::

     >>> from glue.core.coordinates import AffineCoordinates

To initialize it, you will need to provide an augmented matrix, and optionally
lists of units and axis names (as strings). For example, to construct an affine
transformation where the x and y coordinates are each doubled, you would do::

     >>> import numpy as np
     >>> matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
     >>> affine_coords = AffineCoordinates(matrix, units=['m', 'm'], labels=['xw', 'yw'])

To use a custom coordinate system, when creating a data object you should specify
the coordinates object via the ``coords=`` keyword argument::

   >>> from glue.core import Data
   >>> data_double = Data(x=[[1, 2], [3, 4]], coords=affine_coords)
   >>> data_double.coords.pixel_to_world_values(2, 1)
   (4.0, 2.0)
   >>> data_double.coords.world_to_pixel_values(4.0, 2.0)
   (2.0, 1.0)

Identity coordinates
--------------------

A special/simple case of coordinate transformation is the identity transform,
where world coordinates are the same as pixel coordinates. Glue provides an
:class:`~glue.core.coordinates.IdentityCoordinates` class for representing
this transformation::

     >>> from glue.core.coordinates import IdentityCoordinates

To initialize it, you will need to specify the number of dimensions in the
data::

   >>> data_ident = Data(x=[1, 2, 3], coords=IdentityCoordinates(n_dim=1))
   >>> data_ident.coords.pixel_to_world_values(2, 1)
   (2, 1)
   >>> data_ident.coords.world_to_pixel_values(4.0, 2.0)
   (4.0, 2.0)

Custom coordinates
------------------

If you want to define a fully customized coordinate transformation, we
provide a :class:`~glue.core.coordinates.Coordinates` class that you can
start from to make things easier. The only required methods in this case
are the following::

    from glue.core.coordinates import Coordinates


    class MyCoordinates(Coordinates):

        def pixel_to_world_values(self, *args):
            # This should take N arguments (where N is the number of dimensions
            # in your dataset) and assume these are 0-based pixel coordinates,
            # then return N world coordinates with the same shape as the input.

        def world_to_pixel_values(self, *args):
            # This should take N arguments (where N is the number of dimensions
            # in your dataset) and assume these are 0-based pixel coordinates,
            # then return N world coordinates with the same shape as the input.

In addition, you can also optionally specify units and names for all world
coordinates with the two following properties::

        @property
        def world_axis_units(self):
            # Returns an iterable of strings given the units of the world
            # coordinates for each axis.

        @property
        def world_axis_names(self):
            # Returns an iterable of strings given the names of the world
            # coordinates for each axis.

For example, let's consider a coordinate system where the world coordinates are
simply scaled by a factor of two compared to the pixel coordinates. The minimal
class implementing this would look like::

    >>> from glue.core.coordinates import Coordinates

    >>> class DoubleCoordinates(Coordinates):
    ...
    ...     def pixel_to_world_values(self, *args):
    ...        return tuple([2.0 * x for x in args])
    ...
    ...     def world_to_pixel_values(self, *args):
    ...        return tuple([0.5 * x for x in args])

To use a custom coordinate system, when creating a data object you should specify
the coordinates object via the ``coords=`` keyword argument::

    >>> data_double = Data(x=[1, 2, 3], coords=DoubleCoordinates(n_dim=1))
    >>> data_double.coords.pixel_to_world_values(2)
    (4.0,)
    >>> data_double.coords.world_to_pixel_values(4.0)
    (2.0,)

Note that the ``n_dim=`` argument needs to be passed to give the number of
dimensions in the data.

In fact you do not need to start from our :class:`~glue.core.coordinates.Coordinates`
class - any class that conforms to the API described in
`A shared Python interface for World Coordinate Systems
<https://github.com/astropy/astropy-APEs/blob/master/APE14.rst>`_ is valid. If
you want full control over your coordinate transformations, we recomment you
take a look at that document.
