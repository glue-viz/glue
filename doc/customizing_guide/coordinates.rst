Customizing the coordinate system of a data object
==================================================

Data objects represented by the :class:`~glue.core.data.Data` class can have
a coordinate system defined, for display and/or linking purposes. This
coordinate system is defined in the ``.coords`` attribute of data objects::

    >>> from glue.core import Data
    >>> data = Data(x=[1, 2, 3])
    >>> data.coords
    <glue.core.coordinates.Coordinates object at 0x7fa52f5547b8>

This attribute can be used to convert pixel to so-called 'world' coordinates
and vice-versa::

    >>> data.coords.pixel2world(2)
    (2,)
    >>> data.coords.world2pixel(3)
    (3,)

By default the ``coords`` object for :class:`~glue.core.data.Data` objects
created manually is an instance of :class:`~glue.core.coordinates.Coordinates`
which is an identity transform, as can be seen above. However, it is possible to
define your own coordinate system instead.

To do this, you will need to either define a
:class:`~glue.core.coordinates.Coordinates` subclass that defines the following
methods::

    from glue.core.coordinates import Coordinates


    class MyCoordinates(Coordinates):

        def pixel2world(self, *args):
            # This should take N arguments (where N is the number of dimensions
            # in your dataset) and assume these are 0-based pixel coordinates,
            # then return N world coordinates with the same shape as the input.

        def world2pixel(self, *args):
            # This should take N arguments (where N is the number of dimensions
            # in your dataset) and assume these are 0-based pixel coordinates,
            # then return N world coordinates with the same shape as the input.

        def world_axis_unit(self, axis):
            # For a given axis (0-based) return the units of the world
            # coordinate as a string. This is optional and will return '' by
            # default if not defined.

        def axis_label(self, axis):
            # For a given axis (0-based) return the name of the world
            # coordinate as a string. This is optional and will return
            # 'World {axis}' by default if not defined.

        def dependent_axes(self, axis):
            # This should return a tuple of all the world dimensions that are
            # correlated with the specified pixel axis. As an example, for a
            # 2-d coordinate system rotated compared to the pixel coordinates,
            # both world coordinates depend on both pixel coordinates, so this
            # should return (0, 1). If all axes are independent, then this
            # should return (axis,) (the default implementation)

For example, let's consider a coordinate system where the world coordinates are
simply scaled by a factor of two compared to the pixel coordinates. The minimal
class implementing this would look like::

    class DoubleCoordinates(Coordinates):

        def pixel2world(self, *args):
            return tuple([2.0 * x for x in args])

        def world2pixel(self, *args):
            return ([0.5 * x for x in args])

To use a custom coordinate system, when creating a data object you should specify
the coordinates object via the ``coords=`` keyword argument::

    >>> data_double = Data(x=[1, 2, 3], coords=DoubleCoordinates())
    >>> data_double.coords.pixel2world(2)
    (4.0,)
    >>> data_double.coords.world2pixel(4.0)
    (2.0,)
