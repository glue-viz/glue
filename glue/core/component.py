import logging

import numpy as np
import pandas as pd

from glue.core.coordinate_helpers import dependent_axes, pixel2world_single_axis
from glue.utils import shape_to_string, coerce_numeric, categorical_ndarray

try:
    import dask.array as da
    DASK_INSTALLED = True
except ImportError:
    DASK_INSTALLED = False

__all__ = ['Component', 'DerivedComponent', 'CategoricalComponent',
           'CoordinateComponent', 'DateTimeComponent']


class Component(object):

    """ Stores the actual, numerical information for a particular quantity

    Data objects hold one or more components, accessed via
    ComponentIDs. All Components in a data set must have the same
    shape and number of dimensions

    Parameters
    ----------
    data : :class:`~numpy.ndarray`
        The data to store.
    units : `str`, optional
        Unit label.

    Notes
    -----
    Instead of instantiating Components directly, consider using
    :meth:`Component.autotyped`, which chooses a subclass most appropriate
    for the data type.
    """

    def __init__(self, data, units=None):
        # The physical units of the data
        self.units = units

        # The actual data
        # subclasses may pass non-arrays here as placeholders.
        if isinstance(data, np.ndarray):
            if data.dtype.kind == 'M':
                raise TypeError('DateTimeComponent should be used instead of Component for np.datetime64 arrays')
            data = coerce_numeric(data)
            data.setflags(write=False)  # data is read-only

        self._data = data

    @property
    def units(self):
        return self._units or ''

    @units.setter
    def units(self, value):
        if value is None:
            self._units = None
        else:
            self._units = str(value)

    @property
    def data(self):
        """The underlying :class:`~numpy.ndarray`"""
        return self._data

    @property
    def shape(self):
        """Tuple of array dimensions"""
        return self._data.shape

    @property
    def ndim(self):
        """The number of dimensions"""
        return len(self._data.shape)

    def __getitem__(self, key):
        logging.debug("Using %s to index data of shape %s", key, self.shape)
        return self._data[key]

    @property
    def numeric(self):
        """
        Whether or not the datatype is numeric.
        """
        # We need to be careful here to not just access self.data since that
        # would force the computation of the whole component in the case of
        # derived components, so instead we specifically only get the first
        # element.
        return np.can_cast(self[(0,) * self.ndim].dtype, complex)

    @property
    def categorical(self):
        """
        Whether or not the datatype is categorical.
        """
        return False

    @property
    def datetime(self):
        """
        Whether or not or not the datatype is a date/time
        """
        return False

    def __str__(self):
        return "%s with shape %s" % (self.__class__.__name__, shape_to_string(self.shape))

    def jitter(self, method=None):
        raise NotImplementedError

    def to_series(self, **kwargs):
        """ Convert into a pandas.Series object.

        Parameters
        ----------
        **kwargs :
            All kwargs are passed to the Series constructor.
        Returns
        -------
        :class:`pandas.Series`
        """

        return pd.Series(self.data.ravel(), **kwargs)

    @classmethod
    def autotyped(cls, data, units=None):
        """
        Automatically choose between Component and CategoricalComponent,
        based on the input data type.

        Parameters
        ----------
        data : array-like
            The data to pack into a Component.
        units : `str`, optional
            Unit description.

        Returns
        -------
        :class:`Component` (or subclass)
        """

        if DASK_INSTALLED and isinstance(data, da.Array):
            return DaskComponent(data, units=units)

        data = np.asarray(data)

        if np.issubdtype(data.dtype, np.object_):
            return CategoricalComponent(data, units=units)

        if data.dtype.kind == 'M':
            return DateTimeComponent(data)

        n = coerce_numeric(data.ravel()).reshape(data.shape)

        thresh = 0.5
        try:
            use_categorical = np.issubdtype(data.dtype, np.character) and \
                np.isfinite(n).mean() <= thresh
        except TypeError:  # isfinite not supported. non-numeric dtype
            use_categorical = True

        if use_categorical:
            return CategoricalComponent(data, units=units)
        else:
            return Component(n, units=units)


class DerivedComponent(Component):

    """
    A component which derives its data from a function.

    Parameters
    ----------
    data : :class:`~glue.core.data.Data`
        The data object to use for calculation.
    link : :class:`~glue.core.component_link.ComponentLink`
        The link that carries out the function.
    units : `str`, optional
        Unit description.
    """
    def __init__(self, data, link, units=None):
        super(DerivedComponent, self).__init__(data, units=units)
        self._link = link

    def set_parent(self, data):
        """ Reassign the Data object that this DerivedComponent operates on """
        self._data = data

    @property
    def data(self):
        """Return the numerical data as a numpy array"""
        return self._link.compute(self._data)

    @property
    def link(self):
        """Return the component link"""
        return self._link

    def __getitem__(self, key):
        return self._link.compute(self._data, key)


class CoordinateComponent(Component):
    """
    Components associated with pixel or world coordinates

    The numerical values are computed on the fly.
    """

    def __init__(self, data, axis, world=False):
        self.world = world
        self._data = data
        self.axis = axis

    @property
    def data(self):
        return self._calculate()

    @property
    def units(self):
        if self.world:
            return self._data.coords.world_axis_units[self._data.ndim - 1 - self.axis] or ''
        else:
            return ''

    def _calculate(self, view=None):

        if self.world:

            # Calculating the world coordinates can be a bottleneck if we aren't
            # careful, so we need to make sure that if not all dimensions depend
            # on each other, we use smart broadcasting.

            # The unoptimized way to do this for an N-dimensional dataset would
            # be to construct N-dimensional arrays of pixel values for each
            # coordinate. However, if we are computing the coordinates for axis
            # i, and axis i is not dependent on any other axis, then the result
            # will be an N-dimensional array where the same 1D array of
            # coordinates will be repeated over and over.

            # To optimize this, we therefore essentially consider only the
            # dependent dimensions and then broacast the result to the full
            # array size at the very end.

            # view=None actually adds a dimension which is never what we really
            # mean, at least in glue.
            if view is None:
                view = Ellipsis

            # If the view is a tuple or list of arrays, we should actually just
            # convert these straight to world coordinates since the indices
            # of the pixel coordinates are the pixel coordinates themselves.
            if isinstance(view, (tuple, list)) and isinstance(view[0], np.ndarray):
                axis = self._data.ndim - 1 - self.axis
                return pixel2world_single_axis(self._data.coords, *view[::-1],
                                               world_axis=axis)

            # For 1D arrays, slice can be given as a single slice but we need
            # to wrap it in a list to make the following code work correctly,
            # as it is then consistent with higher-dimensional cases.
            if isinstance(view, slice) or np.isscalar(view):
                view = [view]

            # Some views, e.g. with lists of integer arrays, can give arbitrarily
            # complex (copied) subsets of arrays, so in this case we don't do any
            # optimization
            if view is Ellipsis:
                optimize_view = False
            else:
                for v in view:
                    if not np.isscalar(v) and not isinstance(v, slice):
                        optimize_view = False
                        break
                else:
                    optimize_view = True

            pix_coords = []
            dep_coords = dependent_axes(self._data.coords, self.axis)

            final_slice = []
            final_shape = []

            for i in range(self._data.ndim):

                if optimize_view and i < len(view) and np.isscalar(view[i]):
                    final_slice.append(0)
                else:
                    final_slice.append(slice(None))

                # We set up a 1D pixel axis along that dimension.
                pix_coord = np.arange(self._data.shape[i])

                # If a view was specified, we need to take it into account for
                # that axis.
                if optimize_view and i < len(view):
                    pix_coord = pix_coord[view[i]]
                    if not np.isscalar(view[i]):
                        final_shape.append(len(pix_coord))
                else:
                    final_shape.append(self._data.shape[i])

                if i not in dep_coords:
                    # The axis is not dependent on this instance's axis, so we
                    # just compute the values once and broadcast along this
                    # dimension later.
                    pix_coord = 0

                pix_coords.append(pix_coord)

            # We build the list of N arrays, one for each pixel coordinate
            pix_coords = np.meshgrid(*pix_coords, indexing='ij', copy=False)

            # Finally we convert these to world coordinates
            axis = self._data.ndim - 1 - self.axis
            world_coords = pixel2world_single_axis(self._data.coords,
                                                   *pix_coords[::-1],
                                                   world_axis=axis)

            # We get rid of any dimension for which using the view should get
            # rid of that dimension.
            if optimize_view:
                world_coords = world_coords[tuple(final_slice)]

            # We then broadcast the final array back to what it should be
            world_coords = np.broadcast_to(world_coords, tuple(final_shape))

            # We apply the view if we weren't able to optimize before
            if optimize_view:
                return world_coords
            else:
                return world_coords[view]

        else:

            slices = [slice(0, s, 1) for s in self.shape]
            grids = np.broadcast_arrays(*np.ogrid[slices])
            if view is not None:
                grids = [g[view] for g in grids]
            return grids[self.axis]

    @property
    def shape(self):
        """Tuple of array dimensions."""
        return self._data.shape

    @property
    def ndim(self):
        """Number of dimensions"""
        return len(self._data.shape)

    def __getitem__(self, key):
        return self._calculate(key)

    def __lt__(self, other):
        if self.world == other.world:
            return self.axis < other.axis
        return self.world

    def __gluestate__(self, context):
        return dict(axis=self.axis, world=self.world)

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(None, rec['axis'], rec['world'])

    @property
    def numeric(self):
        return True

    @property
    def categorical(self):
        return False


class CategoricalComponent(Component):

    """
    Container for categorical data.

    Parameters
    ----------
    categorical_data : :class:`~numpy.ndarray`
        The underlying array.
    categories : `iterable`, optional
        List of unique values in the data.
    jitter : `str`, optional
         Strategy for jittering the data.
    units : `str`, optional
        Unit description.
    """

    def __init__(self, categorical_data, categories=None, jitter=None, units=None):
        # TOOD: deal with custom categories

        super(CategoricalComponent, self).__init__(None, units)

        self._data = categorical_ndarray(categorical_data, copy=False, categories=categories)

        if self._data.ndim < 1:
            raise ValueError("Categorical Data must be at least 1-dimensional")

        self.jitter(method=jitter)

    @property
    def codes(self):
        """
        The index of the category for each value in the array.
        """
        return self._data.codes

    @property
    def labels(self):
        """
        The original categorical data.
        """
        return self._data.view(np.ndarray)

    @property
    def categories(self):
        """
        The categories.
        """
        return self._data.categories

    @property
    def data(self):
        return self._data

    @property
    def numeric(self):
        return False

    @property
    def categorical(self):
        return True

    def jitter(self, method=None):
        """
        Jitter the codes so the density of points can be easily seen in a
        scatter plot for example.

        Parameters
        ----------
        method : {None, 'uniform'}
            If `None`, no jittering is done (or any jittering is undone).
            If ``'uniform'``, the codes are randomized by a uniformly
            distributed random variable.
        """
        self._data.jitter(method=method)
        self.jitter_method = method

    def to_series(self, **kwargs):
        """
        Convert into a pandas.Series object.

        This will be converted as a dtype=np.object!

        Parameters
        ----------
        **kwargs :
            All kwargs are passed to the Series constructor.

        Returns
        -------
        :class:`pandas.Series`
        """

        return pd.Series(self.labels, dtype=object, **kwargs)


class DateTimeComponent(Component):
    """
    A component representing a date/time.

    Parameters
    ----------
    data : :class:`~numpy.ndarray`
        The data to store, with `~numpy.datetime64` dtype
    """

    def __init__(self, data, units=None):

        self.units = units

        if not isinstance(data, np.ndarray) or data.dtype.kind != 'M':
            raise TypeError("DateTimeComponent should be initialized with a datetim64 Numpy array")

        self._data = data

    @property
    def numeric(self):
        return True

    @property
    def datetime(self):
        return True


class DaskComponent(Component):
    """
    A data component powered by a dask array.
    """

    def __init__(self, data, units=None):
        self._data = data
        self.units = units

    @property
    def units(self):
        return self._units or ''

    @units.setter
    def units(self, value):
        if value is None:
            self._units = None
        else:
            self._units = str(value)

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return len(self._data.shape)

    def __getitem__(self, key):
        return self._data[key].compute()

    @property
    def numeric(self):
        return True

    @property
    def categorical(self):
        return False

    @property
    def datetime(self):
        return False
