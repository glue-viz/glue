from __future__ import absolute_import, division, print_function

import logging
import warnings

import numpy as np
import pandas as pd

from glue.utils import unique, shape_to_string, coerce_numeric, check_sorted

from glue.core.util import row_lookup

__all__ = ['Component', 'DerivedComponent',
           'CategoricalComponent', 'CoordinateComponent']


class Component(object):

    """ Stores the actual, numerical information for a particular quantity

    Data objects hold one or more components, accessed via
    ComponentIDs. All Components in a data set must have the same
    shape and number of dimensions

    Notes
    -----
    Instead of instantiating Components directly, consider using
    :meth:`Component.autotyped`, which chooses a subclass most appropriate
    for the data type.
    """

    def __init__(self, data, units=None):
        """
        :param data: The data to store
        :type data: :class:`numpy.ndarray`

        :param units: Optional unit label
        :type units: str
        """

        # The physical units of the data
        self.units = units

        # The actual data
        # subclasses may pass non-arrays here as placeholders.
        if isinstance(data, np.ndarray):
            data = coerce_numeric(data)
            data.setflags(write=False)  # data is read-only

        self._data = data

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        self._units = str(value)

    @property
    def hidden(self):
        """Whether the Component is hidden by default"""
        return False

    @property
    def data(self):
        """ The underlying :class:`numpy.ndarray` """
        return self._data

    @property
    def shape(self):
        """ Tuple of array dimensions """
        return self._data.shape

    @property
    def ndim(self):
        """ The number of dimensions """
        return len(self._data.shape)

    def __getitem__(self, key):
        logging.debug("Using %s to index data of shape %s", key, self.shape)
        return self._data[key]

    @property
    def numeric(self):
        """
        Whether or not the datatype is numeric
        """
        return np.can_cast(self.data[0], np.complex)

    @property
    def categorical(self):
        """
        Whether or not the datatype is categorical
        """
        return False

    def __str__(self):
        return "Component with shape %s" % shape_to_string(self.shape)

    def jitter(self, method=None):
        raise NotImplementedError

    def to_series(self, **kwargs):
        """ Convert into a pandas.Series object.

        :param kwargs: All kwargs are passed to the Series constructor.
        :return: pandas.Series
        """

        return pd.Series(self.data.ravel(), **kwargs)

    @classmethod
    def autotyped(cls, data, units=None):
        """
        Automatically choose between Component and CategoricalComponent,
        based on the input data type.

        :param data: The data to pack into a Component (array-like)
        :param units: Optional units
        :type units: str

        :returns: A Component (or subclass)
        """
        data = np.asarray(data)

        if np.issubdtype(data.dtype, np.object_):
            return CategoricalComponent(data, units=units)

        n = coerce_numeric(data)
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

    """ A component which derives its data from a function """

    def __init__(self, data, link, units=None):
        """
        :param data: The data object to use for calculation
        :type data: :class:`~glue.core.data.Data`

        :param link: The link that carries out the function
        :type link: :class:`~glue.core.component_link.ComponentLink`

        :param units: Optional unit description
        """
        super(DerivedComponent, self).__init__(data, units=units)
        self._link = link

    def set_parent(self, data):
        """ Reassign the Data object that this DerivedComponent operates on """
        self._data = data

    @property
    def hidden(self):
        return self._link.hidden

    @property
    def data(self):
        """ Return the numerical data as a numpy array """
        return self._link.compute(self._data)

    @property
    def link(self):
        """ Return the component link """
        return self._link

    def __getitem__(self, key):
        return self._link.compute(self._data, key)


class CoordinateComponent(Component):

    """
    Components associated with pixel or world coordinates

    The numerical values are computed on the fly.
    """

    def __init__(self, data, axis, world=False):
        super(CoordinateComponent, self).__init__(None, None)
        self.world = world
        self._data = data
        self.axis = axis

    @property
    def data(self):
        return self._calculate()

    def _calculate(self, view=None):
        slices = [slice(0, s, 1) for s in self.shape]
        grids = np.broadcast_arrays(*np.ogrid[slices])
        if view is not None:
            grids = [g[view] for g in grids]

        if self.world:
            world = self._data.coords.pixel2world(*grids[::-1])[::-1]
            return world[self.axis]
        else:
            return grids[self.axis]

    @property
    def shape(self):
        """ Tuple of array dimensions. """
        return self._data.shape

    @property
    def ndim(self):
        """ Number of dimensions """
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


class CategoricalComponent(Component):

    """
    Container for categorical data.
    """

    def __init__(self, categorical_data, categories=None, jitter=None, units=None):
        """
        :param categorical_data: The underlying :class:`numpy.ndarray`
        :param categories: List of unique values in the data
        :jitter: Strategy for jittering the data
        """

        super(CategoricalComponent, self).__init__(None, units)

        self._categorical_data = np.asarray(categorical_data)
        if self._categorical_data.ndim > 1:
            raise ValueError("Categorical Data must be 1-dimensional")

        # Disable changing of categories
        self._categorical_data.setflags(write=False)

        self._categories = categories
        self._jitter_method = jitter
        self._is_jittered = False
        self._data = None
        if self._categories is None:
            self._update_categories()
        else:
            self._update_data()

    @property
    def codes(self):
        """
        The index of the category for each value in the array.
        """
        return self._data

    @property
    def labels(self):
        """
        The original categorical data.
        """
        return self._categorical_data

    @property
    def categories(self):
        """
        The categories.
        """
        return self._categories

    @property
    def data(self):
        warnings.warn("The 'data' attribute is deprecated. Use 'codes' "
                      "instead to access the underlying index of the "
                      "categories")
        return self.codes

    @property
    def numeric(self):
        return False

    @property
    def categorical(self):
        return True

    def _update_categories(self, categories=None):
        """
        :param categories: A sorted array of categories to find in the dataset.
        If None the categories are the unique items in the data.
        :return: None
        """
        if categories is None:
            categories, inv = unique(self._categorical_data)
            self._categories = categories
            self._data = inv.astype(np.float)
            self._data.setflags(write=False)
            self.jitter(method=self._jitter_method)
        else:
            if check_sorted(categories):
                self._categories = categories
                self._update_data()
            else:
                raise ValueError("Provided categories must be Sorted")

    def _update_data(self):
        """
        Converts the categorical data into the numeric representations given
        self._categories
        """
        self._is_jittered = False

        self._data = row_lookup(self._categorical_data, self._categories)
        self.jitter(method=self._jitter_method)
        self._data.setflags(write=False)

    def jitter(self, method=None):
        """
        Jitter the data so the density of points can be easily seen in a
        scatter plot.

        :param method: None | 'uniform':

        * None: No jittering is done (or any jittering is undone).
        * uniform: A unformly distributed random variable (-0.5, 0.5)
            is applied to each point.

        :return: None
        """

        if method not in set(['uniform', None]):
            raise ValueError('%s jitter not supported' % method)
        self._jitter_method = method
        seed = 1234567890
        rand_state = np.random.RandomState(seed)

        if (self._jitter_method is None) and self._is_jittered:
            self._update_data()
        elif (self._jitter_method is 'uniform') and not self._is_jittered:
            iswrite = self._data.flags['WRITEABLE']
            self._data.setflags(write=True)
            self._data += rand_state.uniform(-0.5, 0.5, size=self._data.shape)
            self._is_jittered = True
            self._data.setflags(write=iswrite)

    def to_series(self, **kwargs):
        """ Convert into a pandas.Series object.

        This will be converted as a dtype=np.object!

        :param kwargs: All kwargs are passed to the Series constructor.
        :return: pandas.Series
        """

        return pd.Series(self._categorical_data.ravel(),
                         dtype=np.object, **kwargs)
