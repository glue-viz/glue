from __future__ import absolute_import, division, print_function

import operator
import logging

import numpy as np
import pandas as pd

from .coordinates import Coordinates
from .visual import VisualAttributes
from .visual import COLORS
from .exceptions import IncompatibleAttribute
from .component_link import (ComponentLink, CoordinateComponentLink,
                             BinaryComponentLink)
from .subset import Subset, InequalitySubsetState, SubsetState
from .hub import Hub
from .util import (split_component_view, view_shape,
                   coerce_numeric, check_sorted)
from .decorators import clear_cache
from .message import (DataUpdateMessage,
                      DataAddComponentMessage, NumericalDataChangedMessage,
                      SubsetCreateMessage, ComponentsChangedMessage)

from .odict import OrderedDict
from ..external import six

__all__ = ['Data', 'ComponentID', 'Component', 'DerivedComponent',
           'CategoricalComponent', 'CoordinateComponent']

# access to ComponentIDs via .item[name]


class ComponentIDDict(object):

    def __init__(self, data, **kwargs):
        self.data = data

    def __getitem__(self, key):
        result = self.data.find_component_id(key)
        if result is None:
            raise KeyError("ComponentID not found or not unique: %s"
                           % key)
        return result


class ComponentID(object):

    """ References a :class:`Component` object within a :class:`Data` object.

    ComponentIDs behave as keys::

       component_id = data.id[name]
       data[component_id] -> numpy array

    """

    def __init__(self, label, hidden=False):
        """:param label: Name for the ID
           :type label: str"""
        self._label = str(label)
        self._hidden = hidden

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        """Change label.

        .. warning::
            Label changes are not currently tracked by client
            classes. Label's should only be changd before creating other
            client objects
        """
        self._label = str(value)

    @property
    def hidden(self):
        """Whether to hide the component by default"""
        return self._hidden

    def __str__(self):
        return str(self._label)

    def __repr__(self):
        return str(self._label)

    def __eq__(self, other):
        if np.issubsctype(type(other), np.number):
            return InequalitySubsetState(self, other, operator.eq)
        return other is self

    # In Python 3, if __eq__ is defined, then __hash__ has to be re-defined
    if six.PY3:
        __hash__ = object.__hash__

    def __ne__(self, other):
        if np.issubsctype(type(other), np.number):
            return InequalitySubsetState(self, other, operator.ne)
        return other is not self

    def __gt__(self, other):
        return InequalitySubsetState(self, other, operator.gt)

    def __ge__(self, other):
        return InequalitySubsetState(self, other, operator.ge)

    def __lt__(self, other):
        return InequalitySubsetState(self, other, operator.lt)

    def __le__(self, other):
        return InequalitySubsetState(self, other, operator.le)

    def __add__(self, other):
        return BinaryComponentLink(self, other, operator.add)

    def __radd__(self, other):
        return BinaryComponentLink(other, self, operator.add)

    def __sub__(self, other):
        return BinaryComponentLink(self, other, operator.sub)

    def __rsub__(self, other):
        return BinaryComponentLink(other, self, operator.sub)

    def __mul__(self, other):
        return BinaryComponentLink(self, other, operator.mul)

    def __rmul__(self, other):
        return BinaryComponentLink(other, self, operator.mul)

    def __div__(self, other):
        return BinaryComponentLink(self, other, operator.div)

    def __rdiv__(self, other):
        return BinaryComponentLink(other, self, operator.div)

    def __truediv__(self, other):
        return BinaryComponentLink(self, other, operator.truediv)

    def __rtruediv__(self, other):
        return BinaryComponentLink(other, self, operator.truediv)

    def __pow__(self, other):
        return BinaryComponentLink(self, other, operator.pow)

    def __rpow__(self, other):
        return BinaryComponentLink(other, self, operator.pow)


class Component(object):

    """ Stores the actual, numerical information for a particular quantity

    Data objects hold one or more components, accessed via
    ComponentIDs. All Components in a data set must have the same
    shape and number of dimensions

    Note
    ----
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

    def __str__(self):
        return "Component with shape %s" % (self.shape,)

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

        :param data: The data to pack into a Component
        :type data: Array-like
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

        # Check that categorical data is of uniform type
        if isinstance(categorical_data, np.ndarray):
            if categorical_data.dtype.kind == "O":
                raise TypeError("Numpy object type not supported")
        else:
            common_type = type(categorical_data[0])
            for item in categorical_data:
                if not (isinstance(item, common_type)):
                    raise TypeError("Items in categorical data should all be the same type")

        # Assign categorical data, converting to strings. We force the copy
        # because next we will be calling setflags and we don't want to call
        # that on the original data.
        self._categorical_data = np.array(categorical_data, copy=True, dtype=str)

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

    def _update_categories(self, categories=None):
        """
        :param categories: A sorted array of categories to find in the dataset.
        If None the categories are the unique items in the data.
        :return: None
        """
        if categories is None:
            categories, inv = np.unique(self._categorical_data,
                                        return_inverse=True)
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
        # Complicated because of the case of items not in
        # self._categories may be on either side of the sorted list
        left = np.searchsorted(self._categories,
                               self._categorical_data,
                               side='left')
        right = np.searchsorted(self._categories,
                                self._categorical_data,
                                side='right')
        self._data = left.astype(float)
        self._data[(left == 0) & (right == 0)] = np.nan
        self._data[left == len(self._categories)] = np.nan

        self._data[self._data == len(self._categories)] = np.nan
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

        if method not in {'uniform', None}:
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


class Data(object):

    """The basic data container in Glue.

    The data object stores data as a collection of
    :class:`~glue.core.data.Component` objects.  Each component stored in a
    dataset must have the same shape.

    Catalog data sets are stored such that each column is a distinct
    1-dimensional :class:`~glue.core.data.Component`.

    There are several ways to extract the actual numerical data stored in a
    :class:`~glue.core.data.Data` object::

       data = Data(x=[1, 2, 3], label='data')
       xid = data.id['x']

       data[xid]
       data.get_component(xid).data
       data['x']  # if 'x' is a unique component name

    Likewise, datasets support :ref:`fancy indexing <numpy:basics.indexing>`::

        data[xid, 0:2]
        data[xid, [True, False, True]]

    See also: :ref:`data_tutorial`
    """

    def __init__(self, label="", **kwargs):
        """

        :param label: label for data
        :type label: str

        Extra array-like keywords are extracted into components
        """
        # Coordinate conversion object
        self.coords = Coordinates()
        self._shape = ()

        # Components
        self._components = OrderedDict()
        self._pixel_component_ids = []
        self._world_component_ids = []

        self.id = ComponentIDDict(self)

        # Tree description of the data
        # (Deprecated)
        self.tree = None

        # Subsets of the data
        self._subsets = []

        # Hub that the data is attached to
        self.hub = None

        self.style = VisualAttributes(parent=self)

        self._coordinate_links = None

        self.data = self
        self.label = label

        self.edit_subset = None

        for lbl, data in sorted(kwargs.items()):
            self.add_component(data, lbl)

        self._key_joins = {}

    @property
    def subsets(self):
        """
        Tuple of subsets attached to this dataset
        """
        return tuple(self._subsets)

    @property
    def ndim(self):
        """
        Dimensionality of the dataset
        """
        return len(self.shape)

    @property
    def shape(self):
        """
        Tuple of array dimensions, like :attr:`numpy.ndarray.shape`
        """
        return self._shape

    @property
    def label(self):
        """ Convenience access to data set's label """
        return self._label

    @label.setter
    def label(self, value):
        """ Set the label to value
        """
        self._label = value
        self.broadcast(attribute='label')

    @property
    def size(self):
        """
        Total number of elements in the dataset.
        """
        return np.product(self.shape)

    def _check_can_add(self, component):
        if isinstance(component, DerivedComponent):
            return component._data is self
        else:
            if len(self._components) == 0:
                return True
            return component.shape == self.shape

    def dtype(self, cid):
        """Lookup the dtype for the data associated with a ComponentID"""

        # grab a small piece of data
        ind = tuple([slice(0, 1)] * self.ndim)
        arr = self[cid, ind]
        return arr.dtype

    def remove_component(self, component_id):
        """ Remove a component from a data set

        :param component_id: the component to remove
        :type component_id: :class:`~glue.core.data.ComponentID`
        """
        if component_id in self._components:
            self._components.pop(component_id)

    def join_on_key(self, other, cid, cid_other):
        """
        Create an *element* mapping to another dataset, by
        joining on values of ComponentIDs in both datasets.

        This join allows any subsets defined on `other` to be
        propagated to self.

        :param other: :class:`Data` to join with
        :param cid: str or :class:`ComponentID` in this dataset to use as a key
        :param cid_other: ComponentID in the other dataset to use as a key

        :example:

        >>> d1 = Data(x=[1, 2, 3, 4, 5], k1=[0, 0, 1, 1, 2], label='d1')
        >>> d2 = Data(y=[2, 4, 5, 8, 4], k2=[1, 3, 1, 2, 3], label='d2')
        >>> d2.join_on_key(d1, 'k2', 'k1')

        >>> s = d1.new_subset()
        >>> s.subset_state = d1.id['x'] > 2
        >>> s.to_mask()
        array([False, False,  True,  True,  True], dtype=bool)

        >>> s = d2.new_subset()
        >>> s.subset_state = d1.id['x'] > 2
        >>> s.to_mask()
        array([ True, False,  True,  True, False], dtype=bool)

        The subset state selects the last 3 items in d1. These have
        key values k1 of 1 and 2. Thus, the selected items in d2
        are the elements where k2 = 1 or 2.
        """
        _i1, _i2 = cid, cid_other
        cid = self.find_component_id(cid)
        cid_other = other.find_component_id(cid_other)
        if cid is None:
            raise ValueError("ComponentID not found in %s: %s" %
                             (self.label, _i1))
        if cid_other is None:
            raise ValueError("ComponentID not found in %s: %s" %
                             (other.label, _i2))

        self._key_joins[other] = (cid, cid_other)
        other._key_joins[self] = (cid_other, cid)

    def add_component(self, component, label, hidden=False):
        """ Add a new component to this data set.

        :param component: object to add. Can be a Component,
                          array-like object, or ComponentLink

        :param label:
              The label. If this is a string,
              a new :class:`ComponentID` with this label will be
              created and associated with the Component

        :type component: :class:`~glue.core.data.Component` or
                         array-like
        :type label: :class:`str` or :class:`~glue.core.data.ComponentID`

        :raises:

           TypeError, if label is invalid
           ValueError if the component has an incompatible shape

        :returns:

           The ComponentID associated with the newly-added component
        """

        if isinstance(component, ComponentLink):
            component = DerivedComponent(self, component)

        if not isinstance(component, Component):
            component = Component.autotyped(component)

        if isinstance(component, DerivedComponent):
            component.set_parent(self)

        if not(self._check_can_add(component)):
            raise ValueError("The dimensions of component %s are "
                             "incompatible with the dimensions of this data: "
                             "%r vs %r" % (label, component.shape, self.shape))

        if isinstance(label, ComponentID):
            component_id = label
        else:
            component_id = ComponentID(label, hidden=hidden)

        is_present = component_id in self._components
        self._components[component_id] = component

        first_component = len(self._components) == 1
        if first_component:
            if isinstance(component, DerivedComponent):
                raise TypeError("Cannot add a derived component as "
                                "first component")
            self._shape = component.shape
            self._create_pixel_and_world_components()

        if self.hub and (not is_present):
            msg = DataAddComponentMessage(self, component_id)
            self.hub.broadcast(msg)
            msg = ComponentsChangedMessage(self)
            self.hub.broadcast(msg)

        return component_id

    def add_component_link(self, link, label=None):
        """ Shortcut method for generating a new :class:`DerivedComponent`
        from a ComponentLink object, and adding it to a data set.

        :param link: :class:`~glue.core.component_link.ComponentLink`
        :param label: The ComponentID or label to attach to.
        :type label: :class:`~glue.core.data.ComponentID` or str

        :returns:
            The :class:`DerivedComponent` that was added
        """
        if label is not None:
            if not isinstance(label, ComponentID):
                label = ComponentID(label)
            link.set_to_id(label)

        if link.get_to_id() is None:
            raise TypeError("Cannot add component_link: "
                            "has no 'to' ComponentID")

        dc = DerivedComponent(self, link)
        to_ = link.get_to_id()
        self.add_component(dc, to_)
        return dc

    def _create_pixel_and_world_components(self):
        for i in range(self.ndim):
            comp = CoordinateComponent(self, i)
            label = pixel_label(i, self.ndim)
            cid = self.add_component(comp, "Pixel %s" % label, hidden=True)
            self._pixel_component_ids.append(cid)
        if self.coords:
            for i in range(self.ndim):
                comp = CoordinateComponent(self, i, world=True)
                label = self.coords.axis_label(i)
                cid = self.add_component(comp, label, hidden=True)
                self._world_component_ids.append(cid)

    @property
    def components(self):
        """ All :class:`ComponentIDs <ComponentID>` in the Data

        :rtype: list
        """
        return sorted(self._components.keys(), key=lambda x: x.label)

    @property
    def visible_components(self):
        """ :class:`ComponentIDs <ComponentID>` for all non-hidden components.

        :rtype: list
        """
        return [cid for cid, comp in self._components.items()
                if not cid.hidden and not comp.hidden]

    @property
    def primary_components(self):
        """The ComponentIDs not associated with a :class:`DerivedComponent`

        :rtype: list
        """
        return [c for c in self.component_ids() if
                not isinstance(self._components[c], DerivedComponent)]

    @property
    def derived_components(self):
        """The ComponentIDs for each :class:`DerivedComponent`

        :rtype: list
        """
        return [c for c in self.component_ids() if
                isinstance(self._components[c], DerivedComponent)]

    @property
    def pixel_component_ids(self):
        """
        The :class:`ComponentIDs <ComponentID>` for each pixel coordinate.
        """
        return self._pixel_component_ids

    @property
    def world_component_ids(self):
        """
        The :class:`ComponentIDs <ComponentID>` for each world coordinate.
        """
        return self._world_component_ids

    def find_component_id(self, label):
        """ Retrieve component_ids associated by label name.

        :param label: ComponentID or string to search for

        :returns:
            The associated ComponentID if label is found and unique, else None
        """
        result = [cid for cid in self.component_ids() if
                  cid.label == label or cid is label]
        if len(result) == 1:
            return result[0]

    @property
    def coordinate_links(self):
        """A list of the ComponentLinks that connect pixel and
        world. If no coordinate transformation object is present,
        return an empty list.
        """
        if self._coordinate_links:
            return self._coordinate_links

        if not self.coords:
            return []

        if self.ndim != len(self._pixel_component_ids) or \
                self.ndim != len(self._world_component_ids):
                # haven't populated pixel, world coordinates yet
            return []

        def make_toworld_func(i):
            def pix2world(*args):
                return self.coords.pixel2world(*args[::-1])[::-1][i]
            return pix2world

        def make_topixel_func(i):
            def world2pix(*args):
                return self.coords.world2pixel(*args[::-1])[::-1][i]
            return world2pix

        result = []
        for i in range(self.ndim):
            link = CoordinateComponentLink(self._pixel_component_ids,
                                           self._world_component_ids[i],
                                           self.coords, i)
            result.append(link)
            link = CoordinateComponentLink(self._world_component_ids,
                                           self._pixel_component_ids[i],
                                           self.coords, i, pixel2world=False)
            result.append(link)

        self._coordinate_links = result
        return result

    def get_pixel_component_id(self, axis):
        """Return the pixel :class:`ComponentID` associated with a given axis
        """
        return self._pixel_component_ids[axis]

    def get_world_component_id(self, axis):
        """Return the world :class:`ComponentID` associated with a given axis
        """
        return self._world_component_ids[axis]

    def component_ids(self):
        """
        Equivalent to :attr:`Data.components`
        """
        return list(self._components.keys())

    def new_subset(self, subset=None, color=None, label=None, **kwargs):
        """
        Create a new subset, and attach to self.

        .. note:: The preferred way for creating subsets is via
            :meth:`~glue.core.data_collection.DataCollection.new_subset_group`.
            Manually-instantiated subsets will **not** be
            represented properly by the UI

        :param subset: optional, reference subset or subset state.
                       If provided, the new subset will copy the logic of
                       this subset.

        :returns: The new subset object
        """
        nsub = len(self.subsets)
        color = color or COLORS[nsub % len(COLORS)]
        label = label or "%s.%i" % (self.label, nsub + 1)
        new_subset = Subset(self, color=color, label=label, **kwargs)
        if subset is not None:
            new_subset.subset_state = subset.subset_state.copy()

        self.add_subset(new_subset)
        return new_subset

    def add_subset(self, subset):
        """Assign a pre-existing subset to this data object.

        :param subset: A :class:`~glue.core.subset.Subset` or
                       :class:`~glue.core.subset.SubsetState` object

        If input is a :class:`~glue.core.subset.SubsetState`,
        it will be wrapped in a new Subset automatically

        .. note:: The preferred way for creating subsets is via
            :meth:`~glue.core.data_collection.DataCollection.new_subset_group`.
            Manually-instantiated subsets will **not** be
            represented properly by the UI
        """
        if subset in self.subsets:
            return  # prevents infinite recursion
        if isinstance(subset, SubsetState):
            # auto-wrap state in subset
            state = subset
            subset = Subset(None)
            subset.subset_state = state

        self._subsets.append(subset)

        if subset.data is not self:
            subset.do_broadcast(False)
            subset.data = self
            subset.label = subset.label  # hacky. disambiguates name if needed

        if self.hub is not None:
            msg = SubsetCreateMessage(subset)
            self.hub.broadcast(msg)

        subset.do_broadcast(True)

    def register_to_hub(self, hub):
        """ Connect to a hub.

        This method usually doesn't have to be called directly, as
        DataCollections manage the registration of data objects
        """
        if not isinstance(hub, Hub):
            raise TypeError("input is not a Hub object: %s" % type(hub))
        self.hub = hub

    def broadcast(self, attribute=None):
        """
        Send a :class:`~glue.core.message.DataUpdateMessage` to the hub

        :param attribute: Name of an attribute that has changed
        :type attribute: str
        """
        if not self.hub:
            return
        msg = DataUpdateMessage(self, attribute=attribute)
        self.hub.broadcast(msg)

    def update_id(self, old, new):
        """Reassign a component to a different :class:`ComponentID`

        :param old: The old :class:`ComponentID`.
        :param new: The new :class:`ComponentID`.
        """

        # note: its problematic to remove an component
        #      during updating, since plots may already
        # be using it (issue #279). Instead,
        #      we just mark it as hidden
        if new is old:
            return

        changed = False
        if old in self._components:
            self._components[new] = self._components[old]
            changed = True
        try:
            index = self._pixel_component_ids.index(old)
            self._pixel_component_ids[index] = new
            changed = True
        except ValueError:
            pass
        try:
            index = self._world_component_ids.index(old)
            self._world_component_ids[index] = new
            changed = True
        except ValueError:
            pass

        if changed and self.hub is not None:
            # obfuscante name if needed
            if new.label == old.label:
                old.label = '_' + old.label
            old._hidden = True
            self.hub.broadcast(ComponentsChangedMessage(self))

    def __str__(self):
        s = "Data Set: %s" % self.label
        s += "Number of dimensions: %i\n" % self.ndim
        s += "Shape: %s\n" % ' x '.join([str(x) for x in self.shape])
        s += "Components:\n"
        for i, component in enumerate(self._components):
            s += " %i) %s\n" % (i, component)
        return s[:-1]

    def __repr__(self):
        return 'Data (label: %s)' % self.label

    def __setattr__(self, name, value):
        if name == "hub" and hasattr(self, 'hub') \
                and self.hub is not value and self.hub is not None:
            raise AttributeError("Data has already been assigned "
                                 "to a different hub")
        object.__setattr__(self, name, value)

    def __getitem__(self, key):
        """ Shortcut syntax to access the numerical data in a component.
        Equivalent to:

        ``component = data.get_component(component_id).data``

        :param key:
          The component to fetch data from

        :type key: :class:`~glue.core.data.ComponentID`

        :returns: :class:`~numpy.ndarray`
        """
        key, view = split_component_view(key)
        if isinstance(key, six.string_types):
            _k = key
            key = self.find_component_id(key)
            if key is None:
                raise IncompatibleAttribute(_k)

        if isinstance(key, ComponentLink):
            return key.compute(self, view)

        try:
            comp = self._components[key]
        except KeyError:
            raise IncompatibleAttribute(key)

        shp = view_shape(self.shape, view)
        if view is not None:
            result = comp[view]
        else:
            result = comp.data

        assert result.shape == shp, \
            "Component view returned bad shape: %s %s" % (result.shape, shp)
        return result

    def __setitem__(self, key, value):
        """
        Wrapper for data.add_component()
        """
        self.add_component(value, key)

    def get_component(self, component_id):
        """Fetch the component corresponding to component_id.

        :param component_id: the component_id to retrieve
        """
        if component_id is None:
            raise IncompatibleAttribute()

        if isinstance(component_id, six.string_types):
            component_id = self.id[component_id]

        try:
            return self._components[component_id]
        except KeyError:
            raise IncompatibleAttribute(component_id)

    def to_dataframe(self, index=None):
        """ Convert the Data object into a pandas.DataFrame object

        :param index: Any 'index-like' object that can be passed to the
        pandas.Series constructor

        :return: pandas.DataFrame
        """

        h = lambda comp: self.get_component(comp).to_series(index=index)
        df = pd.DataFrame({comp.label: h(comp) for comp in self.components})
        order = [comp.label for comp in self.components]
        return df[order]

    def update_components(self, mapping):
        """
        Change the numerical data associated with some of the Components
        in this Data object.

        All changes to component numerical data should use this method,
        which broadcasts the state change to the appropriate places.

        :param mapping: A dict mapping Components or ComponenIDs to arrays.

        This method has the following restrictions:
          - New compoments must have the same shape as old compoments
          - Component subclasses cannot be updated.
        """
        for comp, data in mapping.items():
            if isinstance(comp, ComponentID):
                comp = self.get_component(comp)
            data = np.asarray(data)
            if data.shape != self.shape:
                raise ValueError("Cannot change shape of data")

            comp._data = data

        # alert hub of the change
        if self.hub is not None:
            msg = NumericalDataChangedMessage(self)
            self.hub.broadcast(msg)

        for subset in self.subsets:
            clear_cache(subset.subset_state.to_mask)


def pixel_label(i, ndim):
    if ndim == 2:
        return ['y', 'x'][i]
    if ndim == 3:
        return ['z', 'y', 'x'][i]
    return "Axis %s" % i
