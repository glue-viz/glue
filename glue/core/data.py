import operator
import logging

import numpy as np

from .coordinates import Coordinates
from .visual import VisualAttributes
from .visual import RED, GREEN, BLUE, BROWN, ORANGE, PURPLE, PINK
from .exceptions import IncompatibleAttribute
from .component_link import (ComponentLink, CoordinateComponentLink,
                             BinaryComponentLink)
from .subset import Subset, InequalitySubsetState, SubsetState
from .hub import Hub
from .tree import Tree
from .util import split_component_view, view_shape
from .message import (DataUpdateMessage,
                      DataAddComponentMessage,
                      SubsetCreateMessage, ComponentsChangedMessage)

from .util import coerce_numeric
from .odict import OrderedDict

__all__ = ['ComponentID', 'Component', 'DerivedComponent', 'Data',
           'CoordinateComponent']

COLORS = [RED, GREEN, BLUE, BROWN, ORANGE, PURPLE, PINK]


class ComponentID(object):

    """ References a Component object within a data object

    Components are retrieved from data objects via ComponentIDs::

       component = data.get_component(component_id)
    """

    def __init__(self, label, hidden=False):
        """:param label: Name for the ID
           :type label: str"""
        self._label = label
        self._hidden = hidden

    @property
    def label(self):
        """ Return the label """
        return self._label

    @label.setter
    def label(self, value):
        """Change label.

        WARNING: Label changes are not currently tracked by client
        classes. Label's should only be changd before creating other
        client objects
        """
        self._label = value

    @property
    def hidden(self):
        """Whether to hide the component in lists"""
        return self._hidden

    def __str__(self):
        return str(self._label)

    def __repr__(self):
        return str(self._label)

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

    def __pow__(self, other):
        return BinaryComponentLink(self, other, operator.pow)

    def __rpow__(self, other):
        return BinaryComponentLink(other, self, operator.pow)


class Component(object):

    """ Stores the actual, numerical information for a particular quantity

    Data objects hold one or more components, accessed via
    ComponentIDs. All Components in a data set must have the same
    shape and number of dimensions
    """

    def __init__(self, data, units=None):
        """
        :param data: The data to store
        :type data: numpy array

        :param units: Optional unit label
        :type units: str
        """

        # The physical units of the data
        self.units = units

        # The actual data
        # subclasses may pass non-arrays here as placeholders.
        if isinstance(data, np.ndarray):
            data = coerce_numeric(data)
        self._data = data

    @property
    def hidden(self):
        return False

    @property
    def data(self):
        """ Returns the data """
        return self._data

    @property
    def shape(self):
        """ Return the shape of the data """
        return self._data.shape

    @property
    def ndim(self):
        """ Return the number of dimensions """
        return len(self._data.shape)

    def __getitem__(self, key):
        logging.debug("Using %s to index data of shape %s", key, self.shape)
        return self._data[key]

    @property
    def numeric(self):
        return np.can_cast(self.data[0], np.complex)

    def __str__(self):
        return "Component with shape %s" % self.shape

    def jitter(self, method=None):
        raise NotImplementedError

    @property
    def creation_info(self):
        """A 4-tuple describing how this component was created

        :rtype: (callable, tuple, dict, indexers)
        A 4-tuple of (factory, args, kwargs, indexers),
        which communicates that this component's data
        can be created via

        bundle = factory(*args, **kwargs)
        for i in indexers:
            bundle = bundle[i]
        component.data == bundle
        """
        from .data_factories import load_numpy_str
        from cStringIO import StringIO
        f = StringIO()
        np.save(f, self.data)
        f.seek(0)
        data = f.read().encode('base64')
        return load_numpy_str, (data,), {}, []


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
        return self._data.shape

    @property
    def ndim(self):
        return len(self._data.shape)

    def __getitem__(self, key):
        return self._calculate(key)


class CategoricalComponent(Component):

    def __init__(self, categorical_data, categories=None, jitter=None):
        super(CategoricalComponent, self).__init__(None, None)
        self._categorical_data = np.asarray(categorical_data, dtype=np.object)
        self._categories = categories
        self._jitter_method = jitter
        self._is_jittered = False
        self._data = None
        if self._categories is None:
            self._update_categories()
        else:
            self._update_data()

    def _update_categories(self, categories=None):
        if categories is None:
            categories, inv = np.unique(self._categorical_data, return_inverse=True)
            self._categories = categories
            self._data = inv.astype(np.float)
            self.jitter(method=self._jitter_method)
        else:
            self._categories = categories
            self._update_data()

    def _update_data(self):
        self._is_jittered = False
        self._data = np.nan*np.zeros(self._categorical_data.shape)
        for num, category in enumerate(self._categories):
            self._data[self._categorical_data == category] = num

        self.jitter(method=self._jitter_method)

    def jitter(self, method=None):
        """
        :param method: Currently only supports None
        :return:
        """
        self._jitter_method = method
        seed = np.abs(hash(tuple(self._categorical_data.flatten())))
        rand_state = np.random.RandomState(seed)

        if (self._jitter_method is None) and self._is_jittered:
            self._update_data()
        elif (self._jitter_method is 'uniform') and not self._is_jittered:
            self._data += rand_state.uniform(-0.5, 0.5, size=self._data.shape)
            self._is_jittered = True


class Data(object):

    """Stores data and manages subsets.

    The data object stores data as a collection of
    :class:`~glue.core.data.Component` objects.  Each component stored in a
    dataset must have the same shape.

    Catalog data sets are stored such that each column is a distinct
    1-dimensional ``Component``.

    There two ways to extract the actual numerical data stored in a
    :class:`~glue.core.data.Data` object::

       data.get_component(component_id)
       data[component_id]

    These statements are equivalent. The second is provided since the
    first is rather verbose
    """

    def __init__(self, label="", **kwargs):
        """:param label: label for data
        :type label: str"""
        # Coordinate conversion object
        self.coords = Coordinates()
        self._shape = ()

        # Components
        self._components = OrderedDict()
        self._pixel_component_ids = []
        self._world_component_ids = []

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

        self.id = ComponentIDDict(self)

        # Tree description of the data
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

        for lbl, data in kwargs.items():
            c = Component(np.asarray(data))
            self.add_component(c, lbl)

    @property
    def subsets(self):
        return tuple(self._subsets)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape(self):
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

    def add_component(self, component, label, hidden=False):
        """ Add a new component to this data set.

        :param component: object to add
        :param label:
              The label. If this is a string,
              a new ComponentID with this label will be
              created and associated with the Component

        :type component: :class:`~glue.core.data.Component` or
                         array-like
        :type label: :class:`str` or :class:`~glue.core.data.ComponentID`

        *Raises*

           TypeError, if label is invalid
           ValueError if the component has an incompatible shape

        *Returns*

           The ComponentID associated with the newly-added component
        """
        if not isinstance(component, Component):
            component = Component(np.asarray(component))

        if isinstance(component, DerivedComponent):
            component.set_parent(self)

        if not(self._check_can_add(component)):
            raise ValueError("The dimensions of component %s are "
                             "incompatible with the dimensions of this data: "
                             "%r vs %r" % (label, component.shape, self.shape))

        if isinstance(label, ComponentID):
            component_id = label
        elif isinstance(label, basestring):
            component_id = ComponentID(label, hidden=hidden)
        else:
            raise TypeError("label must be a ComponentID or string")

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

    def add_component_link(self, link, cid=None):
        """ Shortcut method for generating a new DerivedComponent
        from a ComponentLink object, and adding it to a data set.

        :param link: ComponentLink object

        Returns:

            The DerivedComponent that was added
        """
        if cid is not None:
            if isinstance(cid, basestring):
                cid = ComponentID(cid)
            link.set_to_id(cid)

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
        """ Returns a list of ComponentIDs for all components
        (primary and derived) in the data"""
        return sorted(self._components.keys(), key=lambda x: x.label)

    @property
    def visible_components(self):
        """ Returns a list of ComponentIDs for all components
        (primary and derived) in the data"""
        return [cid for cid, comp in self._components.items()
                if not cid.hidden and not comp.hidden]

    @property
    def primary_components(self):
        """Returns a list of ComponentIDs with stored data (as opposed
        to a :class:`~glue.core.data.DerivedComponent` )
        """
        return [c for c in self.component_ids() if
                not isinstance(self._components[c], DerivedComponent)]

    @property
    def derived_components(self):
        """A list of ComponentIDs for each
        :class:`~glue.core.data.DerivedComponent` in the data.

        (Read only)
        """
        return [c for c in self.component_ids() if
                isinstance(self._components[c], DerivedComponent)]

    @property
    def pixel_component_ids(self):
        return self._pixel_component_ids

    @property
    def world_component_ids(self):
        return self._world_component_ids

    def find_component_id(self, label):
        """ Retrieve component_ids associated by label name.

        :param label: string to search for

        Returns:

            The associated ComponentID if label is found and unique, else None
        """
        result = [cid for cid in self.component_ids() if
                  cid.label == label]
        if len(result) == 1:
            return result[0]

    @property
    def coordinate_links(self):
        """Return a list of the ComponentLinks that connect pixel and
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

        for r in result:
            r.hide_from_editor = True

        self._coordinate_links = result
        return result

    def get_pixel_component_id(self, axis):
        return self._pixel_component_ids[axis]

    def get_world_component_id(self, axis):
        return self._world_component_ids[axis]

    def component_ids(self):
        return self._components.keys()

    def new_subset(self, subset=None, color=None, label=None, **kwargs):
        """ Create a new subset, and attach to self.

        This is the preferred way for creating subsets, as it
        takes care setting the label, color, and link between
        data and subset

        :param subset: optional, reference subset or subset state.
        If provided, the new subset will copy the logic of this subset.

        Returns:

           The new subset object
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

        :param subset: A :class:`~glue.core.Subset` or
        :class:`glue.core.subset.SubsetState` object

        if input is a SubsetState, it will be wrapped in a new Subset
        automatically

        NOTE:

        The preferred way for creating empty subsets is through the
        data.new_subset method
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

    def read_tree(self, filename):
        '''
        Read a tree describing the data from a file
        '''
        self.tree = Tree(filename)

    def broadcast(self, attribute=None):
        if not self.hub:
            return
        msg = DataUpdateMessage(self, attribute=attribute)
        self.hub.broadcast(msg)

    def create_subset_from_clone(self, subset, **kwargs):
        result = Subset(self, **kwargs)
        result.register()
        result.subset_state = subset.subset_state
        return result

    def update_id(self, old, new):
        """Reassign a component to a different ComponentID

        :param old: :class:`~glue.core.data.ComponentID`. The old componentID
        :param new: :class:`~glue.core.data.ComponentID`. The new componentID
        """
        changed = False
        if old in self._components:
            self._components[new] = self._components.pop(old)
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
        """
        key, view = split_component_view(key)
        if isinstance(key, basestring):
            _k = key
            key = self.find_component_id(key)
            if key is None:
                raise IncompatibleAttribute("%s not in data set %s" %
                                            (_k, self.label))

        if isinstance(key, ComponentLink):
            return key.compute(self, view)

        try:
            comp = self._components[key]
        except KeyError:
            raise IncompatibleAttribute("%s not in data set %s" %
                                        (key, self.label))

        shp = view_shape(self.shape, view)
        if view is not None:
            result = comp[view]
        else:
            result = comp.data

        assert result.shape == shp, \
            "Component view returned bad shape: %s %s" % (result.shape, shp)
        return result

    def get_component(self, component_id):
        """Fetch the component corresponding to component_id.

        :param component_id: the component_id to retrieve
        """
        if component_id is None:
            raise IncompatibleAttribute("None not in data set")
        try:
            return self._components[component_id]
        except KeyError:
            raise IncompatibleAttribute("%s not in data set" %
                                        component_id.label)


def pixel_label(i, ndim):
    if ndim == 2:
        return ['y', 'x'][i]
    if ndim == 3:
        return ['z', 'y', 'x'][i]
    return "Axis %s" % i
