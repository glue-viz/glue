import operator
import logging

import numpy as np

from .io import extract_data_fits, extract_data_hdf5
from .coordinates import Coordinates, coordinates_from_header
from .visual import VisualAttributes
from .visual import RED, GREEN, BLUE, YELLOW, BROWN, ORANGE, PURPLE, PINK
from .exceptions import IncompatibleAttribute
from .component_link import ComponentLink, BinaryComponentLink
from .subset import Subset, InequalitySubsetState, SubsetState
from .hub import Hub
from .tree import Tree
from .registry import Registry
from .util import split_component_view, view_shape
from .message import (DataUpdateMessage,
                      DataAddComponentMessage,
                      SubsetCreateMessage)

from .util import file_format
from .odict import OrderedDict

__all__ = ['ComponentID', 'Component', 'DerivedComponent', 'Data',
           'TabularData', 'GriddedData', 'CoordinateComponent']

COLORS = [RED, GREEN, BLUE, YELLOW, BROWN, ORANGE, PURPLE, PINK]


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
        self._data = data

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

    def __init__(self, label=""):
        """:param label: label for data
        :type label: str"""
        # Coordinate conversion object
        self.coords = Coordinates()
        self._shape = ()

        # Components
        self._components = OrderedDict()
        self._pixel_component_ids = []
        self._world_component_ids = []

        # Tree description of the data
        self.tree = None

        # Subsets of the data
        self.subsets = []

        # Hub that the data is attached to
        self.hub = None

        self.style = VisualAttributes(parent=self)

        self._coordinate_links = None

        self.data = self
        self._label = None
        self.label = label  # trigger disambiguation

        self.edit_subset = None

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

        Each data label in a glue session must be unique. The input
        will be auto-disambiguated if necessary
        """
        value = Registry().register(self, value, group=Data)
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

        :type component: :class:`~glue.core.data.Component`
        :type label: :class:`str` or :class:`~glue.core.data.ComponentID`

        *Raises*

           TypeError, if label is invalid, or if the component has
           an incompatible shape

        *Returns*

           The ComponentID associated with the newly-added component
        """
        if not(self._check_can_add(component)):
            raise TypeError("Compoment is incompatible with "
                            "other components in this data")

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

        return component_id

    def add_component_link(self, link, cid=None):
        """ Shortcut method for generating a new DerivedComponent
        from a ComponentLink object, and adding it to a data set.

        :param link: ComponentLink object

        Returns:

            The DerivedComponent that was added
        """
        if cid is not None:
            if type(cid) is str:
                cid = ComponentID(cid)
            link = ComponentLink(link.get_from_ids(), cid, link.get_using())

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
        return self._components.keys()

    @property
    def visible_components(self):
        """ Returns a list of ComponentIDs for all components
        (primary and derived) in the data"""
        return [c for c in self._components.keys() if not c.hidden]

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

            The associated ComponentID, or None of not found
        """
        result = [cid for cid in self.component_ids() if
                  cid.label == label]
        if len(result) > 0:
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
            link = ComponentLink(self._pixel_component_ids,
                                 self._world_component_ids[i],
                                 make_toworld_func(i))
            result.append(link)
            link = ComponentLink(self._world_component_ids,
                                 self._pixel_component_ids[i],
                                 make_topixel_func(i))
            result.append(link)

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

        self.subsets.append(subset)

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

    def __str__(self):
        s = ""
        s += "Number of dimensions: %i\n" % self.ndim
        s += "Shape: %s\n" % ' x '.join([str(x) for x in self.shape])
        s += "Components:\n"
        for component in self._components:
            s += " * %s\n" % component
        return s[:-1]

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
        try:
            return self._components[component_id]
        except KeyError:
            raise IncompatibleAttribute("%s not in data set" %
                                        component_id.label)


class TabularData(Data):
    '''
    A class to represent any form of tabular data. We restrict
    ourselves to tables with 1D columns.
    '''

    def read_data(self, *args, **kwargs):
        '''
        Read a table from a file or database. All arguments are passed to
        ATpy.Table(...).
        '''
        try:
            import atpy
        except ImportError:
            raise ImportError("TabularData requires ATPy")
        atpy.registry.register_extensions('ascii', ['csv', 'tsv', 'txt'],
                                          override=True)

        # Read the table
        table = atpy.Table()
        table.read(*args, **kwargs)

        # Loop through columns and make component list
        for column_name in table.columns:
            c = Component(table[column_name],
                          units=table.columns[column_name].unit)
            self.add_component(c, column_name)


class GriddedData(Data):
    '''
    A class to represent uniformly gridded data (images, data cubes, etc.)
    '''

    def read_data(self, filename, format='auto', **kwargs):
        '''
        Read n-dimensional data from `filename`. If the format cannot be
        determined from the extension, it can be specified using the
        `format=` option. Valid formats are 'fits' and 'hdf5'.
        '''

        # Try and automatically find the format if not specified
        if format == 'auto':
            format = file_format(filename)

        # Read in the data
        if format in ['fits', 'fit']:
            from astropy.io import fits
            arrays = extract_data_fits(filename, **kwargs)
            header = fits.open(filename, memmap=True)[0].header
            self.coords = coordinates_from_header(header)
        elif format in ['hdf', 'hdf5', 'h5']:
            arrays = extract_data_hdf5(filename, **kwargs)
        else:
            raise Exception("Unkonwn format: %s" % format)

        for component_name in arrays:
            comp = Component(arrays[component_name])
            self.add_component(comp, component_name)


def pixel_label(i, ndim):
    if ndim == 2:
        return ['y', 'x'][i]
    if ndim == 3:
        return ['z', 'y', 'x'][i]
    return "Axis %s" % i
