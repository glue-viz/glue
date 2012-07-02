import numpy as np
import atpy
import pyfits

import glue
from glue.io import extract_data_fits, extract_data_hdf5
from glue.coordinates import WCSCoordinates
from glue.coordinates import WCSCubeCoordinates
from glue.coordinates import Coordinates
from glue.visual import VisualAttributes
from glue.exceptions import IncompatibleAttribute
from glue.component_link import ComponentLink
from glue.util import file_format

class ComponentID(object):
    """ References a Component object within a data object

    Components are retrieved from data objects via ComponentIDs::

       component = data.get_component(component_id)
    """

    def __init__(self, label):
        """:param label: Name for the ID
           :type label: str"""
        self._label = label

    @property
    def label(self):
        """ Return the label """
        return self._label

    def __str__(self):
        return str(self._label)

    def __repr__(self):
        return str(self._label)

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


class DerivedComponent(Component):
    """ A component which derives its data from a function """
    def __init__(self, data, link, units=None):
        """
        :param data: The data object to use for calculation
        :type data: :class:`~glue.data.Data`

        :param link: The link that carries out the function
        :type link: :class:`~glue.component_link.ComponentLink`

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


class Data(object):
    """Stores data and manages subsets.

    The data object stores data as a collection of
    :class:`~glue.data.Component` objects.  Each component stored in a
    dataset must have the same shape.

    Catalog data sets are stored such that each column is a distinct
    1-dimensional ``Component``.

    There two ways to extract the actual numerical data stored in a
    :class:`~glue.Data` object::

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
        self._components = {}
        self._pixel_component_ids = []
        self._world_component_ids = []

        # Tree description of the data
        self.tree = None

        # Subsets of the data
        self.subsets = []

        # Hub that the data is attached to
        self.hub = None

        self.style = VisualAttributes(parent=self, washout=True)

        self._coordinate_links = None

        self.style.label = label

        self.data = self

        # The default-editable subset
        self.edit_subset = glue.Subset(self, label='Editable Subset')
        self.add_subset(self.edit_subset)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape(self):
        return self._shape

    @property
    def label(self):
        """ Convenience access to data set's label """
        return self.style.label

    @label.setter
    def label(self, value):
        """ Set the label to value """
        self.style.label = value

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
        :type component_id: :class:`~glue.data.ComponentID`
        """
        if component_id in self._components:
            self._components.pop(component_id)

    def add_component(self, component, label):
        """ Add a new component to this data set.

        :param component: object to add
        :param label:
              The label. If this is a string,
              a new ComponentID with this label will be
              created and associated with the Component

        :type component: :class:`~glue.data.Component`
        :type label: :class:`str` or :class:`~glue.data.ComponentID`

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
        elif type(label) == str:
            component_id = ComponentID(label)
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
            msg = glue.message.DataAddComponentMessage(self, component_id)
            self.hub.broadcast(msg)

        return component_id

    def add_component_link(self, link):
        """ Shortcut method for generating a new DerivedComponent
        from a ComponentLink object, and adding it to a data set.

        :param link: ComponentLink object

        Returns:

            The DerivedComponent that was added
        """
        dc = DerivedComponent(self, link)
        to_ = link.get_to_id()
        self.add_component(dc, to_)
        return dc

    def _create_pixel_and_world_components(self):
        shape = self.shape
        slices = [slice(0, s, 1) for s in shape]
        grids = np.mgrid[slices]
        for i in range(len(shape)):
            comp = Component(grids[i])
            cid = self.add_component(comp, "Pixel Axis %i" % i)
            self._pixel_component_ids.append(cid)
        if self.coords:
            world = self.coords.pixel2world(*grids[::-1])[::-1]
            for i in range(len(shape)):
                comp = Component(world[i])
                label = self.coords.axis_label(i)
                cid = self.add_component(comp, label)
                self._world_component_ids.append(cid)

    @property
    def components(self):
        """ Returns a list of ComponentIDs for all components
        (primary and derived) in the data"""
        return self._components.keys()

    @property
    def primary_components(self):
        """Returns a list of ComponentIDs with stored data (as opposed
        to a :class:`~glue.data.DerivedComponent` )
        """
        return [c for c in self.component_ids() if
                not isinstance(self._components[c], DerivedComponent)]

    @property
    def derived_components(self):
        """A list of ComponentIDs for each :class:`~glue.data.DerivedComponent`
        in the data. (Read only)
        """
        return [c for c in self.component_ids() if
                isinstance(self._components[c], DerivedComponent)]

    def find_component_id(self, label):
        """ Retrieve component_ids associated by label name.

        :param label: string to search for

        Returns:

            A list of all component_ids with matching labels
        """
        result = [cid for cid in self.component_ids() if
                  cid.label.upper() == label.upper()]
        return result

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

    def new_subset(self):
        """ Create a new subset, and attach to self.

        This is the preferred way for creating subsets, as it
        takes care of setting up the links between data and subset

        Returns:

           The new subset object
        """
        subset = glue.Subset(self)
        self.add_subset(subset)
        return subset

    def add_subset(self, subset):
        """ Assign a pre-existing subset to this data object.
        The preferred way of dealing with subsets is through the new_subset
        method, which both creates and adds the subset """
        if subset in self.subsets:
            return  # prevents infinite recursion
        self.subsets.append(subset)
        if self.hub is not None:
            msg = glue.message.SubsetCreateMessage(subset)
            self.hub.broadcast(msg)
        subset.do_broadcast(True)

    def remove_subset(self, subset):
        if self.hub is not None:
            msg = glue.message.SubsetDeleteMessage(subset)
            self.hub.broadcast(msg)
        self.subsets.remove(subset)

    def register_to_hub(self, hub):
        """ Connect to a hub.

        This method usually doesn't have to be called directly, as
        DataCollections manage the registration of data objects
        """
        if not isinstance(hub, glue.Hub):
            raise TypeError("input is not a Hub object: %s" % type(hub))
        self.hub = hub

    def read_tree(self, filename):
        '''
        Read a tree describing the data from a file
        '''
        self.tree = glue.Tree(filename)

    def broadcast(self, attribute=None):
        if not self.hub:
            return
        msg = glue.message.DataUpdateMessage(self, attribute=attribute)
        self.hub.broadcast(msg)

    def create_subset_from_clone(self, subset, **kwargs):
        result = glue.Subset(self, **kwargs)
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

        :type key: :class:`~glue.data.ComponentID`
        """
        try:
            return self._components[key].data
        except KeyError:
            raise IncompatibleAttribute("%s not in data set %s" %
                                        (key.label, self.label))

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
            arrays = extract_data_fits(filename, **kwargs)
            header = pyfits.open(filename)[0].header
            self._parse_coordinates(header)
        elif format in ['hdf', 'hdf5', 'h5']:
            arrays = extract_data_hdf5(filename, **kwargs)
        else:
            raise Exception("Unkonwn format: %s" % format)

        for component_name in arrays:
            comp = Component(arrays[component_name])
            self.add_component(comp, component_name)


    def _parse_coordinates(self, header):
        if 'NAXIS' in header and header['NAXIS'] == 2:
            self.coords = WCSCoordinates(header)
        elif 'NAXIS' in header and header['NAXIS'] == 3:
            self.coords = WCSCubeCoordinates(header)
