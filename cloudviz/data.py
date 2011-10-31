import string

import numpy as np
import atpy
import pyfits

import cloudviz
from cloudviz.io import extract_data_fits, extract_data_hdf5
from cloudviz.coordinates import WCSCoordinates
from cloudviz.coordinates import WCSCubeCoordinates
from cloudviz.coordinates import Coordinates

class Component(object):

    def __init__(self, data, units=None):

        # The physical units of the data
        self.units = units

        # The actual data
        self.data = data

        def __getitem__(self, key):
            """ Extract the data values for each element in a subset

            Parameters:
            -----------
            key: Subset instance
                 The subset to use when extracting elements from the
                 component

            """
            try:
                return self.data[key.to_mask()]
            except AttributeError:
                raise AttributeError("Components can only be indexed "
                                     "by subset objects that implement "
                                     "the to_mask() method")


class Data(object):

    def __init__(self, label=None):
        # Coordinate conversion object
        self.coords = Coordinates()

        # Number of dimensions
        self.ndim = None

        # Dataset shape
        self.shape = None

        # Components
        self.components = {}

        # Tree description of the data
        self.tree = None

        # Subsets of the data
        self.subsets = []

        # The default-edible subset
        self.active_subset = None

        # Hub that the data is attached to
        self.hub = None

        self.metadata = {}

        self.label = label

    def new_subset(self):
        subset = cloudviz.Subset(self)
        self.add_subset(subset)
        return subset

    def set_active_subset(self, subset):
        if subset == self.active_subset: return
        if subset not in self.subsets:
            raise Exception("Input not in data's collection of subsets")
        self.active_subset = subset
        if self.hub is not None:
            msg = cloudviz.message.ActiveSubsetUpdateMessage(subset)
            self.hub.broadcast(msg)

    def get_active_subset(self):
        return self.active_subset

    def add_subset(self, subset):
        subset.do_broadcast(True)
        first = len(self.subsets) == 0
        self.subsets.append(subset)
        if self.hub is not None:
            msg = cloudviz.message.SubsetCreateMessage(subset)
            self.hub.broadcast(msg)
        if first:
            self.set_active_subset(subset)


    def remove_subset(self, subset):
        if self.hub is not None:
            msg = cloudviz.SubsetDeleteMessage(subset)
            self.hub.boradcast(msg)
        self.subsets.pop(subset)

    def register_to_hub(self, hub):
        if not isinstance(hub, cloudviz.Hub):
            raise TypeError("input is not a Hub object: %s" % type(hub))
        self.hub = hub

    def read_tree(self, filename):
        '''
        Read a tree describing the data from a file
        '''
        self.tree = cloudviz.Tree(filename)

    def __str__(self):
        s = ""
        s += "Number of dimensions: %i\n" % self.ndim
        s += "Shape: %s\n" % string.join([str(x) for x in self.shape], ' x ')
        s += "Components:\n"
        for component in self.components:
            s += " * %s\n" % component
        return s[:-1]

    def __setattr__(self, name, value):
        if name == "hub" and hasattr(self, 'hub') \
           and self.hub is not value and self.hub is not None:
            raise AttributeError("Data has already been assigned "
                                 "to a different hub")
        object.__setattr__(self, name, value)

    def __getitem__(self, key):
        """ Shortcut syntax to access the raw data in a component

        Parameters:
        -----------
        key : string
          The component to fetch data from
        """        
        if type(key) != str or key not in self.components:
            raise KeyError("Input must be the name of "
                           " a valid component: %s" % str(key))
        return self.components[key].data



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
        table = atpy.Table(*args, **kwargs)

        # Loop through columns and make component list
        for column_name in table.columns:
            c = Component(table[column_name],
                          units=table.columns[column_name].unit)
            self.components[column_name] = c

        # Set number of dimensions
        self.ndim = 1

        # Set data shape
        self.shape = (len(table),)


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
            if filename.lower().endswith('.gz'):
                format = filename.lower().rsplit('.', 2)[1]
            else:
                format = filename.lower().rsplit('.', 1)[1]

        # Read in the data
        if format in ['fits', 'fit']:
            arrays = extract_data_fits(filename, **kwargs)
            for component_name in arrays:
                self.components[component_name] = \
                    Component(arrays[component_name])

            # parse header, create coordinate object
            header = pyfits.open(filename)[0].header
            if 'NAXIS' in header and header['NAXIS'] == 2:
                self.coords = WCSCoordinates(header)
            elif 'NAXIS' in header and header['NAXIS'] == 3:
                self.coords = WCSCubeCoordinates(header)            
            
        elif format in ['hdf', 'hdf5', 'h5']:
            arrays = extract_data_hdf5(filename, **kwargs)
            for component_name in arrays:
                self.components[component_name] = \
                    Component(arrays[component_name])
        else:
            raise Exception("Unkonwn format: %s" % format)

        # Set number of dimensions
        self.ndim = self.components[self.components.keys()[0]].data.ndim

        # Set data shape
        self.shape = self.components[self.components.keys()[0]].data.shape

        # If 2D, then set XPIX and YPIX
        if self.ndim == 2:
            x = np.arange(self.shape[1])
            y = np.arange(self.shape[0])
            x, y = np.meshgrid(x, y)
            self.components['XPIX'] = Component(x)
            self.components['YPIX'] = Component(y)


class AMRData(Data):
    pass
