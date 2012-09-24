""" Factory methods to build Data objects from files """
import pyfits
import atpy

from .data import Component, Data
from .tree import DendroMerge
from .io import extract_data_fits, extract_data_hdf5
from .util import file_format
from .coordinates import coordinates_from_header

__all__ = ['gridded_data', 'tabular_data', 'data_dendro_cpp']

def gridded_data(filename, format='auto', **kwargs):
    """
    Construct an n-dimensional data object from `filename`. If the
    format cannot be determined from the extension, it can be
    specified using the `format=` option. Valid formats are 'fits' and
    'hdf5'.
    """
    result = Data()

    # Try and automatically find the format if not specified
    if format == 'auto':
        format = file_format(filename)

    # Read in the data
    if format in ['fits', 'fit']:
        arrays = extract_data_fits(filename, **kwargs)
        header = pyfits.open(filename)[0].header
        result.coords = coordinates_from_header(header)
    elif format in ['hdf', 'hdf5', 'h5']:
        arrays = extract_data_hdf5(filename, **kwargs)
    else:
        raise Exception("Unkonwn format: %s" % format)

    for component_name in arrays:
        #XXX ALMA HACK
        arr = arrays[component_name]
        shp = arr.shape
        if len(shp) == 4:
            arr.shape = (shp[1], shp[2], shp[3])
        comp = Component(arrays[component_name])
        result.add_component(comp, component_name)

    return result


def data_dendro_cpp(file):
    """
    Construct a data object from a C++-generated dendrogram file

    :param file: Name of a file to read

    *Returns*
    A glue data structure representing the file
    """
    #XXX This doesn't belong in core. its too specific

    data = extract_data_fits(file, use_hdu=[0, 1])
    m = extract_data_fits(file, use_hdu=[2])
    merge_list = m['CLUSTERS']
    merge_list = merge_list[(merge_list.shape[0] + 1) / 2:]

    im = data['INDEX_MAP']
    val = data['PRIMARY']

    c = Component(val)

    result = gridded_data(file, use_hdu=['PRIMARY', 'INDEX_MAP'])
    result.tree = DendroMerge(merge_list, index_map=im)
    return result


def tabular_data(*args, **kwargs):
    """
    Build a data set from a table. We restrict ourselves to tables
    with 1D columns.

    All arguments are passed to
        ATpy.Table(...).
    """
    result = Data()

    # Read the table
    table = atpy.Table()
    table.read(*args, **kwargs)

    # Loop through columns and make component list
    for column_name in table.columns:
        c = Component(table[column_name],
                      units=table.columns[column_name].unit)
        result.add_component(c, column_name)

    return result
