""" Factory methods to build Data objects from files

Each factory method conforms to the folowing structure, which
helps the GUI Frontend easily load data:

1) The first argument is a file name to open

2) The return value is a Data object

3) The function has a .label attribute that describes (in human
language) what kinds of files it understands

4) The function has a .file_filter attribute that lists the extensions
it can open. The string is formatted like "*.fits *.fit *.hdf5"

5) The function is added to the __factories__ list

6) Optionally, the function is registered to open a given extension by
default by calling set_default_factory

Putting this together, the simplest data factory code looks like this:

    def dummy_factory(file_name):
        return glue.core.Data()
    dummy_factory.label = "Foo file"
    dummy_factory.file_filter = "*.foo *.FOO"
    __factories__.append(dummy_factory)
    set_default_factory("foo", dummy_factory)
"""
import numpy as np
import atpy

from .data import Component, Data
from .tree import DendroMerge
from .io import extract_data_fits, extract_data_hdf5
from .util import file_format
from .coordinates import coordinates_from_header, coordinates_from_wcs

__all__ = ['gridded_data', 'tabular_data', 'data_dendro_cpp']
__factories__ = []
_default_factory = {}


def load_data(path, factory):
    d = factory(path)
    d.label = data_label(path)
    return d


def data_label(path):
    """Convert a file path into a data label, by stripping out
    slashes, file extensions, etc."""
    import os
    base, fname = os.path.split(path)
    name, ext = os.path.splitext(fname)
    return name


def set_default_factory(extension, factory):
    """Register an extension that should be handled by a factory by default

    :param extension: File extension (do not include the '.')
    :param factory: The factory function to dispatch to
    """
    _default_factory[extension] = factory


def get_default_factory(extension):
    """Return the default factory function to read a given file extension.

    :param extension: The extension to lookup

    :rtype: A factory function, or None if the extension has no default
    """
    try:
        return _default_factory[extension]
    except KeyError:
        return None


def auto_data(filename):
    """Attempt to automatically construct a data object,
    by looking at the file extension and dispatching to a default factory.
    """
    import os
    base, ext = os.path.splitext(filename)
    ext = ext.strip('.')
    fac = get_default_factory(ext)
    if fac is None:
        raise KeyError("Don't know what to do with file extension: %s" % ext)
    return fac(filename)

auto_data.label = 'Auto'
auto_data.file_filter = '*.*'
__factories__.append(auto_data)

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
        from astropy.io import fits
        arrays = extract_data_fits(filename, **kwargs)
        header = fits.Header.fromfile(filename)
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


gridded_data.label = "Image"
gridded_data.file_filter = "*.fits *.FITS *hdf5 *hd5"
__factories__.append(gridded_data)
set_default_factory('fits', gridded_data)
set_default_factory('hd5', gridded_data)
set_default_factory('hdf5', gridded_data)


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


tabular_data.label = "Catalog"
tabular_data.file_filter = "*.txt *.vot *.xml *csv *tsv *.fits"
__factories__.append(tabular_data)
set_default_factory('vot', tabular_data)
set_default_factory('csv', tabular_data)
set_default_factory('txt', tabular_data)
set_default_factory('tsv', tabular_data)


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


data_dendro_cpp.label = "C++ Dendrogram"
data_dendro_cpp.file_filter = "*.fits"
__factories__.append(data_dendro_cpp)


try:
    from PIL import Image
    __all__.append('pil_data')

    def pil_data(file_name):
        """Use the Python Imaging Library to load an
        image into a data object"""
        result = Data()

        data = np.asarray(Image.open(file_name).convert('L'))
        data = np.flipud(data)
        shp = data.shape

        comps = []
        labels = []

        #3 color image
        if len(shp) == 3 and shp[2] in [3, 4]:
            comps.append(data[:, :, 0])
            labels.append('red')
            comps.append(data[:, :, 1])
            labels.append('green')
            comps.append(data[:, :, 2])
            labels.append('blue')
            if shp[2] == 4:
                comps.append(data[:, :, 3])
                labels.append('alpha')
        else:
            comps = [data]
            labels = ['PRIMARY']

        #look for AVM coordinate metadata
        #XXX not debugged
        #try:
        #    from pyavm import AVM, NoAVMPresent
        #    avm = AVM(str(file_name))  # avoid unicode
        #    wcs = avm.to_wcs()
        #    result.coords = coordinates_from_wcs(wcs)
        #except (NoAVMPresent, ImportError):
        #    pass

        for c, l in zip(comps, labels):
            result.add_component(c, l)

        return result

    pil_data.label = "Image"
    pil_data.file_filter = "*.jpg *.jpeg *.bmp *.png *.tiff"
    __factories__.append(pil_data)
    set_default_factory('jpeg', pil_data)
    set_default_factory('jpg', pil_data)
    set_default_factory('png', pil_data)
    set_default_factory('bmp', pil_data)
    set_default_factory('tiff', pil_data)

except ImportError:
        pass
