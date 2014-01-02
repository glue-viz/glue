""" Factory methods to build Data objects from files

Each factory method conforms to the folowing structure, which
helps the GUI Frontend easily load data:

1) The first argument is a file name to open

2) The return value is a Data object

3) The function has a .label attribute that describes (in human
language) what kinds of files it understands

4) The function has a callable .identifier attribute that returns
whether it can handle a requested filename and keyword set

5) The function is added to the __factories__ list

6) Optionally, the function is registered to open a given extension by
default by calling set_default_factory

Putting this together, the simplest data factory code looks like this:

    def dummy_factory(file_name):
        return glue.core.Data()
    dummy_factory.label = "Foo file"
    dummy_factory.identifier = has_extension('foo FOO')
    __factories__.append(dummy_factory)
    set_default_factory("foo", dummy_factory)
"""
import os

import numpy as np

from .data import Component, Data
from .tree import DendroMerge
from .io import extract_data_fits, extract_data_hdf5
from .util import file_format
from .coordinates import coordinates_from_header, coordinates_from_wcs
from ..external.astro import fits


__all__ = ['gridded_data', 'tabular_data', 'data_dendro_cpp']
__factories__ = []
_default_factory = {}


def load_numpy_str(x):
    """
    Load the contents of a npz file into a numpy array

    :param x: str
    """
    from cStringIO import StringIO
    s = StringIO(x.decode('base64'))
    return np.load(s)


def as_list(x):
    if isinstance(x, list):
        return x
    return [x]


def _extension(path):
    # split a path into an extension with possibly multipler periods
    #  test.fits -> fits
    #  test.fits.gz -> fits.gz
    _, path = os.path.split(path)
    if '.' not in path:
        return ''
    ext = '.'.join(path.split('.')[1:])
    return ext


def has_extension(exts):
    """
    A simple default filetype identifier function

    It returns a function that tests whether its input
    filename contains a particular extension

    Inputs
    ------
    exts : str
      A space-delimited string listing the extensions
      (e.g., 'txt', or 'txt csv fits')

    Returns
    -------
    A function suitable as a factory identifier function
    """

    def tester(x, **kwargs):
        return _extension(x) in set(exts.split())
    return tester


def is_hdf5(filename):
    # All hdf5 files begin with the same sequence
    with open(filename) as infile:
        return infile.read(8) == '\x89HDF\r\n\x1a\n'


def is_fits(filename):
    try:
        with fits.open(filename):
            return True
    except IOError:
        return False


def load_data(path, factory=None, **kwargs):
    """Use a factory to load a file and assign a label

    :param path: Path to a file
    :type path: str
    :param factory: factory function to use. Defaults to auto_data
    :type factory: function

    Extra keywords are passed through to factory functions
    """
    factory = factory or auto_data
    d = factory(path, **kwargs)
    lbl = data_label(path)
    for item in as_list(d):
        item.label = lbl
    return d


def data_label(path):
    """Convert a file path into a data label, by stripping out
    slashes, file extensions, etc."""
    _, fname = os.path.split(path)
    name, _ = os.path.splitext(fname)
    return name


def set_default_factory(extension, factory):
    """Register an extension that should be handled by a factory by default

    :param extension: File extension (do not include the '.')
    :param factory: The factory function to dispatch to
    """
    for ex in extension.split():
        _default_factory[ex] = factory


def get_default_factory(extension):
    """Return the default factory function to read a given file extension.

    :param extension: The extension to lookup

    :rtype: A factory function, or None if the extension has no default
    """
    try:
        return _default_factory[extension]
    except KeyError:
        return None


def find_factory(filename, **kwargs):
    from ..config import data_factory

    # on first pass, only try the default factory
    default = _default_factory.get(_extension(filename))
    for func, _, identifier in data_factory:
        if func is auto_data:
            continue
        if (func is default) and identifier(filename, **kwargs):
            return func

    # if that fails, try everything
    for func, _, identifier in data_factory:
        if func is auto_data:
            continue
        if identifier(filename, **kwargs):
            return func


def auto_data(filename, *args, **kwargs):
    """Attempt to automatically construct a data object,
    by trying all known factory methods.
    """
    fac = find_factory(filename, **kwargs)
    if fac is None:
        raise KeyError("Don't know how to open file: %s" % filename)
    return fac(filename, *args, **kwargs)

auto_data.label = 'Auto'
auto_data.identifier = lambda x: True
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
    if is_fits(filename):
        arrays = extract_data_fits(filename, **kwargs)
        header = fits.getheader(filename)
        result.coords = coordinates_from_header(header)
    elif is_hdf5(filename):
        arrays = extract_data_hdf5(filename, **kwargs)
    else:
        raise Exception("Unkonwn format: %s" % format)

    for component_name in arrays:
        comp = Component(arrays[component_name])
        result.add_component(comp, component_name)
    return result


def is_gridded_data(filename, **kwargs):
    if is_hdf5(filename):
        return True

    if is_fits(filename):
        with fits.open(filename) as hdulist:
            for hdu in hdulist:
                if not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)):
                    return False
            return True
    return False


gridded_data.label = "FITS/HDF5 Image"
gridded_data.identifier = is_gridded_data
__factories__.append(gridded_data)
set_default_factory('fits', gridded_data)
set_default_factory('hd5', gridded_data)
set_default_factory('hdf5', gridded_data)


def _ascii_identifier_v02(origin, args, kwargs):
    # this works for astropy v0.2
    if isinstance(args[0], basestring):
        return args[0].endswith(('csv', 'tsv', 'txt', 'tbl', 'dat',
                                 'csv.gz', 'tsv.gz', 'txt.gz', 'tbl.gz',
                                 'dat.gz'))
    else:
        return False


def _ascii_identifier_v03(origin, *args, **kwargs):
    # this works for astropy v0.3
    return _ascii_identifier_v02(origin, args, kwargs)


def tabular_data(*args, **kwargs):
    """
    Build a data set from a table. We restrict ourselves to tables
    with 1D columns.

    All arguments are passed to
        astropy.table.Table.read(...).
    """
    from distutils.version import LooseVersion
    from astropy import __version__
    if LooseVersion(__version__) < LooseVersion("0.2"):
        raise RuntimeError("Glue requires astropy >= v0.2. Please update")

    result = Data()

    # Read the table
    from astropy.table import Table

    # Add identifiers for ASCII data
    from astropy.io import registry
    if LooseVersion(__version__) < LooseVersion("0.3"):
        registry.register_identifier('ascii', Table, _ascii_identifier_v02,
                                     force=True)
    else:
        registry.register_identifier('ascii', Table, _ascii_identifier_v03,
                                     force=True)

    # Import FITS compatibility (for Astropy 0.2.x)
    from ..external import fits_io

    table = Table.read(*args, **kwargs)

    # Loop through columns and make component list
    for column_name in table.columns:
        c = table[column_name]
        u = c.units

        if table.masked:
            # fill array for now
            try:
                c = c.filled(fill_value=np.nan)
            except ValueError:  # assigning nan to integer dtype
                c = c.filled(fill_value=-1)

        c = Component(np.asarray(c), units=u)
        result.add_component(c, column_name)

    return result

tabular_data.label = "Catalog"
tabular_data.identifier = has_extension('xml vot csv txt tsv tbl dat fits '
                                        'xml.gz vot.gz csv.gz txt.gz tbl.bz '
                                        'dat.gz fits.gz')

__factories__.append(tabular_data)
set_default_factory('xml', tabular_data)
set_default_factory('vot', tabular_data)
set_default_factory('csv', tabular_data)
set_default_factory('txt', tabular_data)
set_default_factory('tsv', tabular_data)
set_default_factory('tbl', tabular_data)
set_default_factory('dat', tabular_data)


def data_dendro_cpp(file):
    """
    Construct a data object from a C++-generated dendrogram file

    :param file: Name of a file to read

    *Returns*
    A glue data structure representing the file
    """

    data = extract_data_fits(file, use_hdu=[0, 1])
    m = extract_data_fits(file, use_hdu=[2])
    merge_list = m['CLUSTERS']
    merge_list = merge_list[(merge_list.shape[0] + 1) / 2:]

    im = data['INDEX_MAP']
    val = data['PRIMARY']

    result = gridded_data(file, use_hdu=['PRIMARY', 'INDEX_MAP'])
    result.tree = DendroMerge(merge_list, index_map=im)
    return result

# XXX This doesn't belong in core. its too specific
#data_dendro_cpp.label = "C++ Dendrogram"
#data_dendro_cpp.identifier = has_extension('fits')
#__factories__.append(data_dendro_cpp)


img_fmt = ['jpg', 'jpeg', 'bmp', 'png', 'tiff', 'tif']


def img_loader(file_name):
    """Load an image to a numpy array, using either PIL or skimage

    :param file_name: Path of file to load
    :rtype: Numpy array
    """
    try:
        from skimage.io import imread
        return np.asarray(imread(file_name))
    except ImportError:
        pass

    try:
        from PIL import Image
        return np.asarray(Image.open(file_name))
    except ImportError:
        raise ImportError("Reading %s requires PIL or scikit-image" %
                          file_name)


def img_data(file_name):
    """Load common image files into a Glue data object"""
    result = Data()

    data = img_loader(file_name)
    data = np.flipud(data)
    shp = data.shape

    comps = []
    labels = []

    # split 3 color images into each color plane
    if len(shp) == 3 and shp[2] in [3, 4]:
        comps.extend([data[:, :, 0], data[:, :, 1], data[:, :, 2]])
        labels.extend(['red', 'green', 'blue'])
        if shp[2] == 4:
            comps.append(data[:, :, 3])
            labels.append('alpha')
    else:
        comps = [data]
        labels = ['PRIMARY']

    # look for AVM coordinate metadata
    try:
        from pyavm import AVM
        avm = AVM(str(file_name))  # avoid unicode
        wcs = avm.to_wcs()
    except:
        pass
    else:
        result.coords = coordinates_from_wcs(wcs)

    for c, l in zip(comps, labels):
        result.add_component(c, l)

    return result

img_data.label = "Image"
img_data.identifier = has_extension(' '.join(img_fmt))
for i in img_fmt:
    set_default_factory(i, img_data)

__factories__.append(img_data)
__all__.append('img_data')
