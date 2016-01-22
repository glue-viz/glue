from __future__ import absolute_import, division, print_function

from glue.core.data_factories.hdf5 import is_hdf5, extract_hdf5_datasets
from glue.core.data_factories.fits import is_fits, is_image_hdu
from glue.core.coordinates import coordinates_from_header
from glue.core.data import Component, Data
from glue.config import data_factory
from glue.utils import file_format


__all__ = ['is_gridded_data', 'gridded_data']


def extract_data_hdf5(filename, use_datasets='all'):
    '''
    Extract non-tabular datasets from an HDF5 file. If `use_datasets` is
    'all', then all non-tabular datasets are extracted, otherwise only the
    ones specified by `use_datasets` are extracted (`use_datasets` should
    then contain a list of paths). If the requested datasets do not have
    the same dimensions, an Exception is raised.
    '''

    import h5py

    # Open file
    file_handle = h5py.File(filename, 'r')

    # Define function to read

    # Read in all datasets
    datasets = extract_hdf5_datasets(file_handle)

    # Only keep non-tabular datasets
    remove = []
    for key in datasets:
        if datasets[key].dtype.fields is not None:
            remove.append(key)
    for key in remove:
        datasets.pop(key)

    # Check that dimensions of all datasets are the same
    reference_shape = datasets[list(datasets.keys())[0]].value.shape
    for key in datasets:
        if datasets[key].value.shape != reference_shape:
            raise Exception("Datasets are not all the same dimensions")

    # Extract data
    arrays = {}
    for key in datasets:
        arrays[key] = datasets[key].value

    # Close HDF5 file
    file_handle.close()

    return arrays


def filter_hdulist_by_shape(hdulist, use_hdu='all'):
    """
    Remove empty HDUs, and ensure that all HDUs can be
    packed into a single Data object (ie have the same shape)

    Parameters
    ----------
    use_hdu : 'all' or list of integers (optional)
        Which HDUs to use

    Returns
    -------
    a new HDUList
    """
    from astropy.io import fits

    # If only a subset are requested, extract those
    if use_hdu != 'all':
        hdulist = [hdulist[hdu] for hdu in use_hdu]

    # Now only keep HDUs that are not tables or empty.
    valid_hdus = []
    for hdu in hdulist:
        if (isinstance(hdu, fits.PrimaryHDU) or
                isinstance(hdu, fits.ImageHDU)) and \
                hdu.data is not None:
            valid_hdus.append(hdu)

    # Check that dimensions of all HDU are the same
    # Allow for HDU's that have no data.
    reference_shape = valid_hdus[0].data.shape
    for hdu in valid_hdus:
        if hdu.data.shape != reference_shape:
            raise Exception("HDUs are not all the same dimensions")

    return valid_hdus


def extract_data_fits(filename, use_hdu='all'):
    '''
    Extract non-tabular HDUs from a FITS file. If `use_hdu` is 'all', then
    all non-tabular HDUs are extracted, otherwise only the ones specified
    by `use_hdu` are extracted (`use_hdu` should then contain a list of
    integers). If the requested HDUs do not have the same dimensions, an
    Exception is raised.
    '''
    from astropy.io import fits

    # Read in all HDUs
    hdulist = fits.open(filename, ignore_blank=True)
    hdulist = filter_hdulist_by_shape(hdulist)

    # Extract data
    arrays = {}
    for hdu in hdulist:
        arrays[hdu.name] = hdu.data

    return arrays


def is_gridded_data(filename, **kwargs):

    if is_hdf5(filename):
        return True

    if is_fits(filename):
        from astropy.io import fits
        with fits.open(filename) as hdulist:
            return is_image_hdu(hdulist[0])

    return False


@data_factory(label="FITS/HDF5 Image",
              identifier=is_gridded_data,
              deprecated=True)
def gridded_data(filename, format='auto', **kwargs):

    result = Data()

    # Try and automatically find the format if not specified
    if format == 'auto':
        format = file_format(filename)

    # Read in the data
    if is_fits(filename):
        from astropy.io import fits
        arrays = extract_data_fits(filename, **kwargs)
        header = fits.getheader(filename)
        result.coords = coordinates_from_header(header)
    elif is_hdf5(filename):
        arrays = extract_data_hdf5(filename, **kwargs)
    else:
        raise Exception("Unkonwn format: %s" % format)

    for component_name in arrays:
        comp = Component.autotyped(arrays[component_name])
        result.add_component(comp, component_name)

    return result
