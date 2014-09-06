from __future__ import absolute_import, division, print_function

def extract_data_fits(filename, use_hdu='all'):
    '''
    Extract non-tabular HDUs from a FITS file. If `use_hdu` is 'all', then
    all non-tabular HDUs are extracted, otherwise only the ones specified
    by `use_hdu` are extracted (`use_hdu` should then contain a list of
    integers). If the requested HDUs do not have the same dimensions, an
    Exception is raised.
    '''
    from ..external.astro import fits

    # Read in all HDUs
    hdulist = fits.open(filename, ignore_blank=True)

    # If only a subset are requested, extract those
    if use_hdu != 'all':
        hdulist = [hdulist[hdu] for hdu in use_hdu]

    # Now only keep HDUs that are not tables
    for hdu in hdulist:
        if not isinstance(hdu, fits.PrimaryHDU) and \
                not isinstance(hdu, fits.ImageHDU):
            hdulist.remove(hdu)

    # Check that dimensions of all HDU are the same
    reference_shape = hdulist[0].data.shape
    for hdu in hdulist:
        if hdu.data.shape != reference_shape:
            raise Exception("HDUs are not all the same dimensions")

    # Extract data
    arrays = {}
    for hdu in hdulist:
        arrays[hdu.name] = hdu.data

    return arrays


def extract_hdf5_datasets(handle):
    '''
    Recursive function that returns a dictionary with all the datasets
    found in an HDF5 file or group. `handle` should be an instance of
    h5py.highlevel.File or h5py.highlevel.Group.
    '''

    import h5py

    datasets = {}
    for group in handle:
        if isinstance(handle[group], h5py.highlevel.Group):
            sub_datasets = extract_hdf5_datasets(handle[group])
            for key in sub_datasets:
                datasets[key] = sub_datasets[key]
        elif isinstance(handle[group], h5py.highlevel.Dataset):
            datasets[handle[group].name] = handle[group]
    return datasets


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
