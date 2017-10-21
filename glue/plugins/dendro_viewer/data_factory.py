"""
Load files created by the astrodendro package.

astrodendro must be installed in order to use this loader
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from astrodendro import Dendrogram

from glue.core.data_factories.hdf5 import is_hdf5
from glue.core.data_factories.fits import is_fits
from glue.core.data_factories.helpers import data_label
from glue.core.data import Data
from glue.config import data_factory


__all__ = ['load_dendro', 'is_dendro']


def is_dendro(file, **kwargs):

    if is_hdf5(file):

        import h5py

        f = h5py.File(file, 'r')

        return 'data' in f and 'index_map' in f and 'newick' in f

    elif is_fits(file):

        from astropy.io import fits

        with fits.open(file, ignore_missing_end=True) as hdulist:

            # For recent versions of astrodendro the HDUs have a recongnizable
            # set of names.

            if 'DATA' in hdulist and 'INDEX_MAP' in hdulist and 'NEWICK' in hdulist:
                return True

            # For older versions of astrodendro, the HDUs did not have names

            # Here we use heuristics to figure out if this is likely to be a
            # dendrogram. Specifically, there should be three HDU extensions.
            # The primary HDU should be empty, HDU 1 and HDU 2 should have
            # matching shapes, and HDU 3 should have a 1D array. Also, if the
            # HDUs do have names then this is not a dendrogram since the old
            # files did not have names

            # This branch can be removed once we think most dendrogram files
            # will have HDU names.

            if len(hdulist) != 4:
                return False

            if hdulist[1].name != '' or hdulist[2].name != '' or hdulist[3].name != '':
                return False

            if hdulist[0].data is not None:
                return False

            if hdulist[1].data is None or hdulist[2].data is None or hdulist[3].data is None:
                return False

            if hdulist[1].data.shape != hdulist[2].data.shape:
                return False

            if hdulist[3].data.ndim != 1:
                return False

        # We're probably ok, so return True
        return True

    else:

        return False


@data_factory(label='Dendrogram', identifier=is_dendro, priority=1000)
def load_dendro(filename):
    """
    Load a dendrogram saved by the astrodendro package

    :param file: Path to a dendrogram file
    :returns: A list of 2 glue Data objects: the original dataset, and dendrogram.
    """

    label = data_label(filename)

    dg = Dendrogram.load_from(filename)
    structs = np.arange(len(dg))
    parent = np.array([dg[i].parent.idx
                       if dg[i].parent is not None else -1
                       for i in structs])
    height = np.array([dg[i].height for i in structs])
    pk = np.array([dg[i].get_peak(True)[1] for i in structs])

    dendro = Data(parent=parent,
                  height=height,
                  peak=pk,
                  label="{} [dendrogram]".format(label))

    im = Data(intensity=dg.data,
              structure=dg.index_map,
              label="{} [data]".format(label))
    im.join_on_key(dendro, 'structure', dendro.pixel_component_ids[0])
    return [dendro, im]
