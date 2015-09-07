"""
Load files created by the astrodendro package.

astrodendro must be installed in order to use this loader
"""
import numpy as np
from astrodendro import Dendrogram
from ..data import Data
from ..config import data_factory

from .gridded import is_fits, is_hdf5

__all__ = ['load_dendro', 'is_dendro']


def is_dendro(file, **kwargs):

    if is_hdf5(file):

        import h5py

        f = h5py.File(file, 'r')

        return 'data' in f and 'index_map' in f and 'newick' in f

    elif is_fits(file):

        from ...external.astro import fits

        hdulist = fits.open(file)

        # In recent versions of Astropy, we could do 'DATA' in hdulist etc. but
        # this doesn't work with Astropy 0.3, so we use the following method
        # instead:
        try:
            hdulist['DATA']
            hdulist['INDEX_MAP']
            hdulist['NEWICK']
        except KeyError:
            pass  # continue
        else:
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
def load_dendro(file):
    """
    Load a dendrogram saved by the astrodendro package

    :param file: Path to a dendrogram file
    :returns: A list of 2 glue Data objects: the original dataset, and dendrogram.
    """

    dg = Dendrogram.load_from(file)
    structs = np.arange(len(dg))
    parent = np.array([dg[i].parent.idx
                       if dg[i].parent is not None else -1
                       for i in structs])
    height = np.array([dg[i].height for i in structs])
    pk = np.array([dg[i].get_peak(True)[1] for i in structs])

    dendro = Data(parent=parent,
                  height=height,
                  peak=pk,
                  label='Dendrogram')

    im = Data(intensity=dg.data, structure=dg.index_map)
    im.join_on_key(dendro, 'structure', dendro.pixel_component_ids[0])
    return [dendro, im]


