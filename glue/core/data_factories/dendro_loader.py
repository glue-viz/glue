"""
Load files created by the astrodendro package.

astrodendro must be installed in order to use this loader
"""
import numpy as np
from astrodendro import Dendrogram
from ..data import Data

__all__ = ['load_dendro']


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
