from glue.viewers.scatter.qt.data_viewer import ScatterViewer
from glue.viewers.scatter_fast.layer_artist import FastScatterLayerArtist

__all__ = ['FastScatterViewer']


class FastScatterViewer(ScatterViewer):

    LABEL = "Fast scatter plot"

    _data_artist_cls = FastScatterLayerArtist
    _subset_artist_cls = FastScatterLayerArtist
