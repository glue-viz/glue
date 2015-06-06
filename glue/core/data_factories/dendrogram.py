from .helpers import __factories__, has_extension

try:
    from ..dendro_loader import load_dendro
    __factories__.append(load_dendro)
    load_dendro.label = 'Dendrogram'
    load_dendro.identifier = has_extension('fits hdf5 h5')
except ImportError:
    pass
