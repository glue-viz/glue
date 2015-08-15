from __future__ import absolute_import, division, print_function

from .helpers import __factories__, has_extension

__all__ = []

try:
    from .dendro_loader import load_dendro, is_dendro
    __factories__.append(load_dendro)
    load_dendro.label = 'Dendrogram'
    load_dendro.identifier = is_dendro
    load_dendro.priority = 1000
except ImportError:
    pass
