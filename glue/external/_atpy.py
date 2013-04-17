"""
This module monkeypatches a bunch of ATpy functions,
to add some support for using astropy libraries
instead of legacy libraries

This module will be deprecated once the astropy Table class is more developed
"""

#trick atpy into using astropy modules if they exist
#this is a bit hacky, for users with old ATpy installed
try:
    from .astro import votable, ascii
    import sys

    sys.modules['vo'] = votable
    sys.modules['vo.table'] = votable
    sys.modules['vo.tree'] = votable.tree
    sys.modules['asciitable'] = ascii

except ImportError:
    pass

import numpy as np
from atpy import *

#patch add_column to handle deal with unicode names
_add_column = Table.add_column


def add_column(self, name, data, **kwargs):
    try:
        data = np.asarray(data, dtype=np.float)
        kwargs['dtype'] = data.dtype
    except ValueError:
        pass

    return _add_column(self, str(name), data, **kwargs)

Table.add_column = add_column


#patch asciitable.read to deal with different call signatures
#call signatures in asciitable/astropy version
_read = ascii.read


def patch_read(table, numpy=True, guess=None, **kwargs):
    #astropy version does not have numpy kwarg,
    if not numpy:
        raise TypeError("Attempting to use asciitable with numpy=False")
    return _read(table, guess=guess, **kwargs)

ascii.read = patch_read
