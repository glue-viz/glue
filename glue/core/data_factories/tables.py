from __future__ import absolute_import, division, print_function

from ...config import data_factory

from .helpers import has_extension

# Backward-compatibility
from .astropy_table import astropy_tabular_data

__all__ = ['tabular_data']


@data_factory(label="Catalog",
              identifier=has_extension('csv txt tsv tbl dat '
                                       'csv.gz txt.gz tbl.bz '
                                       'dat.gz'),
              priority=1)
def tabular_data(path, **kwargs):
    from .pandas import pandas_read_table
    for fac in [astropy_tabular_data, pandas_read_table]:
        try:
            return fac(path, **kwargs)
        except:
            pass
    else:
        raise IOError("Could not parse file: %s" % path)

