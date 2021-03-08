from glue.core.data_factories.helpers import has_extension
from glue.config import data_factory


__all__ = ['tabular_data']


@data_factory(label="ASCII Table",
              identifier=has_extension('csv txt tsv tbl dat '
                                       'csv.gz txt.gz tbl.bz '
                                       'dat.gz'),
              priority=1)
def tabular_data(path, **kwargs):
    from glue.core.data_factories.astropy_table import astropy_tabular_data
    from glue.core.data_factories.pandas import pandas_read_table
    # Try pandas first, as it is an order of mangitude faster than astropy
    for fac in [pandas_read_table, astropy_tabular_data]:
        try:
            return fac(path, **kwargs)
        except Exception:
            pass
    else:
        raise IOError("Could not parse file: %s" % path)
