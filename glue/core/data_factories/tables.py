from __future__ import absolute_import, division, print_function

import numpy as np

from ..data import Component, Data
from ...external import six
from ...config import data_factory

from .helpers import has_extension

__all__ = ['tabular_data', 'sextractor_factory', 'astropy_tabular_data',
           'formatted_table_factory']


def is_astropy_tabular_data(filename, **kwargs):
    try:
        table = astropy_table_read(filename, **kwargs)
    except:
        return False
    else:
        return True


def astropy_table_read(*args, **kwargs):

    from astropy.table import Table

    # In Python 3, as of Astropy 0.4, if the format is not specified, the
    # automatic format identification will fail (astropy/astropy#3013).
    # This is only a problem for ASCII formats however, because it is due
    # to the fact that the file object in io.ascii does not rewind to the
    # start between guesses (due to a bug), so here we can explicitly try
    # the ASCII format if the format keyword was not already present. But 
    # also more generally, we should first try the ASCII readers.
    if 'format' not in kwargs:
        try:
            return Table.read(*args, format='ascii', **kwargs)
        except:
            pass

    # If the above didn't work, attempt to read with no specified format
    return Table.read(*args, **kwargs)


@data_factory(label="Catalog (Astropy Parser)", identifier=is_astropy_tabular_data)
def astropy_tabular_data(*args, **kwargs):
    """
    Build a data set from a table. We restrict ourselves to tables
    with 1D columns.

    All arguments are passed to
        astropy.table.Table.read(...).
    """

    result = Data()

    table = astropy_table_read(*args, **kwargs)

    # Loop through columns and make component list
    for column_name in table.columns:
        c = table[column_name]
        u = c.unit if hasattr(c, 'unit') else c.units

        if table.masked:
            # fill array for now
            try:
                c = c.filled(fill_value=np.nan)
            except ValueError:  # assigning nan to integer dtype
                c = c.filled(fill_value=-1)

        nc = Component.autotyped(c, units=u)
        result.add_component(nc, column_name)

    return result


@data_factory(label="Catalog",
              identifier=has_extension('xml vot csv txt tsv tbl dat fits '
                                        'xml.gz vot.gz csv.gz txt.gz tbl.bz '
                                        'dat.gz fits.gz'),
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


# Add explicit factories for the formats which astropy.table
# can parse, but does not auto-identify


def formatted_table_factory(format, label):

    @data_factory(label=label, identifier=lambda *a, **k: False)
    def factory(file, **kwargs):
        kwargs['format'] = 'ascii.%s' % format
        return astropy_tabular_data(file, **kwargs)

    # rename function to its variable reference below
    # allows pickling to work
    factory.__name__ = '%s_factory' % format

    return factory

sextractor_factory = formatted_table_factory('sextractor', 'SExtractor Catalog')
cds_factory = formatted_table_factory('cds', 'CDS Catalog')
daophot_factory = formatted_table_factory('daophot', 'DAOphot Catalog')
ipac_factory = formatted_table_factory('ipac', 'IPAC Catalog')
aastex_factory = formatted_table_factory('aastex', 'AASTeX Table')
latex_factory = formatted_table_factory('latex', 'LaTeX Table')
