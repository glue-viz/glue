from __future__ import absolute_import, division, print_function

import numpy as np

from glue.core.data_factories.helpers import has_extension
from glue.core.data import Component, Data
from glue.config import data_factory, qglue_parser


__all__ = ['astropy_tabular_data', 'sextractor_factory', 'cds_factory',
           'daophot_factory', 'ipac_factory', 'aastex_factory',
           'latex_factory']


# In this file, we define data factories based on the Astropy table reader.


def is_readable_by_astropy(filename, **kwargs):
    # This identifier is not efficient, because it involves actually trying
    # to read in the table. However, we only use this as the identifier for
    # the astropy_tabular_data factory which has a priority of 0 and is
    # therefore only used as a last attempt if all else fails.
    try:
        astropy_table_read(filename, **kwargs)
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


@data_factory(label="Catalog (astropy.table parser)",
              identifier=is_readable_by_astropy,
              priority=0)
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


@data_factory(label="VO table",
              identifier=has_extension('xml vot xml.gz vot.gz'),
              priority=1)
def astropy_tabular_data_votable(*args, **kwargs):
    kwargs['format'] = 'votable'
    return astropy_tabular_data(*args, **kwargs)


@data_factory(label="FITS table",
              identifier=has_extension('fits fits.gz fit fit.gz'),
              priority=1)
def astropy_tabular_data_fits(*args, **kwargs):
    kwargs['format'] = 'fits'
    return astropy_tabular_data(*args, **kwargs)


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

try:
    from astropy.table import Table
except ImportError:
    pass
else:
    @qglue_parser(Table)
    def _parse_data_astropy_table(data, label):
        kwargs = dict((c, data[c]) for c in data.columns)
        return [Data(label=label, **kwargs)]
