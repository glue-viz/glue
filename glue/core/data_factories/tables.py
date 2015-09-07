from __future__ import absolute_import, division, print_function

import numpy as np

from ..data import Component, Data
from ...external import six
from ..config import data_factory

from .helpers import has_extension

__all__ = ['tabular_data', 'sextractor_factory', 'astropy_tabular_data',
           'formatted_table_factory']


def _ascii_identifier_v02(origin, args, kwargs):
    # this works for astropy v0.2
    if isinstance(args[0], six.string_types):
        return args[0].endswith(('csv', 'tsv', 'txt', 'tbl', 'dat',
                                 'csv.gz', 'tsv.gz', 'txt.gz', 'tbl.gz',
                                 'dat.gz'))
    else:
        return False


def _ascii_identifier_v03(origin, *args, **kwargs):
    # this works for astropy v0.3
    return _ascii_identifier_v02(origin, args, kwargs)


@data_factory(label="Catalog (Astropy Parser)",
              identifier=has_extension('xml vot csv txt tsv tbl dat fits '
                                       'xml.gz vot.gz csv.gz txt.gz tbl.bz '
                                       'dat.gz fits.gz'))
def astropy_tabular_data(*args, **kwargs):
    """
    Build a data set from a table. We restrict ourselves to tables
    with 1D columns.

    All arguments are passed to
        astropy.table.Table.read(...).
    """
    from distutils.version import LooseVersion
    from astropy import __version__
    if LooseVersion(__version__) < LooseVersion("0.2"):
        raise RuntimeError("Glue requires astropy >= v0.2. Please update")

    result = Data()

    # Read the table
    from astropy.table import Table

    # Add identifiers for ASCII data
    from astropy.io import registry
    if LooseVersion(__version__) < LooseVersion("0.3"):
        registry.register_identifier('ascii', Table, _ascii_identifier_v02,
                                     force=True)
    else:
        # Basically, we always want the plain ascii reader for now.
        # But astropy will complain about ambiguous formats (or use another reader)
        # unless we remove other registry identifiers and set up our own reader

        nope = lambda *a, **k: False
        registry.register_identifier('ascii.glue', Table, _ascii_identifier_v03,
                                     force=True)
        registry.register_identifier('ascii.csv', Table, nope, force=True)
        registry.register_identifier('ascii.fast_csv', Table, nope, force=True)
        registry.register_identifier('ascii', Table, nope, force=True)
        registry.register_reader('ascii.glue', Table,
                                 lambda path: Table.read(path, format='ascii'),
                                 force=True)

    try:
        table = Table.read(*args, **kwargs)
    except:
        # In Python 3, as of Astropy 0.4, if the format is not specified, the
        # automatic format identification will fail (astropy/astropy#3013).
        # This is only a problem for ASCII formats however, because it is due
        # to the fact that the file object in io.ascii does not rewind to the
        # start between guesses (due to a bug), so here we can explicitly try
        # the ASCII format if the format keyword was not already present.
        if 'format' not in kwargs:
            table = Table.read(*args, format='ascii.glue', **kwargs)
        else:
            raise

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
        return tabular_data(file, **kwargs)

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
