from __future__ import absolute_import, division, print_function

import os

from glue.core import Subset
from glue.config import data_exporter

__all__ = []


def data_to_astropy_table(data, components=None):

    if isinstance(data, Subset):
        mask = data.to_mask()
        data = data.data
    else:
        mask = None

    from astropy.table import Table

    table = Table()
    for cid in data.visible_components:

        if components is not None and cid not in components:
            continue

        comp = data.get_component(cid)
        if comp.categorical:
            values = comp.labels
        else:
            values = comp.data

        if mask is not None:
            values = values[mask]

        table[cid.label] = values

    return table


def table_exporter(fmt, label, extension):

    @data_exporter(label=label, extension=extension)
    def factory(filename, data, components=None):
        if os.path.exists(filename):
            os.remove(filename)
        return data_to_astropy_table(data, components=components).write(filename, format=fmt)

    # rename function to its variable reference below
    # allows pickling to work
    factory.__name__ = '%s_factory' % fmt.replace('.', '_')

    return factory


csv_exporter = table_exporter('ascii.csv', 'Comma-separated table', ['csv'])
ipac_exporter = table_exporter('ascii.ipac', 'IPAC Catalog', ['tbl'])
latex_exporter = table_exporter('ascii.latex', 'LaTeX Table', ['tex'])
votable_exporter = table_exporter('votable', 'VO Table', ['xml', 'vot'])
fits_exporter = table_exporter('fits', 'FITS Table', ['fits', 'fit'])
