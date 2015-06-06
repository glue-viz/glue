from __future__ import absolute_import, division, print_function

import numpy as np

from ..data import Data, Component, CategoricalComponent

from .helpers import has_extension, __factories__


def panda_process(indf):
    """
    Build a data set from a table using pandas. This attempts to respect
    categorical data input by letting pandas.read_csv infer the type

    """
    result = Data()
    for name, column in indf.iteritems():
        if (column.dtype == np.object) | (column.dtype == np.bool):
            # try to salvage numerical data
            coerced = column.convert_objects(convert_numeric=True)
            if (coerced.dtype != column.dtype) and coerced.isnull().mean() < 0.4:
                c = Component(coerced.values)
            else:
                # pandas has a 'special' nan implementation and this doesn't
                # play well with np.unique
                c = CategoricalComponent(column.fillna(''))
        else:
            c = Component(column.values)

        # strip off leading #
        name = name.strip()
        if name.startswith('#'):
            name = name[1:].strip()

        result.add_component(c, name)

    return result


def pandas_read_table(path, **kwargs):
    """ A factory for reading tabular data using pandas
    :param path: path/to/file
    :param kwargs: All kwargs are passed to pandas.read_csv
    :returns: :class:`glue.core.data.Data` object
    """
    import pandas as pd
    try:
        from pandas.parser import CParserError
    except ImportError:
        from pandas._parser import CParserError

    # iterate over common delimiters to search for best option
    delimiters = kwargs.pop('delimiter', [None] + list(',|\t '))

    fallback = None

    for d in delimiters:
        try:
            indf = pd.read_csv(path, delimiter=d, **kwargs)

            # ignore files parsed to empty dataframes
            if len(indf) == 0:
                continue

            # only use files parsed to single-column dataframes
            # if we don't find a better strategy
            if len(indf.columns) < 2:
                fallback = indf
                continue

            return panda_process(indf)

        except CParserError:
            continue

    if fallback is not None:
        return panda_process(fallback)
    raise IOError("Could not parse %s using pandas" % path)

pandas_read_table.label = "Pandas Table"
pandas_read_table.identifier = has_extension('csv csv txt tsv tbl dat')
__factories__.append(pandas_read_table)
