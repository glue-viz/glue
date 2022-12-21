import numpy as np

import pandas as pd

from glue.core.data_factories.helpers import has_extension
from glue.core.component import Component, CategoricalComponent
from glue.core.data import Data
from glue.config import data_factory, data_translator

__all__ = ['pandas_read_table']


def panda_process(indf):
    """
    Build a data set from a table using pandas. This attempts to respect
    categorical data input by letting pandas.read_csv infer the type

    """
    result = Data()
    for name, column in indf.items():

        if (column.dtype == object) | (column.dtype == bool):

            # try to salvage numerical data
            try:
                coerced = pd.to_numeric(column, errors='coerce')
            except AttributeError:  # pandas < 0.19
                coerced = column.convert_objects(convert_numeric=True)

            if (coerced.dtype != column.dtype) and coerced.isnull().mean() < 0.4:
                c = Component(coerced.values)
            else:
                # pandas has a 'special' nan implementation and this doesn't
                # play well with np.unique
                c = CategoricalComponent(np.array(column.fillna(''), dtype='U'))
        else:
            c = Component.autotyped(column.values)

        # convert header to string - in some cases if the first row contains
        # numbers, these are cast to numerical types, so we want to change that
        # here.
        if not isinstance(name, str):
            name = str(name)

        # strip off leading #
        name = name.strip()
        if name.startswith('#'):
            name = name[1:].strip()

        result.add_component(c, name)

    return result


@data_factory(label="Pandas Table", identifier=has_extension('csv csv txt tsv tbl dat'))
def pandas_read_table(path, **kwargs):
    """
    A factory for reading tabular data using pandas.

    Parameters
    ----------
    path : str
        Path to the file.

    **kwargs
        All other kwargs are passed to :func:`pandas.read_csv`.

    Returns
    -------
    :class:`glue.core.data.Data` object
    """

    try:
        from pandas.errors import ParserError
    except ImportError:
        try:
            from pandas.io.common import CParserError as ParserError
        except ImportError:  # pragma: no cover
            try:
                from pandas.parser import CParserError as ParserError
            except ImportError:  # pragma: no cover
                from pandas._parser import CParserError as ParserError

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

        except ParserError:
            continue

    if fallback is not None:
        return panda_process(fallback)
    raise IOError("Could not parse %s using pandas" % path)


@data_translator(pd.DataFrame)
class PandasTranslator:

    def to_data(self, obj):
        result = Data()
        for c in obj.columns:
            result.add_component(obj[c], str(c))
        return result

    def to_object(self, data_or_subset, attribute=None):
        df = pd.DataFrame()
        coords = data_or_subset.coordinate_components
        for cid in data_or_subset.components:
            if cid not in coords:
                df[cid.label] = data_or_subset[cid]
        return df
