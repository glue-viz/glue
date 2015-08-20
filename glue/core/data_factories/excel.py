from __future__ import absolute_import, division, print_function

from ...config import data_factory

from .pandas import panda_process
from .helpers import has_extension

__all__ = []


@data_factory(label="Excel", identifier=has_extension('xls xlsx'))
def panda_read_excel(path, sheet=None, **kwargs):
    """ A factory for reading excel data using pandas.
    :param path: path/to/file
    :param sheet: The sheet to read. If `None`, all sheets are read.
    :param kwargs: All other kwargs are passed to pandas.read_excel
    :return: core.data.Data object.
    """

    try:
        import pandas as pd
    except ImportError:
        raise ImportError('Pandas is required for Excel input.')

    try:
        import xlrd
    except ImportError:
        raise ImportError('xlrd is required for Excel input.')

    xl_workbook = xlrd.open_workbook(path)

    if sheet is None:
        sheet_names = xl_workbook.sheet_names()
    else:
        sheet_names = [sheet]

    data = []
    for sheet in sheet_names:
        indf = pd.read_excel(path, sheet, **kwargs)
        data.append(panda_process(indf))

    return data
