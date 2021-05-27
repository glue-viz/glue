import os

from glue.core.data_factories.helpers import has_extension
from glue.core.data_factories.pandas import panda_process
from glue.config import data_factory


__all__ = []


@data_factory(label="Excel", identifier=has_extension('xls'))
def panda_read_xls(path, sheet=None, **kwargs):
    """
    A factory for reading excel xls data using pandas and xlrd.

    Parameters
    ----------
    path : str
        Path to the file
    sheet : str, optional
        The sheet to read. If `None`, all sheets are read.
    kwargs
        All other kwargs are passed to pandas.read_excel
    """

    try:
        import pandas as pd
    except ImportError:
        raise ImportError('Pandas is required for Excel input.')

    try:
        import xlrd
    except ImportError:
        raise ImportError('xlrd is required for xls input.')

    name = os.path.basename(path)
    if '.xls' in name:
        name = name.rsplit('.xls', 1)[0]

    xl_workbook = xlrd.open_workbook(path)

    if sheet is None:
        sheet_names = xl_workbook.sheet_names()
    else:
        sheet_names = [sheet]

    all_data = []
    for sheet in sheet_names:
        indf = pd.read_excel(path, sheet, **kwargs)
        data = panda_process(indf)
        data.label = "{0}:{1}".format(name, sheet)
        all_data.append(data)

    return all_data


@data_factory(label="Excel", identifier=has_extension('xlsx'))
def panda_read_xlsx(path, sheet=None, **kwargs):
    """
    A factory for reading excel xlsx data using pandas and openpyxl.

    Parameters
    ----------
    path : str
        Path to the file
    sheet : str, optional
        The sheet to read. If `None`, all sheets are read.
    kwargs
        All other kwargs are passed to pandas.read_excel

    """

    try:
        import pandas as pd
    except ImportError:
        raise ImportError('Pandas is required for Excel input.')

    try:
        from openpyxl import load_workbook
    except ImportError:
        raise ImportError('openpyxl is required for xlsx input.')

    name = os.path.basename(path)
    if '.xlsx' in name:
        name = name.rsplit('.xlsx', 1)[0]

    xl_workbook = load_workbook(filename=path)

    if sheet is None:
        sheet_names = xl_workbook.sheetnames
    else:
        sheet_names = [sheet]

    all_data = []
    for sheet in sheet_names:
        indf = pd.read_excel(path, sheet, **kwargs)
        data = panda_process(indf)
        data.label = "{0}:{1}".format(name, sheet)
        all_data.append(data)

    return all_data
