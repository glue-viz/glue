from unittest.mock import patch, MagicMock
from collections import namedtuple
from glue.core import Data

from ..dialog import export_data


def test_export(tmpdir):

    filename = tmpdir.join('data')

    data = Data(x=[1, 2, 3])

    mock = MagicMock()

    test_exporter_cls = namedtuple('exporter', 'function label extension')
    test_exporter = test_exporter_cls(function=mock, label='Test', extension='')

    with patch('qtpy.compat.getsavefilename') as dialog:
        with patch('glue.config.data_exporter') as data_exporter:
            def test_iter(x):
                yield test_exporter
            data_exporter.__iter__ = test_iter
            dialog.return_value = filename, 'Test (*)'
            export_data(data)

    assert test_exporter.function.call_args[0] == (filename, data)
