from unittest.mock import patch

from glue.main import load_plugins
from glue.main import list_plugins


def test_load_plugins(capsys):
    """
    Test customisable list of plugins load
    """
    from glue.logger import logger

    with patch.object(logger, 'info') as info:
        load_plugins()

    # plugins = [call[0][0] for call in info.call_args_list if ('succeeded' or 'loaded') in call[0][0]]
    plugins = []
    for acall in info.call_args_list:
        if ('loaded' or 'succeeded') in acall[0][0]:
            plugins.append(acall[0][0])

    assert len(plugins) == 5


def test_no_duplicate_loading(capsys):
    """
    Regression test for duplicated loading of plugins
    on subsequent calls of `load_plugins()` after initial
    glue-qt startup.

    """
    from glue.logger import logger

    with patch.object(logger, 'info') as info:
        load_plugins()

    for acall in info.call_args_list:
        if 'Loading plugin' in acall[0][0]:
            assert 'failed' in acall[0][0]


def test_list_plugins():
    """
    Regression test for retrieving the list of currently loaded plugins
    """
    load_plugins(require_qt_plugins=False)
    plugins = list_plugins()
    assert isinstance(plugins, list)
    assert len(plugins) == 5
