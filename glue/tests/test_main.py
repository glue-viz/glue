from unittest.mock import patch

from glue.main import load_plugins, list_loaded_plugins, list_available_plugins
from glue._plugin_helpers import REQUIRED_PLUGINS


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
            plugins.append(acall[0][0].split(' ')[1])

    assert 'coordinate_helpers' in plugins


def test_no_duplicate_loading(capsys):
    """
    Regression test for duplicated loading of plugins
    on subsequent calls of `load_plugins()` after initial
    glue-qt startup.

    """
    from glue.logger import logger

    with patch.object(logger, 'info') as info:
        load_plugins()

    plugins = []
    for acall in info.call_args_list:
        plugins.append(acall[0][0])
        if 'Loading plugin' in acall[0][0]:
            assert 'failed' in acall[0][0]

    loaded_plugins = list_loaded_plugins()
    assert 'glue.plugins.wcs_autolinking' in loaded_plugins
    assert 'glue.core.data_exporters' in loaded_plugins
    assert 'glue.plugins.coordinate_helpers' in loaded_plugins


def test_list_loaded_plugins():
    """
    Unit test for retrieving the list of currently loaded plugins
    """
    load_plugins(require_qt_plugins=False)
    plugins = list_loaded_plugins()
    assert isinstance(plugins, list)
    for test_plugin in REQUIRED_PLUGINS:
        assert test_plugin in plugins


def test_list_available_plugins():
    """
    Unit test for retrieving the list of currently available plugins
    """
    available_plugins = list_available_plugins()
    assert isinstance(available_plugins, list)
    assert 'glue.plugins.wcs_autolinking' in available_plugins
    assert 'glue.core.data_exporters' in available_plugins
    assert 'glue.plugins.coordinate_helpers' in available_plugins
