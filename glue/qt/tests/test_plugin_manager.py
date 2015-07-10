import os
import pytest
from mock import patch
from ...external.qt import QtCore

from ..plugin_manager import QtPluginManager
from ... import _plugin_helpers as ph
from ...utils.qt import QMessageBoxPatched
from ...main import load_plugins


def setup_function(func):
    from ... import config
    func.CFG_DIR_ORIG = config.CFG_DIR


def teardown_function(func):
    from ... import config
    config.CFG_DIR = func.CFG_DIR_ORIG


def test_basic_empty(tmpdir):

    # Test that things work when the plugin cfg file is empty

    from ... import config
    config.CFG_DIR = tmpdir.join('.glue').strpath

    w = QtPluginManager()
    w.clear()
    w.update_list()
    w.finalize()


def test_basic(tmpdir):

    # Test that things work when the plugin cfg file is populated

    from ... import config
    config.CFG_DIR = tmpdir.join('.glue').strpath

    load_plugins()

    config = ph.PluginConfig.load()
    config.plugins['spectrum_tool'] = False
    config.plugins['pv_slicer'] = False
    config.save()

    w = QtPluginManager()
    w.clear()
    w.update_list()
    w.finalize()

    config2 = ph.PluginConfig.load()

    assert config.plugins == config2.plugins


def test_permission_fail(tmpdir):

    from ... import config
    config.CFG_DIR = tmpdir.join('.glue').strpath

    # Make a *file* at that location so that reading the plugin file will fail

    with open(config.CFG_DIR, 'w') as f:
        f.write("test")

    config2 = ph.PluginConfig.load()

    with patch.object(QMessageBoxPatched, 'exec_', return_value=None) as qmb:
        w = QtPluginManager()
        w.finalize()

    assert qmb.call_count == 1
