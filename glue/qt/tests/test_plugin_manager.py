import os
import pytest
from mock import patch
from ...external.qt import QtCore

from ..plugin_manager import QtPluginManager
from ... import _plugin_helpers as ph
from ...utils.qt import QMessageBoxPatched
from ...main import load_plugins


def setup_function(func):
    func.cfg_dir_orig = ph.CFG_DIR


def teardown_function(func):
    ph.CFG_DIR = func.cfg_dir_orig


def test_basic_empty(tmpdir):

    # Test that things work when the plugin cfg file is empty

    ph.CFG_DIR = tmpdir.join('.glue').strpath

    w = QtPluginManager()
    w.clear()
    w.update_list()
    w.finalize()


def test_basic(tmpdir):

    # Test that things work when the plugin cfg file is populated

    ph.CFG_DIR = tmpdir.join('.glue').strpath

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

    print(config)
    print(config2)

    assert config.plugins == config2.plugins


def test_permission_fail(tmpdir):

    ph.CFG_DIR = tmpdir.join('.glue').strpath
    os.mkdir(ph.CFG_DIR)
    os.chmod(ph.CFG_DIR, 0o000)

    with patch.object(QMessageBoxPatched, 'exec_', return_value=None) as qmb:
        w = QtPluginManager()
        w.finalize()

    assert qmb.call_count == 1
