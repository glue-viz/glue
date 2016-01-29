from __future__ import absolute_import, division, print_function

from subprocess import check_call
import sys

from mock import patch

from glue.tests.helpers import requires_qt

from .._deps import Dependency, categories


class TestDependency(object):

    def test_installed(self):
        d = Dependency('math', 'the math module')
        assert d.installed

    def test_uninstalled(self):
        d = Dependency('asdfasdf', 'Non-existent module')
        assert not d.installed

    def test_noinstall(self):
        with patch('glue._deps.check_call') as check_call:
            d = Dependency('math', 'exists')
            d.install()
            assert check_call.call_count == 0

    def test_install(self):
        with patch('glue._deps.check_call') as check_call:
            d = Dependency('asdfasdf', 'never exists')
            d.install()
            check_call.assert_called_once_with(['pip',
                                                'install',
                                                'asdfasdf'])

    def test_install_with_package_arg(self):
        with patch('glue._deps.check_call') as check_call:
            d = Dependency('asdfasdf', 'never exists', package='bcd')
            d.install()
            check_call.assert_called_once_with(['pip',
                                                'install',
                                                'bcd'])

    def test_installed_str(self):
        d = Dependency('math', 'info')
        assert str(d) == "                math:\tINSTALLED (unknown version)"

    def test_noinstalled_str(self):
        d = Dependency('asdf', 'info')
        assert str(d) == "                asdf:\tMISSING (info)"

    def test_failed_str(self):
        d = Dependency('asdf', 'info')
        d.failed = True
        assert str(d) == "                asdf:\tFAILED (info)"


@requires_qt
def test_optional_dependency_not_imported():
    """
    Ensure that a GlueApplication instance can be created without
    importing any non-required dependency
    """
    optional_deps = categories[2:]
    deps = [dep.module for cateogry, deps in optional_deps for dep in deps]
    deps.extend(['astropy'])

    code = """
class ImportDenier(object):
    __forbidden = set(%s)

    def find_module(self, mod_name, pth):
        if pth:
            return
        if mod_name in self.__forbidden:
            return self

    def load_module(self, mod_name):
        raise ImportError("Importing %%s" %% mod_name)

import sys
sys.meta_path.append(ImportDenier())

from glue.app.qt import GlueApplication
from glue.core import data_factories
ga = GlueApplication()
""" % deps

    cmd = [sys.executable, '-c', code]
    check_call(cmd)
