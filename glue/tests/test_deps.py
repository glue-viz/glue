from mock import MagicMock, patch

from .._deps import Dependency
from .. import _deps as dep

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
        assert str(d) == "                math:\tINSTALLED"

    def test_noinstalled_str(self):
        d = Dependency('asdf', 'info')
        assert str(d) == "                asdf:\tMISSING (info)"

    def test_failed_str(self):
        d = Dependency('asdf', 'info')
        d.failed = True
        assert str(d) == "                asdf:\tFAILED (info)"
