from __future__ import absolute_import, division, print_function

import os
import pytest
from mock import patch

from glue.tests.helpers import requires_qt

from ..core import Data
from ..main import (die_on_error, restore_session, load_data_files,
                    main, start_glue)


@requires_qt
def test_die_on_error_exception():
    """Decorator should spawn a QMessageBox and exit"""
    with pytest.raises(SystemExit):
        with patch('glue.utils.qt.QMessageBoxPatched') as qmb:
            @die_on_error('test_msg')
            def test():
                raise Exception()
            test()
            assert qmb.call_count == 1


def test_die_on_error_noexception():
    """Decorator should have no effect"""
    @die_on_error('test_msg')
    def test():
        return 0
    assert test() == 0


def test_load_data_files():
    with patch('glue.core.data_factories.load_data') as ld:
        ld.return_value = Data()
        dc = load_data_files(['test.py'])
        assert len(dc) == 1


def check_main(cmd, glue, config, data):
    """Pass command to main program, check for expected parsing"""
    with patch('glue.main.start_glue') as sg:
        main(cmd.split())
        args, kwargs = sg.call_args
        assert kwargs.get('datafiles', None) == data
        assert kwargs.get('gluefile', None) == glue
        assert kwargs.get('config', None) == config


def check_exec(cmd, pyfile):
    """Assert that main correctly dispatches to execute_script"""
    with patch('glue.main.execute_script') as es:
        main(cmd.split())
        args, kwargs = es.call_args
        assert args[0] == pyfile


def test_main_single_data():
    check_main('glueqt test.fits', None, None, ['test.fits'])


def test_main_multi_data():
    check_main('glueqt test.fits t2.csv', None, None, ['test.fits', 't2.csv'])


def test_main_config():
    check_main('glueqt -c config.py', None, 'config.py', None)


def test_main_glu_arg():
    check_main('glueqt -g test.glu', 'test.glu', None, None)


def test_main_auto_glu():
    check_main('glueqt test.glu', 'test.glu', None, None)


def test_main_many_args():
    check_main('glueqt -c config.py data.fits d2.csv', None,
               'config.py', ['data.fits', 'd2.csv'])


def test_exec():
    check_exec('glueqt -x test.py', 'test.py')


def test_auto_exec():
    check_exec('glueqt test.py', 'test.py')


@requires_qt
def test_exec_real(tmpdir):
    # Actually test the script execution functionlity
    filename = tmpdir.join('test.py').strpath
    with open(filename, 'w') as f:
        f.write('a = 1')
    with patch('glue.utils.qt.QMessageBoxPatched') as qmb:
        with patch('sys.exit') as exit:
            main('glue -x {0}'.format(os.path.abspath(filename)).split())
    assert exit.called_once_with(0)


@pytest.mark.parametrize(('cmd'), ['glueqt -g test.glu test.fits',
                                   'glueqt -g test.py test.fits',
                                   'glueqt -x test.py -g test.glu',
                                   'glueqt -x test.py -c test.py',
                                   'glueqt -x',
                                   'glueqt -g',
                                   'glueqt -c'])
def test_invalid(cmd):
    with pytest.raises(SystemExit):
        main(cmd.split())


@requires_qt
@pytest.mark.parametrize(('glue', 'config', 'data'),
                         [('test.glu', None, None),
                          (None, 'test.py', None),
                          (None, None, ['test.fits']),
                          (None, None, ['a.fits', 'b.fits']),
                          (None, 'test.py', ['a.fits'])])
def test_start(glue, config, data):
    with patch('glue.main.restore_session') as rs:
        with patch('glue.config.load_configuration') as lc:
            with patch('glue.main.load_data_files') as ldf:
                with patch('glue.app.qt.GlueApplication') as ga:
                    with patch('qtpy.QtWidgets') as qt:

                        rs.return_value = ga
                        ldf.return_value = Data()

                        start_glue(glue, config, data)
                        if glue:
                            rs.assert_called_once_with(glue)
                        if config:
                            lc.assert_called_once_with(search_path=[config])
                        if data:
                            ldf.assert_called_once_with(data)
