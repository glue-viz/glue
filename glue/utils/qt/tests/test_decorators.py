from __future__ import absolute_import, division, print_function

import os
from mock import patch

from ..decorators import messagebox_on_error, die_on_error


def test_messagebox_on_error():

    os.environ['GLUE_TESTING'] = 'False'

    def failing_function():
        raise ValueError("Dialog failure")

    @messagebox_on_error('An error occurred')
    def decorated_failing_function():
        failing_function()

    # Test decorator

    with patch('qtpy.QtWidgets.QMessageBox') as mb:
        decorated_failing_function()
        assert mb.call_args[0][2] == 'An error occurred\nDialog failure'

    # Test context manager

    with patch('qtpy.QtWidgets.QMessageBox') as mb:
        with messagebox_on_error('An error occurred'):
            failing_function()
        assert mb.call_args[0][2] == 'An error occurred\nDialog failure'

    os.environ['GLUE_TESTING'] = 'True'


def test_die_on_error():

    os.environ['GLUE_TESTING'] = 'False'

    def failing_function():
        raise ValueError("Dialog failure")

    @die_on_error('An error occurred')
    def decorated_failing_function():
        failing_function()

    # Test decorator

    with patch('sys.exit') as exit:
        with patch('qtpy.QtWidgets.QMessageBox') as mb:
            decorated_failing_function()
            assert mb.call_args[0][2] == 'An error occurred\nDialog failure'
        assert exit.called_once_with(1)

    # Test context manager

    with patch('sys.exit') as exit:
        with patch('qtpy.QtWidgets.QMessageBox') as mb:
            with die_on_error('An error occurred'):
                failing_function()
            assert mb.call_args[0][2] == 'An error occurred\nDialog failure'
        assert exit.called_once_with(1)

    os.environ['GLUE_TESTING'] = 'True'
