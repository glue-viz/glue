from __future__ import absolute_import, division, print_function

from mock import patch

from ..decorators import messagebox_on_error


def test_messagebox_on_error():

    @messagebox_on_error('an error occurred')
    def failing_function():
        raise ValueError("Dialog failure")

    with patch('qtpy.QtWidgets.QMessageBox.exec_'):
        failing_function()
