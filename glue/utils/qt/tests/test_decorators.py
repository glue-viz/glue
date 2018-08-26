from __future__ import absolute_import, division, print_function

import os
from mock import patch

from ..decorators import messagebox_on_error


def test_messagebox_on_error():

    os.environ['GLUE_TESTING'] = 'False'

    @messagebox_on_error('an error occurred')
    def failing_function():
        raise ValueError("Dialog failure")

    with patch('qtpy.QtWidgets.QMessageBox.exec_'):
        failing_function()

    os.environ['GLUE_TESTING'] = 'True'
