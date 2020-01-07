import pytest

from qtpy import QtTest
from qtpy.QtCore import Qt
from glue.core import Data

from ..equation_editor import EquationEditorDialog


class TestEquationEditor:

    def setup_method(self, method):
        self.data = Data(x=[1, 2, 3], y=[3, 4, 5])
        self.dialog = EquationEditorDialog(label='z', data=self.data, equation='')

    def test_empty(self):
        assert not self.dialog.ui.button_ok.isEnabled()
        assert self.dialog.ui.label_status.text() == ''

    @pytest.mark.parametrize('expression', ['1', '1 + {x}', '1 * np.sin({y}) + {x}'])
    def test_valid_cases(self, expression):
        self.dialog.expression.insertPlainText(expression)
        assert self.dialog.ui.button_ok.isEnabled()
        assert self.dialog.ui.label_status.text() == 'Valid expression'
        self.dialog.ui.button_ok.click()
        assert self.dialog._get_raw_command() == expression

    def test_invalid_syntax(self):
        self.dialog.expression.insertPlainText('1 + {x')
        assert not self.dialog.ui.button_ok.isEnabled()
        assert self.dialog.ui.label_status.text() == 'Incomplete or invalid syntax'

    def test_unknown_component(self):
        self.dialog.expression.insertPlainText('1 + {z}')
        assert not self.dialog.ui.button_ok.isEnabled()
        assert self.dialog.ui.label_status.text() == 'Invalid component: z'

    def test_undefined_name(self):
        self.dialog.expression.insertPlainText('1 + {x} + abc')
        assert not self.dialog.ui.button_ok.isEnabled()
        assert self.dialog.ui.label_status.text() == "name 'abc' is not defined"

    def test_insert_component(self):
        self.dialog.expression.insertPlainText('1 + ')
        self.dialog.button_insert.click()
        assert self.dialog.ui.label_status.text() == 'Valid expression'
        self.dialog.ui.button_ok.click()
        assert self.dialog._get_raw_command() == '1 + {Pixel Axis 0 [x]}'

    def test_nolabel(self):
        self.dialog.ui.text_label.setText('')
        self.dialog.expression.insertPlainText('1 + {x}')
        assert not self.dialog.ui.button_ok.isEnabled()
        assert self.dialog.ui.label_status.text() == 'Attribute name not set'

    def test_typing(self):

        # This ensures that the code that highlights syntax gets called,
        # and also ensures we can test undoing.

        chars = (Qt.Key_1, Qt.Key_Space, Qt.Key_Plus, Qt.Key_Space,
                 Qt.Key_BraceLeft, Qt.Key_X, Qt.Key_BraceRight)

        for char in chars:
            QtTest.QTest.keyClick(self.dialog.expression, char)

        assert self.dialog.expression.toPlainText() == '1 + {x}'

        QtTest.QTest.keyClick(self.dialog.expression, Qt.Key_Z, Qt.ControlModifier)

        assert self.dialog.expression.toPlainText() == '1 + {x'

        for i in range(4):
            QtTest.QTest.keyClick(self.dialog.expression, Qt.Key_Z, Qt.ControlModifier)

        assert self.dialog.expression.toPlainText() == '1 '

    def test_cancel(self):
        self.dialog.expression.insertPlainText('1 + {x}')
        assert self.dialog.ui.label_status.text() == 'Valid expression'
        self.dialog.ui.button_cancel.click()
        assert self.dialog.final_expression is None
