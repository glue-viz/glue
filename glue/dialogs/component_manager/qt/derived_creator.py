
from __future__ import absolute_import, division, print_function

import os
from collections import deque

from qtpy import QtWidgets, QtCore
from qtpy.QtCore import Qt

from glue.external.echo import CallbackProperty
from glue.core.parse import InvalidTagError, ParsedCommand, TAG_RE
from glue.utils.qt import load_ui, CompletionTextEdit

__all__ = ['EquationEditorDialog']


class DerivedComponentEditor(QtWidgets.QDialog):

    def __init__(self, equation=None, references=None, parent=None):

        super(DerivedComponentEditor, self).__init__(parent=parent)

        self.ui = load_ui('derived_creator.ui', self,
                          directory=os.path.dirname(__file__))

        # Populate component combo
        for label, cid in self.references.items():
            self.ui.combosel_component.addItem(label, userData=cid)

        # Set up labels for auto-completion
        labels = ['{' + label + '}' for label in self.references]
        self.ui.expression.set_word_list(labels)

        # Set initial equation
        self.ui.expression.insertPlainText(equation)

        self.ui.button_insert.clicked.connect(self._insert_component)

        self.ui.expression.updated.connect(self._update_status)
        self._update_status()


        # Get mapping from label to component ID
        if references is not None:
            self.references = references
        elif data is not None:
            self.references = OrderedDict()
            for cid in data.primary_components:
                self.references[cid.label] = cid

if __name__ == "__main__":  # pragma: nocover

    from glue.main import load_plugins
    from glue.utils.qt import get_qapp

    app = get_qapp()
    load_plugins()

    from glue.core.data import Data
    d = Data(label='test1', x=[1, 2, 3], y=[2, 3, 4], z=[3, 4, 5])
    widget = EquationEditorDialog(d, '')
    widget.exec_()
