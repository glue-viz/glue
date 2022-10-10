import os

from qtpy import QtWidgets

from glue.config import settings
from glue.utils.qt import load_ui
from glue.core.autolinking import find_possible_links
from glue.dialogs.link_editor.qt.link_editor import LinkEditorWidget

__all__ = ['run_autolinker']

DESCRIPTION = "The auto-linking plugin '{0}' has identified {1} links between your datasets - click on 'More Details' below to find out more about the suggested links."


class AutoLinkPreview(QtWidgets.QDialog):

    def __init__(self, autolinker_name, data_collection, suggested_links, parent=None):

        super(AutoLinkPreview, self).__init__(parent=parent)

        self._data_collection = data_collection

        self._ui = load_ui('autolinker.ui', self,
                           directory=os.path.dirname(__file__))

        self._autolinker_name = autolinker_name

        self.link_widget = LinkEditorWidget(data_collection,
                                            suggested_links=suggested_links,
                                            parent=self)

        self._ui.layout().insertWidget(2, self.link_widget)

        self._ui.label.setText(DESCRIPTION.format(autolinker_name, len(suggested_links)))

        self._ui.button_apply.clicked.connect(self.accept)
        self._ui.button_ignore.clicked.connect(self.reject)

        self._ui.button_details.clicked.connect(self._toggle_details)

        self._set_details_visibility(False)

    def _toggle_details(self, *args):
        self._set_details_visibility(not self._details_visible)

    def _set_details_visibility(self, visible):

        self._details_visible = visible

        self.link_widget.setVisible(visible)
        self.label_viz.setVisible(visible)

        if visible:
            self._ui.button_details.setText('Hide Details')
            self.setFixedHeight(800)
        else:
            self._ui.button_details.setText('Show Details')
            self.setFixedHeight(100)

        # Make sure the dialog is centered on the screen
        try:
            screen = QtWidgets.QApplication.desktop().screenGeometry(0)
            self.move(screen.center() - self.rect().center())
        except AttributeError:  # PySide6
            self.move(QtWidgets.QApplication.primaryScreen().geometry().center())

    def accept(self):
        # Check what we need to do here to apply links
        if self._ui.checkbox_apply_future.isChecked():
            settings.AUTOLINK[self._autolinker_name] = 'always_accept'
        self.link_widget.state.update_links_in_collection()
        super(AutoLinkPreview, self).accept()

    def reject(self):
        if self._ui.checkbox_apply_future.isChecked():
            settings.AUTOLINK[self._autolinker_name] = 'always_ignore'
        super(AutoLinkPreview, self).reject()

    @classmethod
    def suggest_links(cls, autolinker_name, data_collection, links, parent=None):
        mode = settings.AUTOLINK.get(autolinker_name, 'always_show')
        if mode == 'always_show':
            widget = cls(autolinker_name, data_collection, links)
            widget._ui.exec_()
        elif mode == 'always_accept':
            data_collection.add_link(links)
        else:
            pass


def run_autolinker(data_collection):
    suggestions = find_possible_links(data_collection)
    for autolinker_name, links in suggestions.items():
        AutoLinkPreview.suggest_links(autolinker_name, data_collection, links)
