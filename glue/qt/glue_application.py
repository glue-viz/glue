# pylint: disable=W0223

from PyQt4.QtGui import (QKeySequence, QMainWindow, QGridLayout, QMdiArea,
                         QMenu, QMdiSubWindow)
from PyQt4.QtCore import Qt

from .. import core

from .ui.glue_application import Ui_GlueApplication

from .actions import act
from .qtutil import pick_class, get_text, data_wizard


class GlueApplication(QMainWindow, core.hub.HubListener):
    """ The main Glue window """

    def __init__(self):
        super(GlueApplication, self).__init__()
        self._actions = {}
        self._terminal = None
        self._ui = Ui_GlueApplication()
        self._ui.setupUi(self)
        self._ui.layerWidget.set_checkable(False)
        self._data = core.data_collection.DataCollection()
        self._hub = core.hub.Hub(self._data)
        self._new_tab()

        self._create_actions()
        self._connect()
        self._create_menu()
        self._create_terminal()

    @property
    def tab_bar(self):
        return self._ui.tabWidget

    @property
    def current_tab(self):
        return self._ui.tabWidget.currentWidget()

    def _new_tab(self):
        """Spawn a new tab page"""
        layout = QGridLayout()
        layout.setSpacing(1)
        layout.setContentsMargins(0, 0, 0, 0)
        widget = QMdiArea()
        widget.setLayout(layout)
        tab = self.tab_bar
        tab.addTab(widget, str(tab.count()+1))
        tab.setCurrentWidget(widget)

    def _add_to_current_tab(self, new_widget):
        page = self.current_tab
        sub = QMdiSubWindow()

        sub.setWidget(new_widget)
        sub.resize(new_widget.size())
        page.addSubWindow(sub)

    def _rename_current_tab(self):
        """ Prompt the user to rename the current tab """
        label = get_text("New Tab Label")
        if not label:
            return
        index = self.tab_bar.currentIndex()
        self.tab_bar.setTabText(index, label)

    def cascade_current_tab(self):
        """Arrange windows in current tab via cascade"""
        self.current_tab.cascadeSubWindows()

    def tile_current_tab(self):
        """Arrange windows in current tab via tiling"""
        self.current_tab.tileSubWindows()

    def _connect(self):
        self._hub.subscribe(self,
                            core.message.ErrorMessage,
                            handler=self._report_error)
        self._ui.layerWidget.data_collection = self._data
        self._ui.layerWidget.register_to_hub(self._hub)
        self._data.register_to_hub(self._hub)
        self._ui.terminal_button.clicked.connect(self._toggle_terminal)

    def _create_menu(self):
        mbar = self._ui.menubar
        menu = QMenu(mbar)
        menu.setTitle("File")

        menu.addAction(self._actions['data_new'])
        #menu.addAction(self._actions['data_save'])  # XXX add this
        menu.addAction(self._actions['session_restore'])
        menu.addAction(self._actions['session_save'])

        mbar.addMenu(menu)

        menu = QMenu(mbar)
        menu.setTitle("Canvas")
        menu.addAction(self._actions['tab_new'])
        menu.addAction(self._actions['viewer_new'])
        menu.addSeparator()
        menu.addAction(self._actions['cascade'])
        menu.addAction(self._actions['tile'])
        menu.addAction(self._actions['tab_rename'])
        mbar.addMenu(menu)

        menu = QMenu(mbar)
        menu.setTitle("Data Manager")
        menu.addActions(self._ui.layerWidget.actions())

        mbar.addMenu(menu)

    def _load_data(self):
        data = data_wizard()
        if data:
            self._data.append(data)

    def _create_actions(self):
        """ Create and connect actions, store in _actions dict """
        self._actions = {}

        a = act("Open Data Set", self,
                tip="Open a new data set",
                shortcut=QKeySequence.Open)
        a.triggered.connect(self._load_data)
        self._actions['data_new'] = a

        a = act("New Data Viewer", self,
                tip="Open a new visualization window in the current tab",
                shortcut=QKeySequence.New
                )
        a.triggered.connect(self._new_viz_window)
        self._actions['viewer_new'] = a


        a = act('New Tab', self,
                shortcut=QKeySequence.AddTab,
                tip='Add a new tab')
        a.triggered.connect(self._new_tab)
        self._actions['tab_new'] = a

        a = act('Rename Tab', self,
                shortcut="Ctrl+R",
                tip='Set a new label for the current tab')
        a.triggered.connect(self._rename_current_tab)
        self._actions['tab_rename'] = a

        a = act('Cascade', self,
                tip='Cascade the windows in the current tab')
        a.triggered.connect(self.cascade_current_tab)
        self._actions['cascade'] = a

        a = act('Tile', self,
                tip='Tile the windows in the current tab')
        a.triggered.connect(self.tile_current_tab)
        self._actions['tile'] = a

        a = act('Save Session', self,
                tip='Save the current session')
        a.setEnabled(False)
        a.triggered.connect(self._save_session)
        self._actions['session_save'] = a


        a = act('Open Session', self,
                tip='Restore a saved session')
        a.setEnabled(False)
        a.triggered.connect(self._restore_session)
        self._actions['session_restore'] = a

    def _new_viz_window(self):
        """ Create a new visualization window in the current tab
        """

        from .. import env
        client = pick_class(env.qt_clients)
        if client:
            c = client(self._data)
            c.register_to_hub(self._hub)
            self._add_to_current_tab(c)
            c.show()

    def _report_error(self, message):
        self.statusBar().showMessage(str(message))

    def _save_session(self):
        raise NotImplemented

    def _restore_session(self):
        raise NotImplemented

    def _create_terminal(self):
        assert self._terminal is None, "should only call _create_terminal once"
        from widgets.terminal import glue_terminal
        widget = glue_terminal(data_collection=self._data)
        layout = self._ui.centralwidget.layout()
        layout.addWidget(widget)
        self._terminal = widget
        self._hide_terminal()

    def _toggle_terminal(self):
        if self._terminal.isVisible():
            self._hide_terminal()
            assert not self._terminal.isVisible()
        else:
            self._show_terminal()
            assert self._terminal.isVisible()

    def _hide_terminal(self):
        self._terminal.hide()
        button = self._ui.terminal_button
        button.setArrowType(Qt.DownArrow)

    def _show_terminal(self):
        self._terminal.show()
        button = self._ui.terminal_button
        button.setArrowType(Qt.UpArrow)