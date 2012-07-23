# pylint: disable=W0223

from PyQt4.QtGui import (QKeySequence, QMainWindow, QGridLayout,
                         QMenu, QMdiSubWindow, QAction, QMessageBox,
                         QFileDialog)
from PyQt4.QtCore import Qt

from .. import core

from .ui.glue_application import Ui_GlueApplication

from .actions import act
from .qtutil import pick_class, get_text, data_wizard
from .widgets.glue_mdi_area import GlueMdiArea
from .widgets.edit_subset_mode_toolbar import EditSubsetModeToolBar

class GlueApplication(QMainWindow, core.hub.HubListener):
    """ The main Glue window """

    def __init__(self, data_collection=None, hub=None):
        super(GlueApplication, self).__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._actions = {}
        self._terminal = None
        self._ui = Ui_GlueApplication()
        self._ui.setupUi(self)
        self._ui.layerWidget.set_checkable(False)
        self._data = data_collection or core.data_collection.DataCollection()
        self._hub = hub or core.hub.Hub(self._data)

        self._create_actions()
        self._connect()
        self._create_menu()
        self._create_terminal()
        self._new_tab()

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
        widget = GlueMdiArea(self)
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
        self._ui.layerWidget.setup(self._data, self._hub)
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

        menu = QMenu(mbar)
        menu.setTitle("Toolbars")
        tbar = EditSubsetModeToolBar()
        self.addToolBar(tbar)
        tbar.hide()
        act = QAction("Selection Modes", menu)
        act.setCheckable(True)
        act.toggled.connect(tbar.setVisible)
        tbar.visibilityChanged.connect(act.setChecked)
        menu.addAction(act)
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
        a.triggered.connect(self.new_data_viewer)
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
        a.triggered.connect(self._save_session)
        self._actions['session_save'] = a


        a = act('Open Session', self,
                tip='Restore a saved session')
        a.triggered.connect(self._restore_session)
        self._actions['session_restore'] = a

    def new_data_viewer(self, data=None):
        """ Create a new visualization window in the current tab
        """

        from .. import env
        client = pick_class(env.qt_clients, title='Data Viewer',
                            label="Choose a new data viewer")
        if client:
            c = client(self._data)
            c.register_to_hub(self._hub)
            self._add_to_current_tab(c)
            if data:
                c.add_data(data)
            c.show()

    def _report_error(self, message):
        self.statusBar().showMessage(str(message))

    def _save_session(self):
        """ Save the data collection and hub to file.

        Can be restored via restore_session

        Note: Saving of client is not currently supported. Thus,
        restoring this session will lose all current viz windows
        """
        from ..core.glue_pickle import dumps, PicklingError
        state = (self._data, self._hub)

        try:
            data = dumps(state)
        except PicklingError as p:
            QMessageBox.critical(self, "Error",
                                 "Cannot save data object: %s" % p)
            return

        outfile = QFileDialog.getSaveFileName(self)
        if not outfile:
            return
        try:
            with open(outfile, 'w') as out:
                out.write(data)
        except IOError as e:
            QMessageBox.critical(self, "Error",
                                 "Could not write file:\n%s" % e)

    def _restore_session(self):
        """ Load a previously-saved state, and restart the session """
        from pickle import Unpickler, UnpicklingError

        file_name = QFileDialog.getOpenFileName(self)
        if not file_name:
            return
        try:
            state = Unpickler(open(file_name)).load()
        except (IOError, UnpicklingError) as e:
            QMessageBox.critical(self, "Error",
                                 "Could not restore file: %s" % e)
            return
        data, hub = state
        pos = self.pos()
        size = self.size()
        ga = GlueApplication(data_collection=data, hub=hub)
        ga.move(pos)
        ga.resize(size)

        ga.show()
        self.close()

    def _create_terminal(self):
        assert self._terminal is None, \
            "should only call _create_terminal once"

        try:
            from widgets.terminal import glue_terminal
            widget = glue_terminal(data_collection=self._data)
        except Exception as e:
            self._setup_terminal_error_dialog(e)
            return
        layout = self._ui.centralwidget.layout()
        layout.addWidget(widget)
        self._terminal = widget
        self._hide_terminal()

    def _setup_terminal_error_dialog(self, exception):
        """ Reassign the terminal toggle button to show dialog on error"""
        self._ui.terminal_button.clicked.disconnect()
        title = "Terminal unavailable"
        msg = ("Glue encountered an error trying to start the Terminal"
               "\nException:\n%s\n\nTerminal is unavailable" % exception)
        def show_msg():
            QMessageBox.critical(self, title, msg)
        self._ui.terminal_button.clicked.connect(show_msg)

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
