# pylint: disable=W0223

from PyQt4.QtGui import (QKeySequence, QMainWindow, QGridLayout,
                         QMenu, QMdiSubWindow, QAction, QMessageBox,
                         QFileDialog, QLabel, QPixmap, QDesktopWidget,
                         QToolButton, QSplitter)
from PyQt4.QtCore import Qt

from .. import core
from ..qt import get_qapp
from .ui.glue_application import Ui_GlueApplication
from .decorators import set_cursor, messagebox_on_error

from .actions import act
from .qtutil import pick_class, data_wizard, GlueTabBar
from .widgets.glue_mdi_area import GlueMdiArea
from .widgets.edit_subset_mode_toolbar import EditSubsetModeToolBar
from .widgets.layer_tree_widget import PlotAction


class GlueApplication(QMainWindow, core.hub.HubListener):
    """ The main Glue window """

    def __init__(self, data_collection=None, hub=None):
        super(GlueApplication, self).__init__()
        self.app = get_qapp()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._actions = {}
        self._terminal = None
        self._ui = Ui_GlueApplication()
        self._ui.setupUi(self)
        self._ui.tabWidget.setTabBar(GlueTabBar())
        self.tab_widget.setMovable(True)
        self.tab_widget.setTabsClosable(True)
        self._ui.layerWidget.set_checkable(False)

        lwidget = self._ui.layerWidget
        act = PlotAction(lwidget, self)
        lwidget.layerTree.addAction(act)

        self._data = data_collection or core.data_collection.DataCollection()
        self._hub = hub or core.hub.Hub(self._data)

        self._tweak_geometry()
        self._create_actions()
        self._create_menu()
        self._create_terminal()
        self._connect()
        self._new_tab()
        self._welcome_window()

    def has_terminal(self):
        return self._terminal is not None

    @property
    def tab_widget(self):
        return self._ui.tabWidget

    @property
    def tab_bar(self):
        return self._ui.tabWidget.tabBar()

    @property
    def current_tab(self):
        return self._ui.tabWidget.currentWidget()

    def _tweak_geometry(self):
        """Maximize window"""
        self.setWindowState(Qt.WindowMaximized)

    def _new_tab(self):
        """Spawn a new tab page"""
        layout = QGridLayout()
        layout.setSpacing(1)
        layout.setContentsMargins(0, 0, 0, 0)
        widget = GlueMdiArea(self)
        widget.setLayout(layout)
        tab = self.tab_widget
        tab.addTab(widget, str("Tab %i" % (tab.count() + 1)))
        tab.setCurrentWidget(widget)

    def _close_tab(self, index):
        """ Close a tab window and all associated data viewers """
        #do not delete the last tab
        if self.tab_widget.count() == 1:
            return
        w = self.tab_widget.widget(index)
        w.close()
        self.tab_widget.removeTab(index)

    def _add_to_current_tab(self, new_widget, label=None):
        page = self.current_tab
        sub = QMdiSubWindow()

        sub.setWidget(new_widget)
        sub.resize(new_widget.size())
        if label:
            sub.setWindowTitle(label)
        page.addSubWindow(sub)
        page.setActiveSubWindow(sub)
        return sub

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
        self.tab_widget.tabCloseRequested.connect(self._close_tab)

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
        self._mode_toolbar = tbar
        self.addToolBar(tbar)
        tbar.hide()
        a = QAction("Selection Mode Toolbar", menu)
        a.setCheckable(True)
        a.toggled.connect(tbar.setVisible)
        try:
            tbar.visibilityChanged.connect(a.setChecked)
        except AttributeError:  # Qt < 4.7. Signal not supported
            pass

        menu.addAction(a)
        menu.addActions(tbar.actions())
        mbar.addMenu(menu)

    def _load_data(self):
        for data in data_wizard():
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
        a.triggered.connect(self.tab_bar.rename_tab)
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
        a.triggered.connect(lambda x: self._save_session())
        self._actions['session_save'] = a

        a = act('Open Session', self,
                tip='Restore a saved session')
        a.triggered.connect(lambda x: self._restore_session())
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

    @messagebox_on_error("Failed to save session")
    @set_cursor(Qt.WaitCursor)
    def _save_session(self):
        """ Save the data collection and hub to file.

        Can be restored via restore_session

        Note: Saving of client is not currently supported. Thus,
        restoring this session will lose all current viz windows
        """
        from ..core.glue_pickle import PicklingError, CloudPickler
        state = (self._data, self._hub)

        outfile = QFileDialog.getSaveFileName(self)
        if not outfile:
            return

        with open(outfile, 'w') as out:
            cp = CloudPickler(out, protocol=2)
            cp.dump(state)

    @messagebox_on_error("Failed to restore session")
    @set_cursor(Qt.WaitCursor)
    def _restore_session(self):
        """ Load a previously-saved state, and restart the session """
        from pickle import Unpickler

        fltr = "Glue sessions (*.glu)"
        file_name = QFileDialog.getOpenFileName(self,
                                                filter=fltr)
        if not file_name:
            return

        state = Unpickler(open(file_name)).load()

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

        self._terminal_button = QToolButton(None)
        self._terminal_button.setToolTip("Toggle command line")
        self._ui.layerWidget.button_row.addWidget(self._terminal_button)

        try:
            from .widgets.terminal import glue_terminal
            widget = glue_terminal(data_collection=self._data)
            self._terminal_button.clicked.connect(self._toggle_terminal)
        except Exception as e:  # pylint: disable=W0703
            self._setup_terminal_error_dialog(e)
            return

        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Vertical)
        splitter.addWidget(self._ui.centralwidget)
        splitter.addWidget(widget)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)
        self._terminal = widget

        self._hide_terminal()

    def _setup_terminal_error_dialog(self, exception):
        """ Reassign the terminal toggle button to show dialog on error"""
        title = "Terminal unavailable"
        msg = ("Glue encountered an error trying to start the Terminal"
               "\nException:\n%s\n\nTerminal is unavailable" % exception)

        def show_msg():
            QMessageBox.critical(self, title, msg)

        self._terminal_button.clicked.connect(show_msg)

    def _toggle_terminal(self):
        if self._terminal.isVisible():
            self._hide_terminal()
            assert not self._terminal.isVisible()
        else:
            self._show_terminal()
            assert self._terminal.isVisible()

    def _hide_terminal(self):
        self._terminal.hide()
        button = self._terminal_button
        button.setArrowType(Qt.DownArrow)

    def _show_terminal(self):
        self._terminal.show()
        button = self._terminal_button
        button.setArrowType(Qt.UpArrow)

    def _welcome_window(self):
        widget = QLabel(self)
        pm = QPixmap(':icons/glue_welcome.png')
        pm = pm.scaledToHeight(400, mode=Qt.SmoothTransformation)
        widget.setPixmap(pm)
        widget.show()
        widget.resize(pm.size())
        sub = self._add_to_current_tab(widget, label='Getting Started')

        def do_close(win):
            sub.close()

        self.current_tab.subWindowActivated.connect(do_close)

    def exec_(self):
        self.show()
        self.raise_()  # bring window to front
        return self.app.exec_()

    def keyPressEvent(self, event):
        """Hold down modifier keys to temporarily set edit mode"""
        mod = event.modifiers()
        if mod == Qt.ShiftModifier:
            self._mode_toolbar.set_mode('or')

    def keyReleaseEvent(self, event):
        """Unset any temporary edit mode"""
        self._mode_toolbar.unset_mode()
