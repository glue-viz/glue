# pylint: disable=W0223
import sys

from PyQt4.QtGui import (QKeySequence, QMainWindow, QGridLayout,
                         QMenu, QMdiSubWindow, QAction, QMessageBox,
                         QFileDialog,
                         QToolButton, QSplitter, QVBoxLayout, QWidget)
from PyQt4.QtCore import Qt

from .. import core
from .. import env
from ..qt import get_qapp
from .ui.glue_application import Ui_GlueApplication
from .decorators import set_cursor, messagebox_on_error
from ..core.data_factories import load_data

from .actions import act
from .qtutil import pick_class, data_wizard, GlueTabBar
from .widgets.glue_mdi_area import GlueMdiArea
from .widgets.edit_subset_mode_toolbar import EditSubsetModeToolBar
from .widgets.layer_tree_widget import PlotAction, LayerTreeWidget
from .widgets.data_viewer import DataViewer


class GlueApplication(QMainWindow, core.hub.HubListener):
    """ The main Glue window """

    def __init__(self, data_collection=None, hub=None):
        super(GlueApplication, self).__init__()
        self.app = get_qapp()
        self.setWindowIcon(self.app.windowIcon())
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._actions = {}
        self._terminal = None
        self._ui = Ui_GlueApplication()
        self._setup_ui()
        self.tab_widget.setMovable(True)
        self.tab_widget.setTabsClosable(True)

        lwidget = self._ui.layerWidget
        act = PlotAction(lwidget, self)
        lwidget.layerTree.addAction(act)
        lwidget.bind_selection_to_edit_subset()

        self._data = data_collection or core.data_collection.DataCollection()
        self._hub = hub or core.hub.Hub(self._data)

        self._tweak_geometry()
        self._create_actions()
        self._create_menu()
        self._create_terminal()
        self._connect()
        self._new_tab()
        self._update_plot_dashboard(None)

    def _setup_ui(self):
        self._ui.setupUi(self)
        self._ui.tabWidget.setTabBar(GlueTabBar())

        lw = LayerTreeWidget()
        lw.set_checkable(False)
        vb = QVBoxLayout()
        vb.setContentsMargins(0, 0, 0, 0)
        vb.addWidget(lw)
        self._ui.data_layers.setLayout(vb)
        self._ui.layerWidget = lw

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
        self._ui.main_splitter.setSizes([100, 800])
        self._ui.data_plot_splitter.setSizes([100, 200])

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
        widget.subWindowActivated.connect(self._update_plot_dashboard)

    def _get_plot_dashboards(self, sub_window):
        if not isinstance(sub_window, QMdiSubWindow):
            return QWidget(), QWidget(), ""

        widget = sub_window.widget()
        if not isinstance(widget, DataViewer):
            return QWidget(), QWidget(), ""

        return widget.layer_view(), widget.options_widget(), str(widget)

    def _update_plot_dashboard(self, sub_window):
        layer_view, options_widget, title = \
            self._get_plot_dashboards(sub_window)

        layout = self._ui.plot_layers.layout()
        if not layout:
            layout = QVBoxLayout()
            self._ui.plot_layers.setLayout(layout)
        while layout.count():
            layout.takeAt(0).widget().hide()
        layout.addWidget(layer_view)

        layout = self._ui.plot_options.layout()
        if not layout:
            layout = QVBoxLayout()
            self._ui.plot_options.setLayout(layout)
        while layout.count():
            layout.takeAt(0).widget().hide()
        layout.addWidget(options_widget)

        layer_view.show()
        options_widget.show()

        if title:
            self._ui.plot_options.setTitle("Plot Options - %s" % title)
            self._ui.plot_layers.setTitle("Plot Layers - %s" % title)
        else:
            self._ui.plot_options.setTitle("Plot Options")
            self._ui.plot_layers.setTitle("Plot Layers")

        self._update_focus_decoration()

    def _update_focus_decoration(self):
        mdi_area = self.current_tab
        active = mdi_area.activeSubWindow()

        for win in mdi_area.subWindowList():
            widget = win.widget()
            if isinstance(widget, DataViewer):
                widget.set_focus(win is active)

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

    def gather_current_tab(self):
        """Arrange windows in current tab via tiling"""
        self.current_tab.tileSubWindows()

    def _connect(self):
        self.setAcceptDrops(True)
        self._hub.subscribe(self,
                            core.message.ErrorMessage,
                            handler=self._report_error)
        self._ui.layerWidget.setup(self._data, self._hub)

        def sethelp(*args):
            model = self._ui.layerWidget.layerTree.model()
            self.current_tab.show_help = model.rowCount() > 0

        model = self._ui.layerWidget.layerTree.model()
        model.rowsInserted.connect(sethelp)
        model.rowsRemoved.connect(sethelp)

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
        menu.addAction(self._actions['gather'])
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

        if sys.platform == 'darwin':
            mbar.addMenu('Help')

    def _load_data_interactive(self):
        for d in data_wizard():
            self._data.append(d)

    @messagebox_on_error("Could not load data")
    def _load_data(self, path):
        d = load_data(path)
        self._data.append(d)

    def _create_actions(self):
        """ Create and connect actions, store in _actions dict """
        self._actions = {}

        a = act("Open Data Set", self,
                tip="Open a new data set",
                shortcut=QKeySequence.Open)
        a.triggered.connect(self._load_data_interactive)
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

        a = act('Gather Windows', self,
                tip='Gather plot windows side-by-side',
                shortcut='Ctrl+G')
        a.triggered.connect(self.gather_current_tab)
        self._actions['gather'] = a

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

        from ..config import qt_client
        from .widgets import ScatterWidget, ImageWidget

        if data and data.ndim == 1 and ScatterWidget in qt_client.members:
            default = qt_client.members.index(ScatterWidget)
        elif data and data.ndim in [2, 3] and ImageWidget in qt_client.members:
            default = qt_client.members.index(ImageWidget)
        else:
            default = 0

        client = pick_class(list(qt_client.members), title='Data Viewer',
                            label="Choose a new data viewer",
                            default=default)
        if client:
            c = client(self._data)
            c.register_to_hub(self._hub)
            if data and not c.add_data(data):
                c.close(warn=False)
                return

            self._add_to_current_tab(c)
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
        from ..core.glue_pickle import CloudPickler
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
        self._terminal_button.setArrowType(Qt.DownArrow)

        try:
            from .widgets.terminal import glue_terminal
            widget = glue_terminal(data_collection=self._data,
                                   dc=self._data,
                                   hub=self._hub,
                                   **vars(env))
            self._terminal_button.clicked.connect(self._toggle_terminal)
        except Exception as e:  # pylint: disable=W0703
            import traceback
            self._terminal_exception = traceback.format_exc()
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
               "\nReason:\n%s" % exception)

        def show_msg():
            mb = QMessageBox(QMessageBox.Critical,
                             title, msg)
            mb.setDetailedText(self._terminal_exception)
            mb.exec_()

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

    def start(self):
        self.show()
        self.raise_()  # bring window to front
        return self.app.exec_()

    exec_ = start

    def keyPressEvent(self, event):
        """Hold down modifier keys to temporarily set edit mode"""
        mod = event.modifiers()
        if mod == Qt.ShiftModifier:
            self._mode_toolbar.set_mode('or')

    def keyReleaseEvent(self, event):
        """Unset any temporary edit mode"""
        self._mode_toolbar.unset_mode()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            self._load_data(url.path())
        event.accept()
