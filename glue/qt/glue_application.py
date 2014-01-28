# pylint: disable=W0223
import sys
from functools import partial

from ..external.qt.QtGui import (QKeySequence, QMainWindow, QGridLayout,
                                 QMenu, QMdiSubWindow, QAction, QMessageBox,
                                 QFileDialog, QInputDialog,
                                 QToolButton, QVBoxLayout, QWidget)
from ..external.qt.QtCore import Qt, QSize, QSettings

from ..core import command, Session
from .. import env
from ..qt import get_qapp
from .decorators import set_cursor, messagebox_on_error
from ..core.application_base import Application

from .actions import act
from .qtutil import pick_class, data_wizard, GlueTabBar, load_ui, get_icon
from .widgets.glue_mdi_area import GlueMdiArea
from .widgets.edit_subset_mode_toolbar import EditSubsetModeToolBar
from .widgets.layer_tree_widget import PlotAction, LayerTreeWidget
from .widgets.data_viewer import DataViewer
from .widgets.settings_editor import SettingsEditor


def _fix_ipython_pylab():
    try:
        from IPython import get_ipython
    except ImportError:
        return
    shell = get_ipython()
    if shell is None:
        return
    try:
        shell.enable_pylab('inline', import_all=True)
    except ValueError:
        # if the shell is a normal terminal shell, we get here
        pass


class GlueApplication(Application, QMainWindow):

    """ The main Glue window """

    def __init__(self, data_collection=None, session=None):
        QMainWindow.__init__(self)
        Application.__init__(self, data_collection=data_collection,
                             session=session)

        self.app = get_qapp()

        self.setWindowIcon(self.app.windowIcon())
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._actions = {}
        self._terminal = None
        self._setup_ui()
        self.tab_widget.setMovable(True)
        self.tab_widget.setTabsClosable(True)

        lwidget = self._ui.layerWidget
        act = PlotAction(lwidget, self)
        lwidget.layerTree.addAction(act)
        lwidget.bind_selection_to_edit_subset()

        self._tweak_geometry()
        self._create_actions()
        self._create_menu()
        self._connect()
        self.new_tab()
        self._create_terminal()
        self._update_plot_dashboard(None)

    def _setup_ui(self):
        self._ui = load_ui('glue_application', None)
        self.setCentralWidget(self._ui)
        self._ui.tabWidget.setTabBar(GlueTabBar())

        lw = LayerTreeWidget()
        lw.set_checkable(False)
        vb = QVBoxLayout()
        vb.setContentsMargins(0, 0, 0, 0)
        vb.addWidget(lw)
        self._ui.data_layers.setLayout(vb)
        self._ui.layerWidget = lw

    def _tweak_geometry(self):
        """Maximize window"""
        self.setWindowState(Qt.WindowMaximized)
        self._ui.main_splitter.setSizes([100, 800])
        self._ui.data_plot_splitter.setSizes([100, 400])
        self._ui.plot_splitter.setSizes([150, 250])

    @property
    def tab_widget(self):
        return self._ui.tabWidget

    @property
    def tab_bar(self):
        return self._ui.tabWidget.tabBar()

    @property
    def tab_count(self):
        return self._ui.tabWidget.count()

    @property
    def current_tab(self):
        return self._ui.tabWidget.currentWidget()

    def tab(self, index=None):
        if index is None:
            return self.current_tab
        return self._ui.tabWidget.widget(index)

    def new_tab(self):
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

    def close_tab(self, index):
        """ Close a tab window and all associated data viewers """
        # do not delete the last tab
        if self.tab_widget.count() == 1:
            return
        w = self.tab_widget.widget(index)
        w.close()
        self.tab_widget.removeTab(index)

    def add_widget(self, new_widget, label=None, tab=None):
        """ Add a widget to one of the tabs

        :param new_widget: Widge to add
        :type new_widget: QWidget

        :param label: label for the new window. Optional
        :type label: str

        :param tab: Tab to add to. Optional (default: current tab)
        :type tab: int

        :rtype: QMdiSubWindow. The window that this widget is wrapped in
        """
        page = self.tab(tab)

        sub = new_widget.mdi_wrap()
        if label:
            sub.setWindowTitle(label)
        page.addSubWindow(sub)
        page.setActiveSubWindow(sub)
        return sub

    def set_setting(self, key, value):
        super(GlueApplication, self).set_setting(key, value)
        settings = QSettings('glue-viz', 'glue')
        settings.setValue(key, value)

    def _load_settings(self, path=None):
        settings = QSettings('glue-viz', 'glue')
        for k, v in self.settings:
            if settings.contains(k):
                super(GlueApplication, self).set_setting(k, settings.value(k))

    def _edit_settings(self):
        # save it to prevent garbage collection
        self._editor = SettingsEditor(self)
        self._editor.widget.show()

    def gather_current_tab(self):
        """Arrange windows in current tab via tiling"""
        self.current_tab.tileSubWindows()

    def _get_plot_dashboards(self, sub_window):
        if not isinstance(sub_window, QMdiSubWindow):
            return QWidget(), QWidget(), ""

        widget = sub_window.widget()
        if not isinstance(widget, DataViewer):
            return QWidget(), QWidget(), ""

        return widget.layer_view(), widget.options_widget(), str(widget)

    def _update_plot_dashboard(self, sub_window):
        if sub_window is None:
            return

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

    def _connect(self):
        self.setAcceptDrops(True)
        self._ui.layerWidget.setup(self._data, self._hub)

        def sethelp(*args):
            model = self._ui.layerWidget.layerTree.model()
            self.current_tab.show_help = model.rowCount() > 0

        model = self._ui.layerWidget.layerTree.model()
        model.rowsInserted.connect(sethelp)
        model.rowsRemoved.connect(sethelp)

        self._data.register_to_hub(self._hub)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)

    def _create_menu(self):
        mbar = self.menuBar()
        menu = QMenu(mbar)
        menu.setTitle("File")

        menu.addAction(self._actions['data_new'])
        # menu.addAction(self._actions['data_save'])  # XXX add this
        menu.addAction(self._actions['session_restore'])
        menu.addAction(self._actions['session_save'])
        if 'session_export' in self._actions:
            submenu = menu.addMenu("Export")
            for a in self._actions['session_export']:
                submenu.addAction(a)
        menu.addSeparator()
        menu.addAction("Edit Settings", self._edit_settings)
        mbar.addMenu(menu)

        menu = QMenu(mbar)
        menu.setTitle("Edit ")
        menu.addAction(self._actions['undo'])
        menu.addAction(self._actions['redo'])
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

        # trigger inclusion of Mac Native "Help" tool
        if sys.platform == 'darwin':
            mbar.addMenu('Help')

    def _choose_load_data(self):
        for d in data_wizard():
            self._data.append(d)

    def _create_actions(self):
        """ Create and connect actions, store in _actions dict """
        self._actions = {}

        a = act("Open Data Set", self,
                tip="Open a new data set",
                shortcut=QKeySequence.Open)
        a.triggered.connect(self._choose_load_data)
        self._actions['data_new'] = a

        a = act("New Data Viewer", self,
                tip="Open a new visualization window in the current tab",
                shortcut=QKeySequence.New
                )
        a.triggered.connect(self._choose_new_data_viewer)
        self._actions['viewer_new'] = a

        a = act('New Tab', self,
                shortcut=QKeySequence.AddTab,
                tip='Add a new tab')
        a.triggered.connect(self.new_tab)
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
        a.triggered.connect(lambda *args: self._choose_save_session())
        self._actions['session_save'] = a

        from glue.config import exporters
        if len(exporters) > 0:
            acts = []
            name = 'Export Session'
            for e in exporters:
                label, saver, checker, mode = e
                a = act(label, self,
                        tip='Export the current session to %s format' %
                        label)
                a.triggered.connect(partial(self._choose_export_session,
                                            saver, checker, mode))
                acts.append(a)

            self._actions['session_export'] = acts

        a = act('Open Session', self,
                tip='Restore a saved session')
        a.triggered.connect(lambda *args: self._restore_session())
        self._actions['session_restore'] = a

        a = act("Undo", self,
                tip='Undo last action',
                shortcut=QKeySequence.Undo)
        a.triggered.connect(lambda *args: self.undo())
        a.setEnabled(False)
        self._actions['undo'] = a

        a = act("Redo", self,
                tip='Redo last action',
                shortcut=QKeySequence.Redo)
        a.triggered.connect(lambda *args: self.redo())
        a.setEnabled(False)
        self._actions['redo'] = a

    def _choose_new_data_viewer(self, data=None):
        """ Create a new visualization window in the current tab
        """

        from ..config import qt_client
        from .widgets import ScatterWidget, ImageWidget

        if data and data.ndim == 1 and ScatterWidget in qt_client.members:
            default = qt_client.members.index(ScatterWidget)
        elif data and data.ndim > 1 and ImageWidget in qt_client.members:
            default = qt_client.members.index(ImageWidget)
        else:
            default = 0

        client = pick_class(list(qt_client.members), title='Data Viewer',
                            label="Choose a new data viewer",
                            default=default)

        cmd = command.NewDataViewer(viewer=client, data=data)
        return self.do(cmd)

    @set_cursor(Qt.WaitCursor)
    def _choose_save_session(self):
        """ Save the data collection and hub to file.

        Can be restored via restore_session

        Note: Saving of client is not currently supported. Thus,
        restoring this session will lose all current viz windows
        """
        outfile, file_filter = QFileDialog.getSaveFileName(self)
        if not outfile:
            return
        self.save_session(outfile)

    @messagebox_on_error("Failed to export session")
    def _choose_export_session(self, saver, checker, outmode):
        checker(self)
        if outmode in ['file', 'directory']:
            outfile, file_filter = QFileDialog.getSaveFileName(self)
            if not outfile:
                return
            return saver(self, outfile)
        else:
            assert outmode == 'label'
            label, ok = QInputDialog.getText(self, 'Choose a label:',
                                             'Choose a label:')
            if not ok:
                return
            return saver(self, label)

    @messagebox_on_error("Failed to restore session")
    @set_cursor(Qt.WaitCursor)
    def _restore_session(self, show=True):
        """ Load a previously-saved state, and restart the session """
        from pickle import Unpickler

        fltr = "Glue sessions (*.glu)"
        file_name, file_filter = QFileDialog.getOpenFileName(self,
                                                             filter=fltr)
        if not file_name:
            return

        state = Unpickler(open(file_name)).load()

        data, hub = state
        pos = self.pos()
        size = self.size()
        session = Session(data_collection=data, hub=hub)
        ga = GlueApplication(session=session)
        ga.move(pos)
        ga.resize(size)

        if show:
            ga.show()

        self.close()
        return ga

    def has_terminal(self):
        return self._terminal is not None

    def _create_terminal(self):
        assert self._terminal is None, \
            "should only call _create_terminal once"

        self._terminal_button = QToolButton(self._ui)
        self._terminal_button.setToolTip("Toggle IPython Prompt")
        i = get_icon('IPythonConsole')
        self._terminal_button.setIcon(i)
        self._terminal_button.setIconSize(QSize(25, 25))

        self._ui.layerWidget.button_row.addWidget(self._terminal_button)

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

        self._terminal = self.add_widget(widget, label='IPython')
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

    def _show_terminal(self):
        self._terminal.show()
        self._terminal.widget().show()

    def start(self):
        self.show()
        self.raise_()  # bring window to front
        # at some point during all this, the MPL backend
        # switches. This call restores things, so
        # figures are still inlined in the notebook.
        # XXX find out a better place for this
        _fix_ipython_pylab()
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

    def report_error(self, message, detail):
        qmb = QMessageBox(QMessageBox.Critical, "Error", message)
        qmb.setDetailedText(detail)
        qmb.resize(400, qmb.size().height())
        qmb.exec_()

    def _update_undo_redo_enabled(self):
        undo, redo = self._cmds.can_undo_redo()
        self._actions['undo'].setEnabled(undo)
        self._actions['redo'].setEnabled(redo)

    @property
    def viewers(self):
        result = []
        for t in range(self.tab_count):
            tab = self.tab(t)
            item = []
            for subwindow in tab.subWindowList():
                widget = subwindow.widget()
                if isinstance(widget, DataViewer):
                    item.append(widget)
            result.append(tuple(item))
        return tuple(result)

    @property
    def tab_names(self):
        return [self.tab_bar.tabText(i) for i in range(self.tab_count)]
