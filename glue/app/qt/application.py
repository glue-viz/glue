# pylint: disable=W0223

from __future__ import absolute_import, division, print_function

import os
import sys
import warnings
import webbrowser

from qtpy import QtCore, QtWidgets, QtGui, compat
from qtpy.QtCore import Qt

from glue.core.application_base import Application
from glue.core import command, Data
from glue import env
from glue.main import load_plugins
from glue.icons.qt import get_icon
from glue.utils.qt import get_qapp
from glue.app.qt.actions import action
from glue.dialogs.data_wizard.qt import data_wizard
from glue.app.qt.edit_subset_mode_toolbar import EditSubsetModeToolBar
from glue.app.qt.mdi_area import GlueMdiArea, GlueMdiSubWindow
from glue.app.qt.layer_tree_widget import PlotAction, LayerTreeWidget
from glue.app.qt.preferences import PreferencesDialog
from glue.viewers.common.qt.mpl_widget import defer_draw
from glue.viewers.common.qt.data_viewer import DataViewer
from glue.viewers.image.qt import ImageWidget
from glue.viewers.scatter.qt import ScatterWidget
from glue.utils import nonpartial
from glue.utils.qt import (pick_class, GlueTabBar,
                           QMessageBoxPatched as QMessageBox,
                           set_cursor_cm, messagebox_on_error, load_ui)

from glue.app.qt.feedback import submit_bug_report, submit_feedback
from glue.app.qt.plugin_manager import QtPluginManager
from glue.app.qt.versions import show_glue_info


__all__ = ['GlueApplication']
DOCS_URL = 'http://www.glueviz.org'


def _fix_ipython_pylab():
    try:
        from IPython import get_ipython
    except ImportError:
        return
    shell = get_ipython()
    if shell is None:
        return

    from IPython.core.error import UsageError

    try:
        shell.enable_pylab('inline', import_all=True)
    except ValueError:
        # if the shell is a normal terminal shell, we get here
        pass
    except UsageError:
        pass


def status_pixmap(attention=False):
    """
    A small icon to grab attention

    :param attention: If True, return attention-grabbing pixmap
    """
    color = Qt.red if attention else Qt.lightGray

    pm = QtGui.QPixmap(15, 15)
    p = QtGui.QPainter(pm)
    b = QtGui.QBrush(color)
    p.fillRect(-1, -1, 20, 20, b)
    return pm


class ClickableLabel(QtWidgets.QLabel):

    """
    A QtWidgets.QLabel you can click on to generate events
    """

    clicked = QtCore.Signal()

    def mousePressEvent(self, event):
        self.clicked.emit()


class GlueLogger(QtWidgets.QWidget):

    """
    A window to display error messages
    """

    def __init__(self, parent=None):
        super(GlueLogger, self).__init__(parent)
        self._text = QtWidgets.QTextEdit()
        self._text.setTextInteractionFlags(Qt.TextSelectableByMouse)

        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(nonpartial(self._clear))

        report = QtWidgets.QPushButton("Send Bug Report")
        report.clicked.connect(nonpartial(self._send_report))

        self.stderr = sys.stderr
        sys.stderr = self

        self._status = ClickableLabel()
        self._status.setToolTip("View Errors and Warnings")
        self._status.clicked.connect(self._show)
        self._status.setPixmap(status_pixmap())
        self._status.setContentsMargins(0, 0, 0, 0)

        l = QtWidgets.QVBoxLayout()
        h = QtWidgets.QHBoxLayout()
        l.setContentsMargins(2, 2, 2, 2)
        l.setSpacing(2)
        h.setContentsMargins(0, 0, 0, 0)

        l.addWidget(self._text)
        h.insertStretch(0)
        h.addWidget(report)
        h.addWidget(clear)
        l.addLayout(h)

        self.setLayout(l)

    @property
    def status_light(self):
        """
        The icon representing the status of the log
        """
        return self._status

    def write(self, message):
        """
        Interface for sys.excepthook
        """
        self.stderr.write(message)
        self._text.moveCursor(QtGui.QTextCursor.End)
        self._text.insertPlainText(message)
        self._status.setPixmap(status_pixmap(attention=True))

    def flush(self):
        """
        Interface for sys.excepthook
        """
        pass

    def _send_report(self):
        """
        Send the contents of the log as a bug report
        """
        text = self._text.document().toPlainText()
        submit_bug_report(text)

    def _clear(self):
        """
        Erase the log
        """
        self._text.setText('')
        self._status.setPixmap(status_pixmap(attention=False))
        self.close()

    def _show(self):
        """
        Show the log
        """
        self.show()
        self.raise_()

    def keyPressEvent(self, event):
        """
        Hide window on escape key
        """
        if event.key() == Qt.Key_Escape:
            self.hide()


class GlueApplication(Application, QtWidgets.QMainWindow):

    """ The main GUI application for the Qt frontend"""

    def __init__(self, data_collection=None, session=None, maximized=True):

        self.app = get_qapp()

        QtWidgets.QMainWindow.__init__(self)
        Application.__init__(self, data_collection=data_collection,
                             session=session)

        self.app.setQuitOnLastWindowClosed(True)
        icon = get_icon('app_icon')
        self.app.setWindowIcon(icon)

        # Even though we loaded the plugins in start_glue, we re-load them here
        # in case glue was started directly by initializing this class.
        load_plugins()

        self.setWindowTitle("Glue")
        self.setWindowIcon(icon)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._actions = {}
        self._terminal = None
        self._setup_ui()
        self.tab_widget.setMovable(True)
        self.tab_widget.setTabsClosable(True)

        # The following is a counter that never goes down, even if tabs are
        # deleted (this is by design, to avoid having two tabs called the
        # same if a tab is removed then a new one added again)
        self._total_tab_count = 0

        lwidget = self._layer_widget
        a = PlotAction(lwidget, self)
        lwidget.ui.layerTree.addAction(a)
        lwidget.bind_selection_to_edit_subset()

        self._tweak_geometry(maximized=maximized)
        self._create_actions()
        self._create_menu()
        self._connect()
        self.new_tab()
        self._update_plot_dashboard(None)

    def _setup_ui(self):
        self._ui = load_ui('application.ui', None,
                           directory=os.path.dirname(__file__))
        self.setCentralWidget(self._ui)
        self._ui.tabWidget.setTabBar(GlueTabBar())

        lw = LayerTreeWidget()
        lw.set_checkable(False)
        self._vb = QtWidgets.QVBoxLayout()
        self._vb.setContentsMargins(0, 0, 0, 0)
        self._vb.addWidget(lw)
        self._ui.data_layers.setLayout(self._vb)
        self._layer_widget = lw

        # log window + status light
        self._log = GlueLogger()
        self._log.window().setWindowTitle("Console Log")
        self._log.resize(550, 550)
        self.statusBar().addPermanentWidget(self._log.status_light)
        self.statusBar().setContentsMargins(2, 0, 20, 2)
        self.statusBar().setSizeGripEnabled(False)

    def _tweak_geometry(self, maximized=True):
        """Maximize window by default."""
        if maximized:
            self.setWindowState(Qt.WindowMaximized)
        self._ui.main_splitter.setSizes([100, 800])
        self._ui.data_plot_splitter.setSizes([100, 150, 250])

    @property
    def tab_widget(self):
        return self._ui.tabWidget

    @property
    def tab_bar(self):
        return self._ui.tabWidget.tabBar()

    @property
    def tab_count(self):
        """
        The number of open tabs
        """
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
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(1)
        layout.setContentsMargins(0, 0, 0, 0)
        widget = GlueMdiArea(self)
        widget.setLayout(layout)
        tab = self.tab_widget
        self._total_tab_count += 1
        tab.addTab(widget, str("Tab %i" % self._total_tab_count))
        tab.setCurrentWidget(widget)
        widget.subWindowActivated.connect(self._update_plot_dashboard)

    def close_tab(self, index):
        """ Close a tab window and all associated data viewers """

        # do not delete the last tab
        if self.tab_widget.count() == 1:
            return

        if not os.environ.get('GLUE_TESTING'):
            buttons = QMessageBox.Ok | QMessageBox.Cancel
            dialog = QMessageBox.warning(
                self, "Confirm Close",
                "Are you sure you want to close this tab? "
                "This will close all data viewers in the tab.",
                buttons=buttons, defaultButton=QMessageBox.Cancel)
            if not dialog == QMessageBox.Ok:
                return

        w = self.tab_widget.widget(index)

        for window in w.subWindowList():
            widget = window.widget()
            if isinstance(widget, DataViewer):
                widget.close(warn=False)

        w.close()

        self.tab_widget.removeTab(index)

    def add_widget(self, new_widget, label=None, tab=None,
                   hold_position=False):
        """
        Add a widget to one of the tabs.

        Returns the window that this widget is wrapped in.

        :param new_widget: new QtWidgets.QWidget to add

        :param label: label for the new window. Optional
        :type label: str

        :param tab: Tab to add to. Optional (default: current tab)
        :type tab: int

        :param hold_position: If True, then override Qt's default
                              placement and retain the original position
                              of new_widget
        :type hold_position: bool
        """
        page = self.tab(tab)
        pos = getattr(new_widget, 'position', None)
        sub = new_widget.mdi_wrap()

        sub.closed.connect(self._clear_dashboard)

        if label:
            sub.setWindowTitle(label)
        page.addSubWindow(sub)
        page.setActiveSubWindow(sub)
        if hold_position and pos is not None:
            new_widget.move(pos[0], pos[1])
        return sub

    def _edit_settings(self):
        self._editor = PreferencesDialog(self)
        self._editor.show()

    def gather_current_tab(self):
        """Arrange windows in current tab via tiling"""
        self.current_tab.tileSubWindows()

    def _get_plot_dashboards(self, sub_window):

        if not isinstance(sub_window, GlueMdiSubWindow):
            return QtWidgets.QWidget(), QtWidgets.QWidget(), ""

        widget = sub_window.widget()
        if not isinstance(widget, DataViewer):
            return QtWidgets.QWidget(), QtWidgets.QWidget(), ""

        layer_view = widget.layer_view()
        options_widget = widget.options_widget()

        return layer_view, options_widget, str(widget)

    def _clear_dashboard(self):

        for widget, title in [(self._ui.plot_layers, "Plot Layers"),
                              (self._ui.plot_options, "Plot Options")]:
            layout = widget.layout()
            if layout is None:
                layout = QtWidgets.QVBoxLayout()
                layout.setContentsMargins(4, 4, 4, 4)
                widget.setLayout(layout)
            while layout.count():
                layout.takeAt(0).widget().hide()
            widget.setTitle(title)

    def _update_plot_dashboard(self, sub_window):
        self._clear_dashboard()

        if sub_window is None:
            return

        layer_view, options_widget, title = self._get_plot_dashboards(
            sub_window)

        layout = self._ui.plot_layers.layout()
        layout.addWidget(layer_view)

        layout = self._ui.plot_options.layout()
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
        self._layer_widget.setup(self._data)

        self.tab_widget.tabCloseRequested.connect(self.close_tab)

    def _create_menu(self):
        mbar = self.menuBar()
        menu = QtWidgets.QMenu(mbar)
        menu.setTitle("&File")

        menu.addAction(self._actions['data_new'])
        if 'data_importers' in self._actions:
            submenu = menu.addMenu("I&mport data")
            for a in self._actions['data_importers']:
                submenu.addAction(a)
        # menu.addAction(self._actions['data_save'])  # XXX add this
        menu.addAction(self._actions['session_reset'])
        menu.addAction(self._actions['session_restore'])
        menu.addAction(self._actions['session_save'])
        if 'session_export' in self._actions:
            submenu = menu.addMenu("E&xport")
            for a in self._actions['session_export']:
                submenu.addAction(a)
        menu.addSeparator()
        menu.addAction("Edit &Preferences", self._edit_settings)
        menu.addAction("&Quit", self.app.quit)
        mbar.addMenu(menu)

        menu = QtWidgets.QMenu(mbar)
        menu.setTitle("&Edit ")
        menu.addAction(self._actions['undo'])
        menu.addAction(self._actions['redo'])
        mbar.addMenu(menu)

        menu = QtWidgets.QMenu(mbar)
        menu.setTitle("&View ")

        a = QtWidgets.QAction("&Console Log", menu)
        a.triggered.connect(self._log._show)
        menu.addAction(a)
        mbar.addMenu(menu)

        menu = QtWidgets.QMenu(mbar)
        menu.setTitle("&Canvas")
        menu.addAction(self._actions['tab_new'])
        menu.addAction(self._actions['viewer_new'])
        menu.addSeparator()
        menu.addAction(self._actions['gather'])
        menu.addAction(self._actions['tab_rename'])
        mbar.addMenu(menu)

        menu = QtWidgets.QMenu(mbar)
        menu.setTitle("Data &Manager")
        menu.addActions(self._layer_widget.actions())

        mbar.addMenu(menu)

        menu = QtWidgets.QMenu(mbar)
        menu.setTitle("&Toolbars")
        tbar = EditSubsetModeToolBar()
        self._mode_toolbar = tbar
        self.addToolBar(tbar)
        tbar.hide()
        a = QtWidgets.QAction("Selection Mode &Toolbar", menu)
        a.setCheckable(True)
        a.toggled.connect(tbar.setVisible)
        try:
            tbar.visibilityChanged.connect(a.setChecked)
        except AttributeError:  # Qt < 4.7. QtCore.Signal not supported
            pass

        menu.addAction(a)
        menu.addActions(tbar.actions())
        mbar.addMenu(menu)

        menu = QtWidgets.QMenu(mbar)
        menu.setTitle("&Plugins")
        menu.addAction(self._actions['plugin_manager'])
        menu.addSeparator()

        if 'plugins' in self._actions:
            for plugin in self._actions['plugins']:
                menu.addAction(plugin)

        mbar.addMenu(menu)

        # trigger inclusion of Mac Native "Help" tool
        menu = mbar.addMenu("&Help")

        a = QtWidgets.QAction("&Online Documentation", menu)
        a.triggered.connect(nonpartial(webbrowser.open, DOCS_URL))
        menu.addAction(a)

        a = QtWidgets.QAction("Send &Feedback", menu)
        a.triggered.connect(nonpartial(submit_feedback))
        menu.addAction(a)

        menu.addSeparator()
        menu.addAction("Version information", show_glue_info)

    def _choose_load_data(self, data_importer=None):
        if data_importer is None:
            self.add_datasets(self.data_collection, data_wizard())
        else:
            data = data_importer()
            if not isinstance(data, list):
                raise TypeError("Data loader should return list of "
                                "Data objects")
            for item in data:
                if not isinstance(item, Data):
                    raise TypeError("Data loader should return list of "
                                    "Data objects")
            self.add_datasets(self.data_collection, data)

    def _create_actions(self):
        """ Create and connect actions, store in _actions dict """
        self._actions = {}

        a = action("&New Data Viewer", self,
                   tip="Open a new visualization window in the current tab",
                   shortcut=QtGui.QKeySequence.New)
        a.triggered.connect(nonpartial(self.choose_new_data_viewer))
        self._actions['viewer_new'] = a

        a = action('New &Tab', self,
                   shortcut=QtGui.QKeySequence.AddTab,
                   tip='Add a new tab')
        a.triggered.connect(nonpartial(self.new_tab))
        self._actions['tab_new'] = a

        a = action('&Rename Tab', self,
                   shortcut="Ctrl+R",
                   tip='Set a new label for the current tab')
        a.triggered.connect(nonpartial(self.tab_bar.rename_tab))
        self._actions['tab_rename'] = a

        a = action('&Gather Windows', self,
                   tip='Gather plot windows side-by-side',
                   shortcut='Ctrl+G')
        a.triggered.connect(nonpartial(self.gather_current_tab))
        self._actions['gather'] = a

        a = action('&Save Session', self,
                   tip='Save the current session')
        a.triggered.connect(nonpartial(self._choose_save_session))
        self._actions['session_save'] = a

        # Add file loader as first item in File menu for convenience. We then
        # also add it again below in the Import menu for consistency.
        a = action("&Open Data Set", self, tip="Open a new data set",
                   shortcut=QtGui.QKeySequence.Open)
        a.triggered.connect(nonpartial(self._choose_load_data,
                                       data_wizard))
        self._actions['data_new'] = a

        # We now populate the "Import data" menu
        from glue.config import importer

        acts = []

        # Add default file loader (later we can add this to the registry)
        a = action("Import from file", self, tip="Import from file")
        a.triggered.connect(nonpartial(self._choose_load_data,
                                       data_wizard))
        acts.append(a)

        for i in importer:
            label, data_importer = i
            a = action(label, self, tip=label)
            a.triggered.connect(nonpartial(self._choose_load_data,
                                           data_importer))
            acts.append(a)

        self._actions['data_importers'] = acts

        from glue.config import exporters
        if len(exporters) > 0:
            acts = []
            for e in exporters:
                label, saver, checker, mode = e
                a = action(label, self,
                           tip='Export the current session to %s format' %
                           label)
                a.triggered.connect(nonpartial(self._choose_export_session,
                                               saver, checker, mode))
                acts.append(a)

            self._actions['session_export'] = acts

        a = action('Open S&ession', self,
                   tip='Restore a saved session')
        a.triggered.connect(nonpartial(self._restore_session))
        self._actions['session_restore'] = a

        a = action('Reset S&ession', self,
                   tip='Reset session to clean state')
        a.triggered.connect(nonpartial(self._reset_session))
        self._actions['session_reset'] = a

        a = action("Undo", self,
                   tip='Undo last action',
                   shortcut=QtGui.QKeySequence.Undo)
        a.triggered.connect(nonpartial(self.undo))
        a.setEnabled(False)
        self._actions['undo'] = a

        a = action("Redo", self,
                   tip='Redo last action',
                   shortcut=QtGui.QKeySequence.Redo)
        a.triggered.connect(nonpartial(self.redo))
        a.setEnabled(False)
        self._actions['redo'] = a

        # Create actions for menubar plugins
        from glue.config import menubar_plugin
        acts = []
        for label, function in menubar_plugin:
            a = action(label, self, tip=label)
            a.triggered.connect(nonpartial(function,
                                           self.session,
                                           self.data_collection))
            acts.append(a)
        self._actions['plugins'] = acts

        a = action('&Plugin Manager', self,
                   tip='Open plugin manager')
        a.triggered.connect(nonpartial(self.plugin_manager))
        self._actions['plugin_manager'] = a

    def choose_new_data_viewer(self, data=None):
        """ Create a new visualization window in the current tab
        """

        from glue.config import qt_client

        if data and data.ndim == 1 and ScatterWidget in qt_client.members:
            default = ScatterWidget
        elif data and data.ndim > 1 and ImageWidget in qt_client.members:
            default = ImageWidget
        else:
            default = None

        client = pick_class(list(qt_client.members), title='Data Viewer',
                            label="Choose a new data viewer",
                            default=default, sort=True)

        cmd = command.NewDataViewer(viewer=client, data=data)
        return self.do(cmd)

    new_data_viewer = defer_draw(Application.new_data_viewer)

    def _choose_save_session(self):
        """ Save the data collection and hub to file.

        Can be restored via restore_session
        """

        # include file filter twice, so it shows up in Dialog
        outfile, file_filter = compat.getsavefilename(
            parent=self, filters=("Glue Session (*.glu);; "
                                  "Glue Session including data (*.glu)"))

        # This indicates that the user cancelled
        if not outfile:
            return

        # Add extension if not specified
        if '.' not in outfile:
            outfile += '.glu'

        with set_cursor_cm(Qt.WaitCursor):
            self.save_session(
                outfile, include_data="including data" in file_filter)

    @messagebox_on_error("Failed to export session")
    def _choose_export_session(self, saver, checker, outmode):
        checker(self)
        if outmode is None:
            return saver(self)
        elif outmode in ['file', 'directory']:
            outfile, file_filter = compat.getsavefilename(parent=self)
            if not outfile:
                return
            return saver(self, outfile)
        else:
            assert outmode == 'label'
            label, ok = QtWidgets.QInputDialog.getText(self, 'Choose a label:',
                                                       'Choose a label:')
            if not ok:
                return
            return saver(self, label)

    @messagebox_on_error("Failed to restore session")
    def _restore_session(self, show=True):
        """ Load a previously-saved state, and restart the session """
        fltr = "Glue sessions (*.glu)"
        file_name, file_filter = compat.getopenfilename(
            parent=self, filters=fltr)
        if not file_name:
            return

        with set_cursor_cm(Qt.WaitCursor):
            ga = self.restore_session(file_name)
            self.close()
        return ga

    def _reset_session(self, show=True):
        """
        Reset session to clean state.
        """

        if not os.environ.get('GLUE_TESTING'):
            buttons = QMessageBox.Ok | QMessageBox.Cancel
            dialog = QMessageBox.warning(
                self, "Confirm Close",
                "Are you sure you want to reset the session? "
                "This will close all datasets, subsets, and data viewers",
                buttons=buttons, defaultButton=QMessageBox.Cancel)
            if not dialog == QMessageBox.Ok:
                return

        ga = GlueApplication()
        ga.show()
        self.close()

        return ga

    @staticmethod
    def restore_session(path, show=True):
        """
        Reload a previously-saved session

        Parameters
        ----------
        path : str
            Path to the file to load
        show : bool, optional
            If True (the default), immediately show the widget

        Returns
        -------
        app : :class:`glue.app.qt.application.GlueApplication`
            The loaded application
        """
        ga = Application.restore_session(path)
        if show:
            ga.show()
        return ga

    def has_terminal(self):
        """
        Returns True if the IPython terminal is present.
        """
        self._create_terminal()  # ensure terminal is setup
        return self._terminal is not None

    def _create_terminal(self):
        if self._terminal is not None:  # already set up
            return

        if hasattr(self, '_terminal_exception'):  # already failed to set up
            return

        self._terminal_button = QtWidgets.QToolButton(self._ui)
        self._terminal_button.setToolTip("Toggle IPython Prompt")
        i = get_icon('IPythonConsole')
        self._terminal_button.setIcon(i)
        self._terminal_button.setIconSize(QtCore.QSize(25, 25))

        self._layer_widget.ui.button_row.addWidget(self._terminal_button)

        try:
            from glue.app.qt.terminal import glue_terminal
            widget = glue_terminal(data_collection=self._data,
                                   dc=self._data,
                                   hub=self._hub,
                                   session=self.session,
                                   application=self,
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
            if self._terminal.isVisible():
                warnings.warn("An unexpected error occurred while "
                              "trying to hide the terminal")
        else:
            self._show_terminal()
            if not self._terminal.isVisible():
                warnings.warn("An unexpected error occurred while "
                              "trying to show the terminal")

    def _hide_terminal(self):
        self._terminal.hide()

    def _show_terminal(self):
        self._terminal.show()
        self._terminal.widget().show()

    def start(self, size=None, position=None):
        """
        Show the GUI and start the application.

        Parameters
        ----------
        size : (int, int) Optional
            The default width/height of the application.
            If not provided, uses the full screen
        position : (int, int) Optional
            The default position of the application
        """
        self._create_terminal()
        self.show()
        if size is not None:
            self.resize(*size)
        if position is not None:
            self.move(*position)

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

            # Get path to file
            path = url.path()

            # Workaround for a Qt bug that causes paths to start with a /
            # on Windows: https://bugreports.qt.io/browse/QTBUG-46417
            if sys.platform.startswith('win'):
                if path.startswith('/') and path[2] == ':':
                    path = path[1:]

            self.load_data(path)
        event.accept()

    def report_error(self, message, detail):
        """
        Display an error in a modal

        :param message: A short description of the error
        :type message: str
        :param detail: A longer description
        :type detail: str
        """
        qmb = QMessageBox(QMessageBox.Critical, "Error", message)
        qmb.setDetailedText(detail)
        qmb.resize(400, qmb.size().height())
        qmb.exec_()

    def plugin_manager(self):
        from glue.main import _installed_plugins
        pm = QtPluginManager(installed=_installed_plugins)
        pm.ui.exec_()

    def _update_undo_redo_enabled(self):
        undo, redo = self._cmds.can_undo_redo()
        self._actions['undo'].setEnabled(undo)
        self._actions['redo'].setEnabled(redo)
        self._actions['undo'].setText('Undo ' + self._cmds.undo_label)
        self._actions['redo'].setText('Redo ' + self._cmds.redo_label)

    @property
    def viewers(self):
        """
        A list of lists of open Data Viewers.

        Each inner list contains the viewers open on a particular tab.
        """
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
        """
        The name of each tab

        A list of strings
        """
        return [self.tab_bar.tabText(i) for i in range(self.tab_count)]

    @staticmethod
    def _choose_merge(data, others):

        w = load_ui('merge.ui', None, directory=os.path.dirname(__file__))
        w.button_yes.clicked.connect(w.accept)
        w.button_no.clicked.connect(w.reject)
        w.show()
        w.raise_()

        # Add the main dataset to the list. Some of the 'others' may also be
        # new ones, so it doesn't really make sense to distinguish between
        # the two here. The main point is that some datasets, including at
        # least one new one, have a common shape.
        others.append(data)
        others.sort(key=lambda x: x.label)

        label = others[0].label
        w.merged_label.setText(label)

        entries = [QtWidgets.QListWidgetItem(other.label) for other in others]
        for e in entries:
            e.setCheckState(Qt.Checked)

        for d, item in zip(others, entries):
            w.choices.addItem(item)

        if not w.exec_():
            return None, None

        result = [layer for layer, entry in zip(others, entries)
                  if entry.checkState() == Qt.Checked]

        if result:
            return result, str(w.merged_label.text())

        return None, None
