from __future__ import absolute_import, division, print_function

import os

from qtpy.QtCore import Qt
from qtpy import QtCore, QtWidgets
from glue.core import Data
from glue.core.application_base import ViewerBase
from glue.core.qt.layer_artist_model import QtLayerArtistContainer, LayerArtistWidget
from glue.utils.qt import get_qapp
from glue.core.qt.mime import LAYERS_MIME_TYPE, LAYER_MIME_TYPE
from glue.utils.qt import set_cursor
from glue.external import six
from glue.utils.noconflict import classmaker
from glue.core.state import save
from glue.config import viewer_tool
from glue.viewers.common.qt.tool import SimpleToolMenu
from glue.viewers.common.qt.toolbar import BasicToolbar

__all__ = ['DataViewer']


class ToolbarInitializer(object):
    """
    This is a meta-class which ensures that initialize_toolbar is always called
    on DataViewer instances and sub-class instances after all the __init__ code
    has been executed. We need to do this, because often the toolbar can only
    be initialized after everything else (e.g. canvas, etc.) has been set up,
    so we can't do it in DataViewer.__init__.
    """

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.initialize_toolbar()
        return obj


TEMPLATE_SCRIPT = """
# This script was produced by glue and can be used to further customize a
# particular plot.

### Package imports

{imports}

### Set up data

data_collection = load('{data}')

### Set up viewer

{header}

### Set up layers

{layers}

### Finalize viewer

{footer}
""".strip()


@viewer_tool
class SaveTool(SimpleToolMenu):
    """
    A generic 'save/export' tool that plugins can register new save/export tools
    with.

    To register a new save/export option, add an entry to the viewer
    ``subtools['save']`` list.
    """
    tool_id = 'save'
    icon = 'glue_filesave'
    tool_tip = 'Save/export the plot'


# Note: we need to use classmaker here because otherwise we run into issues when
# trying to use the meta-class with the Qt class.

@six.add_metaclass(classmaker(left_metas=(ToolbarInitializer,)))
class DataViewer(ViewerBase, QtWidgets.QMainWindow):
    """
    Base class for all Qt DataViewer widgets.

    This defines a minimal interface, and implemlements the following::

       * An automatic call to unregister on window close
       * Drag and drop support for adding data
    """

    window_closed = QtCore.Signal()
    toolbar_added = QtCore.Signal()

    _layer_artist_container_cls = QtLayerArtistContainer
    _layer_style_widget_cls = None

    LABEL = 'Override this'

    _toolbar_cls = BasicToolbar
    # This defines the mouse mode to be used when no toolbar modes are active
    _default_mouse_mode_cls = None
    _inherit_tools = True
    tools = ['save']
    subtools = {'save': []}

    _close_on_last_layer_removed = True

    _closed = False

    def __init__(self, session, parent=None):
        """
        :type session: :class:`~glue.core.Session`
        """
        QtWidgets.QMainWindow.__init__(self, parent)
        ViewerBase.__init__(self, session)
        self.setWindowIcon(get_qapp().windowIcon())
        self._view = LayerArtistWidget(layer_style_widget_cls=self._layer_style_widget_cls,
                                       hub=session.hub)
        self._view.layer_list.setModel(self._layer_artist_container.model)
        self._tb_vis = {}  # store whether toolbars are enabled
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setAcceptDrops(True)
        self.setAnimated(False)
        self.toolbar = None
        self._toolbars = []
        self._warn_close = True
        self.setContentsMargins(2, 2, 2, 2)
        self._mdi_wrapper = None  # GlueMdiSubWindow that self is embedded in
        self.statusBar().setStyleSheet("QStatusBar{font-size:10px}")

        # close window when last plot layer deleted
        if self._close_on_last_layer_removed:
            self._layer_artist_container.on_empty(lambda: self.close(warn=False))
        self._layer_artist_container.on_changed(self.update_window_title)

    @property
    def selected_layer(self):
        return self._view.layer_list.current_artist()

    def remove_layer(self, layer):
        self._layer_artist_container.pop(layer)

    def dragEnterEvent(self, event):
        """ Accept the event if it has data layers"""
        if event.mimeData().hasFormat(LAYER_MIME_TYPE):
            event.accept()
        elif event.mimeData().hasFormat(LAYERS_MIME_TYPE):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """ Add layers to the viewer if contained in mime data """
        if event.mimeData().hasFormat(LAYER_MIME_TYPE):
            self.request_add_layer(event.mimeData().data(LAYER_MIME_TYPE))

        assert event.mimeData().hasFormat(LAYERS_MIME_TYPE)

        for layer in event.mimeData().data(LAYERS_MIME_TYPE):
            self.request_add_layer(layer)

        event.accept()

    def mousePressEvent(self, event):
        """ Consume mouse press events, and prevent them from propagating
            down to the MDI area """
        event.accept()

    apply_roi = set_cursor(Qt.WaitCursor)(ViewerBase.apply_roi)

    def close(self, warn=True):

        if self._closed:
            return

        if warn and not self._confirm_close():
            return

        self._warn_close = False

        if getattr(self, '_mdi_wrapper', None) is not None:
            self._mdi_wrapper.close()
            self._mdi_wrapper = None
        else:
            try:
                QtWidgets.QMainWindow.close(self)
            except RuntimeError:
                # In some cases the above can raise a "wrapped C/C++ object of
                # type ... has been deleted" error, in which case we can just
                # ignore and carry on.
                pass
            ViewerBase.close(self)

        # We tell the toolbar to do cleanup to make sure we get rid of any
        # circular references
        if self.toolbar:
            self.toolbar.cleanup()

        self._warn_close = True

        self._closed = True

    def mdi_wrap(self):
        """Wrap this object in a GlueMdiSubWindow"""
        from glue.app.qt.mdi_area import GlueMdiSubWindow
        sub = GlueMdiSubWindow()
        sub.setWidget(self)
        self.destroyed.connect(sub.close)
        sub.resize(self.size())
        self._mdi_wrapper = sub

        return sub

    @property
    def position(self):
        target = self._mdi_wrapper or self
        pos = target.pos()
        return pos.x(), pos.y()

    @position.setter
    def position(self, xy):
        x, y = xy
        self.move(x, y)

    def move(self, x=None, y=None):
        """
        Move the viewer to a new XY pixel location

        You can also set the position attribute to a new tuple directly.

        Parameters
        ----------
        x : int (optional)
           New x position
        y : int (optional)
           New y position
        """
        x0, y0 = self.position
        if x is None:
            x = x0
        if y is None:
            y = y0
        if self._mdi_wrapper is not None:
            self._mdi_wrapper.move(x, y)
        else:
            QtWidgets.QMainWindow.move(self, x, y)

    @property
    def viewer_size(self):
        if self._mdi_wrapper is not None:
            sz = self._mdi_wrapper.size()
        else:
            sz = self.size()
        return sz.width(), sz.height()

    @viewer_size.setter
    def viewer_size(self, value):
        width, height = value
        self.resize(width, height)
        if self._mdi_wrapper is not None:
            self._mdi_wrapper.resize(width, height)

    def closeEvent(self, event):
        """ Call unregister on window close """

        if not self._confirm_close():
            event.ignore()
            return

        if self._hub is not None:
            self.unregister(self._hub)

        self._layer_artist_container.clear_callbacks()
        self._layer_artist_container.clear()

        super(DataViewer, self).closeEvent(event)
        event.accept()

        self.window_closed.emit()

    def isVisible(self):
        # Override this so as to catch RuntimeError: wrapped C/C++ object of
        # type ... has been deleted
        try:
            return self.isVisible()
        except RuntimeError:
            return False

    def _confirm_close(self):
        """Ask for close confirmation

        :rtype: bool. True if user wishes to close. False otherwise
        """
        if self._warn_close and not os.environ.get('GLUE_TESTING'):
            buttons = QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel
            dialog = QtWidgets.QMessageBox.warning(self, "Confirm Close",
                                                   "Do you want to close this window?",
                                                   buttons=buttons,
                                                   defaultButton=QtWidgets.QMessageBox.Cancel)
            return dialog == QtWidgets.QMessageBox.Ok
        return True

    def layer_view(self):
        return self._view

    def options_widget(self):
        return QtWidgets.QWidget()

    def addToolBar(self, tb):
        super(DataViewer, self).addToolBar(tb)
        self._toolbars.append(tb)
        self._tb_vis[tb] = True

    def initialize_toolbar(self):

        from glue.config import viewer_tool

        self.toolbar = self._toolbar_cls(self, default_mouse_mode_cls=self._default_mouse_mode_cls)

        def get_tools_recursive(cls, tools, subtools):
            if not issubclass(cls, DataViewer):
                return
            if cls._inherit_tools and cls is not DataViewer:
                for parent_cls in cls.__bases__:
                    get_tools_recursive(parent_cls, tools, subtools)
            for tool_id in cls.tools:
                if tool_id not in tools:
                    tools.append(tool_id)
            for tool_id in cls.subtools:
                if tool_id not in subtools:
                    subtools[tool_id] = []
                for subtool_id in cls.subtools[tool_id]:
                    if subtool_id not in subtools[tool_id]:
                        subtools[tool_id].append(subtool_id)

        # Need to include tools and subtools declared by parent classes unless
        # specified otherwise
        tool_ids = []
        subtool_ids = {}
        get_tools_recursive(self.__class__, tool_ids, subtool_ids)

        for tool_id in tool_ids:
            mode_cls = viewer_tool.members[tool_id]
            if tool_id in subtool_ids:
                subtools = []
                for subtool_id in subtool_ids[tool_id]:
                    subtools.append(viewer_tool.members[subtool_id](self))
                mode = mode_cls(self, subtools=subtools)
            else:
                mode = mode_cls(self)
            self.toolbar.add_tool(mode)

        self.addToolBar(self.toolbar)

        self.toolbar_added.emit()

    def show_toolbars(self):
        """Re-enable any toolbars that were hidden with `hide_toolbars()`

        Does not re-enable toolbars that were hidden by other means
        """
        for tb in self._toolbars:
            if self._tb_vis.get(tb, False):
                tb.setEnabled(True)

    def hide_toolbars(self):
        """ Disable all the toolbars in the viewer.

        This action can be reversed by calling `show_toolbars()`
        """
        for tb in self._toolbars:
            self._tb_vis[tb] = self._tb_vis.get(tb, False) or tb.isVisible()
            tb.setEnabled(False)

    def set_focus(self, state):
        if state:
            css = """
            DataViewer
            {
            border: 2px solid;
            border-color: rgb(56, 117, 215);
            }
            """
            self.setStyleSheet(css)
            self.show_toolbars()
        else:
            css = """
            DataViewer
            {
            border: none;
            }
            """
            self.setStyleSheet(css)
            self.hide_toolbars()

    def __str__(self):
        return self.LABEL

    def unregister(self, hub):
        """
        Override to perform cleanup operations when disconnecting from hub
        """
        pass

    @property
    def window_title(self):
        return str(self)

    def update_window_title(self):
        self.setWindowTitle(self.window_title)

    def set_status(self, message):
        sb = self.statusBar()
        sb.showMessage(message)

    def export_as_script(self, filename):

        data_filename = os.path.relpath(filename) + '.data'

        save(data_filename, self.session.data_collection)

        imports = ['from glue.core.state import load']

        imports_header, header = self._script_header()
        imports.extend(imports_header)

        layers = ""
        for ilayer, layer in enumerate(self.layers):
            if layer.layer.label:
                layers += '## Layer {0}: {1}\n\n'.format(ilayer + 1, layer.layer.label)
            else:
                layers += '## Layer {0}\n\n'.format(ilayer + 1)
            if layer.visible and layer.enabled:
                if isinstance(layer.layer, Data):
                    index = self.session.data_collection.index(layer.layer)
                    layers += "layer_data = data_collection[{0}]\n\n".format(index)
                else:
                    dindex = self.session.data_collection.index(layer.layer.data)
                    sindex = layer.layer.data.subsets.index(layer.layer)
                    layers += "layer_data = data_collection[{0}].subsets[{1}]\n\n".format(dindex, sindex)
            imports_layer, layer_script = layer._python_exporter(layer)
            if layer_script is None:
                continue
            imports.extend(imports_layer)
            layers += layer_script.strip() + "\n"

        imports_footer, footer = self._script_footer()
        imports.extend(imports_footer)

        imports = os.linesep.join(sorted(set(imports)))

        script = TEMPLATE_SCRIPT.format(data=data_filename,
                                        imports=imports.strip(),
                                        header=header.strip(),
                                        layers=layers.strip(),
                                        footer=footer.strip())

        with open(filename, 'w') as f:
            f.write(script)

    def _script_header(self):
        raise NotImplementedError()

    def _script_footer(self):
        raise NotImplementedError()
