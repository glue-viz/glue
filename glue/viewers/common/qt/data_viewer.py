from __future__ import absolute_import, division, print_function

from qtpy.QtCore import Qt
from qtpy import QtWidgets
from glue.core.qt.layer_artist_model import QtLayerArtistContainer, LayerArtistWidget
from glue.utils.qt import set_cursor
from glue.external import six
from glue.core.qt.dialogs import warn
from glue.utils.noconflict import classmaker
from glue.config import viewer_tool
from glue.viewers.common.qt.base_widget import BaseQtViewerWidget
from glue.viewers.common.qt.tool import SimpleToolMenu
from glue.viewers.common.qt.toolbar import BasicToolbar
from glue.viewers.common.viewer import Viewer

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
class DataViewer(Viewer, BaseQtViewerWidget):
    """
    Base class for all Qt DataViewer widgets.

    This defines a minimal interface, and implements the following::

       * An automatic call to unregister on window close
       * Drag and drop support for adding data
    """

    _layer_artist_container_cls = QtLayerArtistContainer
    _layer_style_widget_cls = None

    _toolbar_cls = BasicToolbar

    # This defines the mouse mode to be used when no toolbar modes are active
    _default_mouse_mode_cls = None

    _inherit_tools = True
    tools = ['save']
    subtools = {'save': []}

    _close_on_last_layer_removed = True

    _options_cls = None

    large_data_size = None

    def __init__(self, session, state=None, parent=None):
        """
        :type session: :class:`~glue.core.Session`
        """

        BaseQtViewerWidget.__init__(self, parent)
        Viewer.__init__(self, session, state=state)

        self._view = LayerArtistWidget(layer_style_widget_cls=self._layer_style_widget_cls,
                                       hub=session.hub)
        self._view.layer_list.setModel(self._layer_artist_container.model)

        # Set up the options widget, which will include options that control the
        # viewer state
        if self._options_cls is None:
            self.options = QtWidgets.QWidget()
        else:
            self.options = self._options_cls(viewer_state=self.state,
                                             session=session)

        self._tb_vis = {}  # store whether toolbars are enabled
        self.toolbar = None
        self._toolbars = []
        self._warn_close = True

        # close window when last plot layer deleted
        if self._close_on_last_layer_removed:
            self._layer_artist_container.on_empty(self._close_nowarn)
        self._layer_artist_container.on_changed(self.update_window_title)

    @property
    def selected_layer(self):
        return self._view.layer_list.current_artist()

    @set_cursor(Qt.WaitCursor)
    def apply_roi(self, roi):
        pass

    def warn(self, message, *args, **kwargs):
        return warn(message, *args, **kwargs)

    def _close_nowarn(self):
        return self.close(warn=False)

    def closeEvent(self, event):
        super(DataViewer, self).closeEvent(event)
        Viewer.cleanup(self)
        # We tell the toolbar to do cleanup to make sure we get rid of any
        # circular references
        if self.toolbar:
            self.toolbar.cleanup()

    def layer_view(self):
        return self._view

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
        """
        Re-enable any toolbars that were hidden with `hide_toolbars()`

        Does not re-enable toolbars that were hidden by other means
        """
        for tb in self._toolbars:
            if self._tb_vis.get(tb, False):
                tb.setEnabled(True)

    def hide_toolbars(self):
        """
        Disable all the toolbars in the viewer.

        This action can be reversed by calling `show_toolbars()`
        """
        for tb in self._toolbars:
            self._tb_vis[tb] = self._tb_vis.get(tb, False) or tb.isVisible()
            tb.setEnabled(False)

    def set_focus(self, state):
        super(DataViewer, self).set_focus(state)
        if state:
            self.show_toolbars()
        else:
            self.hide_toolbars()

    def __gluestate__(self, context):
        state = Viewer.__gluestate__(self, context)
        state['size'] = self.viewer_size
        state['pos'] = self.position
        state['_protocol'] = 1
        return state

    def update_viewer_state(rec, context):
        pass

    @classmethod
    def __setgluestate__(cls, rec, context):

        if rec.get('_protocol', 0) < 1:
            cls.update_viewer_state(rec, context)

        viewer = super(DataViewer, cls).__setgluestate__(rec, context)

        viewer.viewer_size = rec['size']
        x, y = rec['pos']
        viewer.move(x=x, y=y)

        return viewer
