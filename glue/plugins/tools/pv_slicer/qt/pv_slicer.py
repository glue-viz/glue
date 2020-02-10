import numpy as np

from matplotlib.lines import Line2D

from glue.core import Data
from glue.core.coordinates import coordinates_from_wcs
from glue.core.coordinate_helpers import axis_label
from glue.viewers.matplotlib.toolbar_mode import PathMode
from glue.config import viewer_tool
from glue.viewers.matplotlib.toolbar_mode import ToolbarModeBase
from glue.viewers.image.qt import ImageViewer
from glue.plugins.tools.pv_slicer.pv_sliced_data import PVSlicedData


@viewer_tool
class PVSlicerMode(PathMode):

    icon = 'glue_slice'
    tool_id = 'slice'
    action_text = 'Slice Extraction'
    tool_tip = ('Extract a slice from an arbitrary path\n'
                '  ENTER accepts the path\n'
                '  ESCAPE clears the path')
    status_tip = 'Draw a path then press ENTER to extract slice, or press ESC to cancel'
    shortcut = 'P'

    def __init__(self, viewer, **kwargs):
        super(PVSlicerMode, self).__init__(viewer, **kwargs)
        self._roi_callback = self._extract_callback
        self._slice_widget = None
        self.viewer.state.add_callback('reference_data', self._on_reference_data_change)
        self._on_reference_data_change()

    def _on_reference_data_change(self, *args):
        if self.viewer.state.reference_data is not None:
            self.enabled = self.viewer.state.reference_data.ndim == 3

    def _clear_path(self):
        self.viewer.hide_crosshairs()
        self.clear()

    def _extract_callback(self, mode):
        """
        Extract a PV-like slice, given a path traced on the widget
        """

        vx, vy = mode.roi().to_polygon()

        selected = self.viewer.session.application.selected_layers()

        open_viewer = False

        all_pvdata = []

        for data in self.viewer.state.layers_data:
            if isinstance(data, Data):

                for pvdata in self.viewer.session.data_collection:
                    if isinstance(pvdata, PVSlicedData):
                        if pvdata.original_data is data:
                            break
                else:
                    pvdata = None

            if pvdata is None:
                pvdata = PVSlicedData(data,
                                      self.viewer.state.x_att, vx,
                                      self.viewer.state.y_att, vy,
                                      label=data.label + " [slice]")
                pvdata.parent_viewer = self.viewer
                self.viewer.session.data_collection.append(pvdata)
                open_viewer = True
            else:
                data = pvdata
                data.original_data = self.viewer.state.reference_data
                data.x_att = self.viewer.state.x_att
                data.y_att = self.viewer.state.y_att
                data.set_xy(vx, vy)

            all_pvdata.append(pvdata)

        if open_viewer:
            viewer = self.viewer.session.application.new_data_viewer(ImageViewer)
            for pvdata in all_pvdata:
                viewer.add_data(pvdata)

            viewer.state.aspect = 'auto'
            viewer.state.reset_limits()


@viewer_tool
class PVLinkCursorMode(ToolbarModeBase):
    """
    Selects pixel under mouse cursor.
    """

    icon = "glue_path"
    tool_id = 'pv:crosshair'
    action_text = 'Show position on original path'
    tool_tip = 'Click and drag to show position of cursor on original slice.'
    status_tip = 'Click and drag to show position of cursor on original slice.'

    _pressed = False

    def __init__(self, *args, **kwargs):
        super(PVLinkCursorMode, self).__init__(*args, **kwargs)
        self._move_callback = self._on_move
        self._press_callback = self._on_press
        self._release_callback = self._on_release
        self._active = False
        self.viewer.state.add_callback('reference_data', self._on_reference_data_change)
        self._on_reference_data_change()

    def _on_reference_data_change(self, *args):
        self.enabled = isinstance(self.viewer.state.reference_data, PVSlicedData)
        self.data = self.viewer.state.reference_data

    def activate(self):
        self._line = Line2D(self.data.x, self.data.y, zorder=1000, color='#669dff',
                            alpha=0.6, lw=2)
        self.data.parent_viewer.axes.add_line(self._line)
        self._crosshair = self.data.parent_viewer.axes.plot([], [], '+', ms=12,
                                                            mfc='none', mec='#669dff',
                                                            mew=1, zorder=100)[0]
        self.data.parent_viewer.figure.canvas.draw()
        super().activate()

    def deactivate(self):
        self._line.remove()
        self._crosshair.remove()
        self.data.parent_viewer.figure.canvas.draw()
        super().deactivate()

    def _on_press(self, mode):
        self._active = True

    def _on_release(self, mode):
        self._active = False

    def _on_move(self, mode):

        if not self._active:
            return

        # Find position of click in the image viewer
        xdata, ydata = self._event_xdata, self._event_ydata

        if xdata is None or ydata is None:
            return

        # TODO: Make this robust in case the axes have been swapped

        # Find position slice where cursor is
        ind = int(round(np.clip(xdata, 0, self.data.shape[1] - 1)))

        # Find pixel coordinate in input image for this slice
        x = self.data.x[ind]
        y = self.data.y[ind]

        # The 3-rd coordinate in the input WCS is simply the second
        # coordinate in the PV slice.
        z = ydata

        self._crosshair.set_xdata([x])
        self._crosshair.set_ydata([y])
        self.data.parent_viewer.figure.canvas.draw()

        s = list(self.data.parent_viewer.state.wcsaxes_slice[::-1])
        s[_slice_index(self.data.parent_viewer.state.reference_data, s)] = int(z)
        self.data.parent_viewer.state.slices = tuple(s)


def _slice_index(data, slc):
    """
    The axis over which to extract PV slices
    """
    for i in range(len(slc)):
        if np.isreal(slc[i]):
            return i
    raise ValueError("Could not find slice index with slc={0}".format(slc))


def _slice_label(data, slc):
    """
    Returns a formatted axis label corresponding to the slice dimension
    in a PV slice

    :param data: Data that slice is extracted from
    :param slc: orientation in the image widget from which the PV slice
                was defined
    """
    idx = _slice_index(data, slc)
    if getattr(data, 'coords') is None:
        return data.pixel_component_ids[idx].label
    else:
        return axis_label(data.coords, idx)
