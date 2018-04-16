import numpy as np

from glue.core import Data
from glue.core.coordinates import coordinates_from_wcs
from glue.core.coordinate_helpers import axis_label
from glue.viewers.common.qt.toolbar_mode import PathMode
from glue.config import viewer_tool
from glue.viewers.common.qt.toolbar_mode import ToolbarModeBase
from glue.viewers.image.qt import ImageViewer


class PVSliceData(Data):
    parent_data = None
    parent_data_x = None
    parent_data_y = None
    parent_viewer = None


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

    def _on_reference_data_change(self, reference_data):
        print(reference_data, reference_data.ndim)
        if reference_data is not None:
            self.enabled = reference_data.ndim == 3

    def _clear_path(self):
        self.viewer.hide_crosshairs()
        self.clear()

    def _extract_callback(self, mode):
        """
        Extract a PV-like slice, given a path traced on the widget
        """

        vx, vy = mode.roi().to_polygon()

        pv_slice, x, y, wcs = _slice_from_path(vx, vy, self.viewer.state.reference_data,
                                               self.viewer.state.layers[0].attribute,
                                               self.viewer.state.wcsaxes_slice[::-1])

        xlabel = "Position along path"
        if wcs is None:
            ylabel = "Cube slice index"
        else:
            ylabel = _slice_label(self.viewer.state.reference_data,
                                  self.viewer.state.wcsaxes_slice[::-1])

        wcs.wcs.ctype = [xlabel, ylabel]

        data = PVSliceData(label=self.viewer.state.reference_data.label + " [slice]")
        data.coords = coordinates_from_wcs(wcs)
        data[self.viewer.state.layers[0].attribute] = pv_slice

        # TODO: use weak references
        data.parent_data = self.viewer.state.reference_data
        data.parent_data_x = x
        data.parent_data_y = y
        data.parent_viewer = self.viewer

        selected = self.viewer.session.application.selected_layers()

        if len(selected) == 1 and isinstance(selected[0], PVSliceData):
            selected[0].update_values_from_data(data)
            data = selected[0]
            for tab in self.viewer.session.application.viewers:
                for viewer in tab:
                    if data in viewer._layer_artist_container:
                        open_viewer = False
                        break
                if not open_viewer:
                    break
            else:
                open_viewer = True
        else:
            self.viewer.session.data_collection.append(data)
            open_viewer = True

        print("OPEN VIEWER", open_viewer)
        if open_viewer:
            self.viewer.session.application.new_data_viewer(ImageViewer, data=data)


@viewer_tool
class PVLinkCursorMode(ToolbarModeBase):
    """
    Selects pixel under mouse cursor.
    """

    icon = "glue_point"
    tool_id = 'pv:crosshair'

    _pressed = False

    def __init__(self, *args, **kwargs):
        super(PVLinkCursorMode, self).__init__(*args, **kwargs)
        self._move_callback = self._on_move
        self._press_callback = self._on_move
        self.viewer.state.add_callback('reference_data', self._on_reference_data_change)

    def _on_reference_data_change(self, reference_data):
        self.enabled = isinstance(reference_data, PVSliceData)
        self.data = reference_data

    def _on_move(self, mode):

        # Find position of click in the image viewer
        xdata, ydata = self._event_xdata, self._event_ydata

        # TODO: Make this robust in case the axes have been swapped

        # Find position slice where cursor is
        ind = int(round(np.clip(xdata, 0, self.data.shape[1] - 1)))

        # Find pixel coordinate in input image for this slice
        x = self.data.parent_data_x[ind]
        y = self.data.parent_data_y[ind]

        # The 3-rd coordinate in the input WCS is simply the second
        # coordinate in the PV slice.
        z = ydata

        self.data.parent_viewer.show_crosshairs(x, y)

        s = list(self.data.parent_viewer.state.wcsaxes_slice[::-1])
        s[_slice_index(self.data.parent_viewer.state.reference_data, s)] = int(z)
        self.data.parent_viewer.state.slices = tuple(s)


def _slice_from_path(x, y, data, attribute, slc):
    """
    Extract a PV-like slice from a cube

    :param x: An array of x values to extract (pixel units)
    :param y: An array of y values to extract (pixel units)
    :param data: :class:`~glue.core.data.Data`
    :param attribute: :claass:`~glue.core.data.Component`
    :param slc: orientation of the image widget that `pts` are defined on

    :returns: (slice, x, y)
              slice is a 2D Numpy array, corresponding to a "PV ribbon"
              cutout from the cube
              x and y are the resampled points along which the
              ribbon is extracted

    :note: For >3D cubes, the "V-axis" of the PV slice is the longest
           cube axis ignoring the x/y axes of `slc`
    """
    from pvextractor import Path, extract_pv_slice
    p = Path(list(zip(x, y)))

    cube = data[attribute]
    dims = list(range(data.ndim))
    s = list(slc)
    ind = _slice_index(data, slc)

    from astropy.wcs import WCS

    if isinstance(data.coords, WCS):
        cube_wcs = data.coords
    else:
        cube_wcs = None

    # transpose cube to (z, y, x, <whatever>)
    def _swap(x, s, i, j):
        x[i], x[j] = x[j], x[i]
        s[i], s[j] = s[j], s[i]

    _swap(dims, s, ind, 0)
    _swap(dims, s, s.index('y'), 1)
    _swap(dims, s, s.index('x'), 2)

    cube = cube.transpose(dims)

    if cube_wcs is not None:
        cube_wcs = cube_wcs.sub([data.ndim - nx for nx in dims[::-1]])

    # slice down from >3D to 3D if needed
    s = tuple([slice(None)] * 3 + [slc[d] for d in dims[3:]])
    cube = cube[s]

    # sample cube
    spacing = 1  # pixel
    x, y = [np.round(_x).astype(int) for _x in p.sample_points(spacing)]

    try:
        result = extract_pv_slice(cube, path=p, wcs=cube_wcs, order=0)
        wcs = WCS(result.header)
    except Exception:  # sometimes pvextractor complains due to wcs. Try to recover
        result = extract_pv_slice(cube, path=p, wcs=None, order=0)
        wcs = None

    data = result.data

    return data, x, y, wcs


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
