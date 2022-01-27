import numpy as np

from glue.viewers.matplotlib.toolbar_mode import PathMode
from glue.viewers.image.qt import StandaloneImageViewer
from glue.config import viewer_tool
from glue.utils import defer_draw
from glue.core.coordinate_helpers import axis_label


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
        self._build_from_vertices(vx, vy)

    def _build_from_vertices(self, vx, vy):
        pv_slice, x, y, wcs = _slice_from_path(vx, vy, self.viewer.state.reference_data,
                                               self.viewer.state.layers[0].attribute,
                                               self.viewer.state.wcsaxes_slice[::-1])
        if self._slice_widget is None:
            self._slice_widget = PVSliceWidget(image=pv_slice, wcs=wcs,
                                               image_viewer=self.viewer,
                                               x=x, y=y, interpolation='nearest')
            self.viewer._session.application.add_widget(self._slice_widget,
                                                        label='Custom Slice')
            self._slice_widget.window_closed.connect(self._clear_path)
        else:
            self._slice_widget.set_image(image=pv_slice, wcs=wcs,
                                         x=x, y=y, interpolation='nearest')

        result = self._slice_widget
        result.axes.set_xlabel("Position along path")
        if wcs is None:
            result.axes.set_ylabel("Cube slice index")
        else:
            result.axes.set_ylabel(_slice_label(self.viewer.state.reference_data,
                                                self.viewer.state.wcsaxes_slice[::-1]))

        result.show()

    def close(self):
        if self._slice_widget:
            self._slice_widget.close()
        return super(PVSlicerMode, self).close()


class PVSliceWidget(StandaloneImageViewer):

    """ A standalone image widget with extra interactivity for PV slices """

    def __init__(self, image=None, wcs=None, image_viewer=None,
                 x=None, y=None, **kwargs):
        """
        :param image: 2D Numpy array representing the PV Slice
        :param wcs: WCS for the PV slice
        :param image_viewer: Parent ImageViewer this was extracted from
        :param kwargs: Extra keywords are passed to imshow
        """
        self._crosshairs = None
        self._parent = image_viewer
        super(PVSliceWidget, self).__init__(image=image, wcs=wcs, **kwargs)
        conn = self.axes.figure.canvas.mpl_connect
        self._down_id = conn('button_press_event', self._on_click)
        self._move_id = conn('motion_notify_event', self._on_move)
        self.axes.format_coord = self._format_coord
        self._x = x
        self._y = y
        self._parent.state.add_callback('x_att', self.reset)
        self._parent.state.add_callback('y_att', self.reset)

    def _format_coord(self, x, y):
        """
        Return a formatted location label for the taskbar

        :param x: x pixel location in slice array
        :param y: y pixel location in slice array
        """

        # xy -> xyz in image view
        pix = self._pos_in_parent(xdata=x, ydata=y)

        # xyz -> data pixel coords
        # accounts for fact that image might be shown transposed/rotated
        s = list(self._slc)
        idx = _slice_index(self._parent.state.reference_data, self._slc)
        s[s.index('x')] = pix[0]
        s[s.index('y')] = pix[1]
        s[idx] = pix[2]

        # labels = self._parent.coordinate_labels(s)
        # return '         '.join(labels)
        return ''

    def set_image(self, image=None, wcs=None, x=None, y=None, **kwargs):
        super(PVSliceWidget, self).set_image(image=image, wcs=wcs, **kwargs)
        self._axes.set_aspect('auto')
        self._axes.set_xlim(-0.5, image.shape[1] - 0.5)
        self._axes.set_ylim(-0.5, image.shape[0] - 0.5)
        self._slc = self._parent.state.wcsaxes_slice[::-1]
        self._x = x
        self._y = y

    @defer_draw
    def _sync_slice(self, event):
        s = list(self._slc)
        # XXX breaks if display_data changes
        _, _, z = self._pos_in_parent(event)
        s[_slice_index(self._parent.state.reference_data, s)] = int(z)
        self._parent.state.slices = tuple(s)

    @defer_draw
    def _draw_crosshairs(self, event):
        x, y, _ = self._pos_in_parent(event)
        self._parent.show_crosshairs(x, y)

    @defer_draw
    def _on_move(self, event):
        if not event.button:
            return

        if not event.inaxes or event.canvas.toolbar.mode != '':
            return

        self._sync_slice(event)
        self._draw_crosshairs(event)

    def _pos_in_parent(self, event=None, xdata=None, ydata=None):

        if event is not None:
            xdata = event.xdata
            ydata = event.ydata

        # Find position slice where cursor is
        ind = int(round(np.clip(xdata, 0, self._im_array.shape[1] - 1)))

        # Find pixel coordinate in input image for this slice
        x = self._x[ind]
        y = self._y[ind]

        # The 3-rd coordinate in the input WCS is simply the second
        # coordinate in the PV slice.
        z = ydata

        return x, y, z

    def _on_click(self, event):
        if not event.inaxes or event.canvas.toolbar.mode != '':
            return
        self._sync_slice(event)
        self._draw_crosshairs(event)

    def reset(self, *args):
        self.close()


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
