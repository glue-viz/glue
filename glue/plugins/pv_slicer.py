import numpy as np
from ..qt.mouse_mode import PathMode
from ..qt.widgets.image_widget import StandaloneImageWidget
from ..qt.widgets.mpl_widget import defer_draw



class PVSlicerTool(object):

    def __init__(self, widget=None):
        self.widget = widget

    def _get_modes(self, axes):
        self._path = PathMode(axes, roi_callback=self._slice)
        return [self._path]

    def _slice(self, mode):
        self._extract_pv_slice(self._path, self.widget, mode.roi())

    def _display_data_hook(self, data):
        if data is not None:
            self._path.enabled = data.ndim > 2


    def _extract_pv_slice(self, path, widget, roi):
        """
        Extract a PV-like slice, given a path traced on the widget
        """
        vx, vy = roi.to_polygon()
        pv_slice, wcs = _slice_from_path(vx, vy, widget.data, widget.attribute, widget.slice)
        print("HERE", self._slice_widget)
        if self._slice_widget is None:
            self._slice_widget = PVSliceWidget(image=pv_slice, wcs=wcs, image_widget=widget,
                                               interpolation='nearest')
            widget._session.application.add_widget(self._slice_widget,
                                                 label='Custom Slice')
            self._slice_widget.window_closed.connect(path.clear)
        else:
            self._slice_widget.set_image(image=pv_slice, wcs=wcs, interpolation='nearest')

        result = self._slice_widget
        result.axes.set_xlabel("Position Along Slice")
        result.axes.set_ylabel(_slice_label(widget.data, widget.slice))

        result.show()


class PVSliceWidget(StandaloneImageWidget):

    """ A standalone image widget with extra interactivity for PV slices """

    def __init__(self, image=None, wcs=None, image_widget=None, **kwargs):
        """
        :param image: 2D Numpy array representing the PV Slice
        :param wcs: WCS for the PV slice
        :param image_widget: Parent widget this was extracted from
        :param kwargs: Extra keywords are passed to imshow
        """
        self._crosshairs = None
        self._parent = image_widget
        super(PVSliceWidget, self).__init__(image=image, wcs=wcs, **kwargs)
        conn = self.axes.figure.canvas.mpl_connect
        self._down_id = conn('button_press_event', self._on_click)
        self._move_id = conn('motion_notify_event', self._on_move)
        self.axes.format_coord = self._format_coord

    def _format_coord(self, x, y):
        """
        Return a formatted location label for the taskbar

        :param x: x pixel location in slice array
        :param y: y pixel location in slice array
        """

        # xy -> xyz in image view
        pix = self._pos_in_parent(xdata=x, ydata=y)

        # xyz -> data pixel coords
        # accounts for fact that image might be shown transpoed/rotated
        s = list(self._slc)
        idx = _slice_index(self._parent.data, self._slc)
        s[s.index('x')] = pix[0]
        s[s.index('y')] = pix[1]
        s[idx] = pix[2]

        labels = self._parent.client.coordinate_labels(s)
        return '         '.join(labels)

    def set_image(self, image=None, wcs=None, **kwargs):
        super(PVSliceWidget, self).set_image(image=image, wcs=wcs, **kwargs)
        self._axes.set_aspect('auto')
        self._axes.set_xlim(-0.5, image.shape[1]-0.5)
        self._axes.set_ylim(-0.5, image.shape[0]-0.5)
        self._slc = self._parent.slice

    @defer_draw
    def _sync_slice(self, event):
        s = list(self._slc)
        # XXX breaks if display_data changes
        _, _, z = self._pos_in_parent(event)
        s[_slice_index(self._parent.data, s)] = z
        self._parent.slice = tuple(s)

    @defer_draw
    def _draw_crosshairs(self, event):
        if self._crosshairs is not None:
            self._crosshairs.remove()

        x, y, _ = self._pos_in_parent(event)
        ax = self._parent.client.axes
        self._crosshairs, = ax.plot([x], [y], '+', ms=12,
                                    mfc='none', mec='#de2d26',
                                    mew=2, zorder=100)
        ax.figure.canvas.draw()

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
        return xdata, ydata, ydata

    def _on_click(self, event):
        if not event.inaxes or event.canvas.toolbar.mode != '':
            return
        self._sync_slice(event)
        self._draw_crosshairs(event)


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
    from ..external.pvextractor import Path, extract_pv_slice
    p = Path(list(zip(x, y)))

    cube = data[attribute]
    dims = list(range(data.ndim))
    s = list(slc)
    ind = _slice_index(data, slc)

    cube_wcs = getattr(data.coords, 'wcs', None)

    # transpose cube to (z, y, x, <whatever>)
    def _swap(x, s, i, j):
        x[i], x[j] = x[j], x[i]
        s[i], s[j] = s[j], s[i]

    _swap(dims, s, ind, 0)
    _swap(dims, s, s.index('y'), 1)
    _swap(dims, s, s.index('x'), 2)
    cube = cube.transpose(dims)

    # slice down from >3D to 3D if needed
    s = [slice(None)] * 3 + [slc[d] for d in dims[3:]]
    cube = cube[s]

    # sample cube
    spacing = 1  # pixel
    x, y = [np.round(_x).astype(int) for _x in p.sample_points(spacing)]
    result = extract_pv_slice(cube, path=p, wcs=cube_wcs, order=0)

    from astropy.wcs import WCS
    data = result.data
    wcs = WCS(result.header)

    return data, wcs


def _slice_index(data, slc):
    """
    The axis over which to extract PV slices
    """
    return max([i for i in range(len(slc))
               if isinstance(slc[i], int)],
               key=lambda x: data.shape[x])

def _slice_label(data, slc):
    """
    Returns a formatted axis label corresponding to the slice dimension
    in a PV slice

    :param data: Data that slice is extracted from
    :param slc: orientation in the image widget from which the PV slice
                was defined
    """
    idx = _slice_index(data, slc)
    return data.get_world_component_id(idx).label
