import numpy as np

from ..core.callback_property import CallbackProperty, add_callback

PICK_THRESH = 10  # pixel distance threshold for picking


class Handle(object):
    def __init__(self, viewer):
        self._viewer = viewer

    def remove(self):
        raise NotImplementedError()

    def pick_dist(self, x, y):
        """
        Return the distance, in pixels,
        between a point in (x,y) data space and
        the handle
        """
        raise NotImplementedError()


class SliderHandle(Handle):
    value = CallbackProperty(None)

    def pick_dist(self, x, y):
        xy = [[x, y], [self.value, y]]
        xypix = self._viewer.axes.transData.transform(xy)
        return abs(xypix[1, 0] - xypix[0, 0])


class RangeHandle(Handle):
    range = CallbackProperty((None, None))

    def pick_dist(self, x, y):
        xy = [[x, y], [self.range[0], y], [self.range[1], y]]
        xypix = self._viewer.axes.transData.transform(xy)
        dx = np.abs(xypix[1:] - xypix[0])[:, 0]
        return min(dx)


class ProfileViewer(object):
    slider_cls = SliderHandle
    range_cls = RangeHandle

    def __init__(self, axes):
        self.axes = axes
        self.handles = []

        self._artist = None
        self._x = self._xatt = self._y = self._yatt = None


    def set_profile(self, x, y, xatt=None, yatt=None, **kwargs):
        """
        Set a new line profile

        :param x: X-coordinate data
        :type x: array-like

        :param y: Y-coordinate data
        :type y: array-like

        :param xatt: ComponentID associated with X axis
        :type xatt: :class:`~glue.core.data.CompoenntID`

        :param yatt: ComponentID associated with Y axis
        :type yatt: :class:`~glue.core.data.CompoenntID`

        Extra kwargs are passed to matplotlib.plot, to
        customize plotting

        Returns the created MPL artist
        """
        self._x = x
        self._xatt = xatt
        self._y = y
        self._yatt = yatt
        if self._artist is not None:
            self._artist.remove()
        self._artist = self.axes.plot(x, y, **kwargs)[0]
        self._redraw()

        return self._artist

    def _redraw(self):
        self.axes.figure.canvas.draw()

    def fit(self, fit_function, xlim=None, plot=True):
        pass

    def new_slider_handle(self, callback=None):
        """
        Create and return new SliderHandle

        :param callback: A callback function to be invoked
        whenever the handle.value property changes
        """
        result = self.slider_cls(self)
        if callback is not None:
            add_callback(result, 'value', callback)
        self.handles.append(result)
        return result

    def new_range_handle(self, callback=None):
        """
        Create and return new RangeHandle

        :param callback: A callback function to be invoked
        whenever the handle.range property changes
        """
        result = self.range_cls(self)
        if callback is not None:
            add_callback(result, 'range', callback)

        self.handles.append(result)
        return result

    def pick_handle(self, x, y):
        """
        Given a coordinate in Data units,
        return the Handle object nearest
        that point, or None if none are nearby
        """
        dist, handle = min((h.pick_dist(x, y), h) for h in self.handles)
        if dist < PICK_THRESH:
            return handle
