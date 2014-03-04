import numpy as np
from matplotlib.transforms import blended_transform_factory

from ..core.callback_property import CallbackProperty, add_callback


PICK_THRESH = 30  # pixel distance threshold for picking


class Handle(object):
    def __init__(self, viewer):
        self.viewer = viewer

    def remove(self):
        raise NotImplementedError()

    def pick_dist(self, x, y):
        """
        Return the distance, in pixels,
        between a point in (x,y) data space and
        the handle
        """
        raise NotImplementedError()

    def select(self, x, y):
        """
        Process a selection event (click) at x,y
        """
        raise NotImplementedError()

    def drag(self, x, y):
        """
        Process a drag to x, y
        """
        raise NotImplementedError()

    def release(self):
        """
        Process a release
        """
        raise NotImplementedError()


class SliderHandle(Handle):
    value = CallbackProperty(None)

    def pick_dist(self, x, y):
        xy = [[x, y], [self.value, y]]
        xypix = self.viewer.axes.transData.transform(xy)
        return abs(xypix[1, 0] - xypix[0, 0])

    def select(self, x, y):
        pass

    def drag(self, x, y):
        self.value = x

    def release(self):
        pass


class RangeHandle(Handle):
    range = CallbackProperty((None, None))

    def __init__(self, viewer):
        super(RangeHandle, self).__init__(viewer)

        # track state during drags
        self._move = None
        self._ref = None
        self._refx = None

    def pick_dist(self, x, y):
        xy = np.array([[x, y],
                       [self.range[0], y],
                       [self.range[1], y],
                       [sum(self.range)/2, y]])
        xypix = self.viewer.axes.transData.transform(xy)
        dx = np.abs(xypix[1:] - xypix[0])[:, 0]
        return min(dx)

    def select(self, x, y):
        cen = sum(self.range) / 2.
        wid = self.range[1] - self.range[0]
        if x < cen - wid / 4.:
            self._move = 'left'
        elif x < cen + wid / 4.:
            self._move = 'center'
            self._ref = self.range
            self._refx = x
        else:
            self._move = 'right'

    def drag(self, x, y):
        if self._move == 'left':
            if x > self.range[1]:
                self._move = 'right'
            self._set_range(x, self.range[1])

        elif self._move == 'center':
            dx = (x - self._refx)
            self._set_range(self._ref[0] + dx, self._ref[1] + dx)
        else:
            if x < self.range[0]:
                self._move = 'left'
            self._set_range(self.range[0], x)

    def _set_range(self, lo, hi):
        self.range = min(lo, hi), max(lo, hi)

    def release(self):
        self._move = None
        self._ref = None
        self._refx = None


class SliderArtist(object):

    def __init__(self, slider, **kwargs):
        self.slider = slider
        add_callback(slider, 'value', self._update)
        ax = self.slider.viewer.axes

        kwargs.setdefault('lw', 2)
        kwargs.setdefault('alpha', 0.5)
        kwargs.setdefault('c', 'k')
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        self._line, = ax.plot([slider.value, slider.value], [0, 1],
                              transform=trans, **kwargs)

    def _update(self, value):
        self._line.set_xdata([value, value])
        self._line.axes.figure.canvas.draw()

class RangeArtist(object):

    def __init__(self, handle, **kwargs):
        self.handle = handle
        add_callback(handle, 'range', self._update)
        ax = handle.viewer.axes
        trans = blended_transform_factory(ax.transData, ax.transAxes)

        kwargs.setdefault('lw', 2)
        kwargs.setdefault('alpha', 0.5)
        kwargs.setdefault('c', 'r')
        self._line, = ax.plot(self.x, self.y, transform=trans, **kwargs)

    @property
    def x(self):
        l, r = self.handle.range
        return [l, l, l, r, r, r]

    @property
    def y(self):
        return [0, 1, .5, .5, 0, 1]

    def _update(self, rng):
        lo, hi = rng
        self._line.set_xdata([lo, lo, lo, hi, hi, hi])
        self._line.axes.figure.canvas.draw()


class ProfileViewer(object):
    slider_cls = SliderHandle
    range_cls = RangeHandle

    def __init__(self, axes):
        self.axes = axes
        self.handles = []

        self._artist = None
        self._x = self._xatt = self._y = self._yatt = None
        self._attach_events()
        self._selected_handle = None

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

    def _attach_events(self):
        connect = self.axes.figure.canvas.mpl_connect
        self._down_id = connect('button_press_event', self._on_down)
        self._up_id = connect('button_release_event', self._on_up)
        self._move_id = connect('motion_notify_event', self._on_move)

    def _on_down(self, event):
        handle = self.pick_handle(event.xdata, event.ydata)
        if handle is not None:
            self._selected_handle = handle
            handle.select(event.xdata, event.ydata)

    def _on_up(self, event):
        if self._selected_handle is not None:
            self._selected_handle.release()
        self._selected_handle = None

    def _on_move(self, event):
        if not event.inaxes:
            return

        if self._selected_handle is None:
            return
        self._selected_handle.drag(event.xdata, event.ydata)

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
        result.value = self._center[0]

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
        center = self._center[0]
        width = self._width
        result.range = center - width / 4, center + width / 4

        if callback is not None:
            add_callback(result, 'range', callback)

        self.handles.append(result)
        return result

    @property
    def _center(self):
        """Return the data coordinates of the axes center, as (x, y)"""
        xy = self.axes.transAxes.transform([.5, .5])
        xy = self.axes.transData.inverted().transform(xy)
        return tuple(xy.ravel())

    @property
    def _width(self):
        """Return the X-width of axes in data units"""
        xlim = self.axes.get_xlim()
        return xlim[1] - xlim[0]

    def pick_handle(self, x, y):
        """
        Given a coordinate in Data units,
        return the Handle object nearest
        that point, or None if none are nearby
        """
        dist, handle = min((h.pick_dist(x, y), h) for h in self.handles)
        if dist < PICK_THRESH:
            return handle


def main():
    from glue.qt.widgets.mpl_widget import MplWidget
    from glue.external.qt.QtGui import QApplication

    w = MplWidget()
    ax = w.canvas.fig.add_subplot(111)
    pv = ProfileViewer(ax)

    x = np.linspace(0, 10, 100)
    y = np.exp(-(x - 5) ** 2) + np.random.normal(0, .1, 100)
    pv.set_profile(x, y)

    handle = pv.new_slider_handle()
    SliderArtist(handle)

    handle = pv.new_range_handle()
    RangeArtist(handle)

    w.show()
    w.raise_()
    QApplication.instance().exec_()

if __name__ == "__main__":
    main()
