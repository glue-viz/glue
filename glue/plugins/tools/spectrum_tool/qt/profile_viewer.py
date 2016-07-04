from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib.transforms import blended_transform_factory

from glue.core.callback_property import CallbackProperty, add_callback


PICK_THRESH = 30  # pixel distance threshold for picking


class Grip(object):

    def __init__(self, viewer, artist=True):
        self.viewer = viewer
        self.enabled = True

        self.artist = None
        if artist:
            self.artist = self._artist_factory()

    def remove(self):
        raise NotImplementedError()

    def _artist_factory(self):
        raise NotImplementedError()

    def pick_dist(self, x, y):
        """
        Return the distance, in pixels,
        between a point in (x,y) data space and
        the grip
        """
        raise NotImplementedError()

    def dblclick(self, x, y):
        """Respond to a double-click event

        Default is to ignore
        """
        pass

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

    def disable(self):
        self.enabled = False
        if self.artist is not None:
            self.artist.set_visible(False)
            self.viewer.axes.figure.canvas.draw()

    def enable(self):
        self.enabled = True
        if self.artist is not None:
            self.artist.set_visible(True)
            self.viewer.axes.figure.canvas.draw()


class ValueGrip(Grip):
    value = CallbackProperty(None)

    def __init__(self, viewer, artist=True):
        super(ValueGrip, self).__init__(viewer, artist)
        self._drag = False

    def _artist_factory(self):
        return ValueArtist(self)

    def dblclick(self, x, y):
        self.value = x

    def pick_dist(self, x, y):
        xy = [[x, y], [self.value, y]]
        xypix = self.viewer.axes.transData.transform(xy)
        return abs(xypix[1, 0] - xypix[0, 0])

    def select(self, x, y):
        if self.pick_dist(x, y) > PICK_THRESH:
            return
        self._drag = True

    def drag(self, x, y):
        if self._drag:
            self.value = x

    def release(self):
        self._drag = False


class RangeGrip(Grip):
    range = CallbackProperty((None, None))

    def __init__(self, viewer):
        super(RangeGrip, self).__init__(viewer)

        # track state during drags
        self._move = None
        self._ref = None
        self._refx = None
        self._refnew = None

    def _artist_factory(self):
        return RangeArtist(self)

    def pick_dist(self, x, y):
        xy = np.array([[x, y],
                       [self.range[0], y],
                       [self.range[1], y],
                       [sum(self.range) / 2, y]])
        xypix = self.viewer.axes.transData.transform(xy)
        dx = np.abs(xypix[1:] - xypix[0])[:, 0]
        return min(dx)

    def select(self, x, y):
        if self.pick_dist(x, y) > PICK_THRESH:
            return self.new_select(x, y)

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

    def new_select(self, x, y):
        """
        Begin a selection in "new range" mode.
        In this mode, the previous grip position is ignored,
        and the new range is defined by the select/release positions
        """
        self._refnew = x
        self.range = (x, x)

    def new_drag(self, x, y):
        """
        Drag the selection in "new mode"
        """
        if self._refnew is not None:
            self._set_range(self._refnew, x)

    def drag(self, x, y):
        if self._refnew is not None:
            return self.new_drag(x, y)

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
        self._refnew = None


class ValueArtist(object):

    def __init__(self, grip, **kwargs):
        self.grip = grip
        add_callback(grip, 'value', self._update)
        ax = self.grip.viewer.axes

        kwargs.setdefault('lw', 2)
        kwargs.setdefault('alpha', 0.5)
        kwargs.setdefault('c', '#ffb304')
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        self._line, = ax.plot([grip.value, grip.value], [0, 1],
                              transform=trans, **kwargs)

    def _update(self, value):
        self._line.set_xdata([value, value])
        self._line.axes.figure.canvas.draw()

    def set_visible(self, visible):
        self._line.set_visible(visible)


class RangeArtist(object):

    def __init__(self, grip, **kwargs):
        self.grip = grip
        add_callback(grip, 'range', self._update)
        ax = grip.viewer.axes
        trans = blended_transform_factory(ax.transData, ax.transAxes)

        kwargs.setdefault('lw', 2)
        kwargs.setdefault('alpha', 0.5)
        kwargs.setdefault('c', '#ffb304')
        self._line, = ax.plot(self.x, self.y, transform=trans, **kwargs)

    @property
    def x(self):
        l, r = self.grip.range
        return [l, l, l, r, r, r]

    @property
    def y(self):
        return [0, 1, .5, .5, 0, 1]

    def _update(self, rng):
        self._line.set_xdata(self.x)
        self._line.axes.figure.canvas.draw()

    def set_visible(self, visible):
        self._line.set_visible(visible)


def _build_axes(figure):

    ax2 = figure.add_subplot(122)
    ax1 = figure.add_subplot(121, sharex=ax2)

    ax1.xaxis.get_major_formatter().set_useOffset(False)
    ax1.yaxis.get_major_formatter().set_useOffset(False)
    ax2.xaxis.get_major_formatter().set_useOffset(False)
    ax2.yaxis.get_major_formatter().set_useOffset(False)

    return ax1, ax2


class ProfileViewer(object):
    value_cls = ValueGrip
    range_cls = RangeGrip

    def __init__(self, figure):
        self.axes, self.resid_axes = _build_axes(figure)

        self._artist = None
        self._resid_artist = None
        self._x = self._xatt = self._y = self._yatt = None
        self._resid = None
        self.connect()

        self._fit_artists = []
        self.active_grip = None  # which grip should receive events?
        self.grips = []
        self._xlabel = ''

    def set_xlabel(self, xlabel):
        self._xlabel = xlabel

    def autoscale_ylim(self):
        x, y = self._x, self._y
        xlim = self.axes.get_xlim()
        mask = (xlim[0] <= x) & (x <= xlim[1])
        ymask = y[mask]
        if ymask.size == 0:
            return

        ylim = np.nan_to_num(np.array([np.nanmin(ymask), np.nanmax(ymask)]))
        self.axes.set_ylim(ylim[0], ylim[1] + .05 * (ylim[1] - ylim[0]))

        if self._resid is None:
            return
        assert self._resid.size == y.size

        ymask = self._resid[mask]
        ylim = np.nan_to_num([np.nanmin(ymask), np.nanmax(ymask)])
        diff = .05 * (ylim[1] - ylim[0])
        self.resid_axes.set_ylim(ylim[0] - diff, ylim[1] + diff)

    def _relayout(self):
        if self._resid_artist is not None:
            self.axes.set_position([0.1, .35, .88, .6])
            self.resid_axes.set_position([0.1, .15, .88, .2])
            self.resid_axes.set_xlabel(self._xlabel)
            self.resid_axes.set_visible(True)
            self.axes.set_xlabel('')
            [t.set_visible(False) for t in self.axes.get_xticklabels()]
        else:
            self.resid_axes.set_visible(False)
            self.axes.set_position([0.1, .15, .88, .83])
            self.axes.set_xlabel(self._xlabel)
            [t.set_visible(True) for t in self.axes.get_xticklabels()]

    def set_profile(self, x, y, xatt=None, yatt=None, **kwargs):
        """
        Set a new line profile

        :param x: X-coordinate data
        :type x: array-like

        :param y: Y-coordinate data
        :type y: array-like

        :param xatt: ComponentID associated with X axis
        :type xatt: :class:`~glue.core.data.ComponentID`

        :param yatt: ComponentID associated with Y axis
        :type yatt: :class:`~glue.core.data.ComponentID`

        Extra kwargs are passed to matplotlib.plot, to
        customize plotting

        Returns the created MPL artist
        """
        self.clear_fit()
        self._x = np.asarray(x).ravel()
        self._xatt = xatt
        self._y = np.asarray(y).ravel()
        self._yatt = yatt
        if self._artist is not None:
            self._artist.remove()

        kwargs.setdefault('drawstyle', 'steps-post')

        self._artist = self.axes.plot(x, y, **kwargs)[0]
        self._relayout()
        self._redraw()

        return self._artist

    def clear_fit(self):
        for a in self._fit_artists:
            a.remove()
        self._fit_artists = []
        if self._resid_artist is not None:
            self._resid_artist.remove()
            self._resid_artist = None

    def connect(self):
        connect = self.axes.figure.canvas.mpl_connect
        self._down_id = connect('button_press_event', self._on_down)
        self._up_id = connect('button_release_event', self._on_up)
        self._move_id = connect('motion_notify_event', self._on_move)

    def disconnect(self):
        off = self.axes.figure.canvas.mpl_disconnect
        self._down_id = off(self._down_id)
        self._up_id = off(self._up_id)
        self._move_id = off(self._move_id)

    def _on_down(self, event):
        if not event.inaxes:
            return

        if event.dblclick:
            if self.active_grip is not None:
                self.active_grip.dblclick(event.xdata, event.ydata)
            return

        if self.active_grip is not None and self.active_grip.enabled:
            self.active_grip.select(event.xdata, event.ydata)

    def _on_up(self, event):
        if not event.inaxes:
            return
        if self.active_grip is None or not self.active_grip.enabled:
            return

        self.active_grip.release()

    def _on_move(self, event):
        if not event.inaxes or event.button != 1:
            return
        if self.active_grip is None or not self.active_grip.enabled:
            return

        self.active_grip.drag(event.xdata, event.ydata)

    def _redraw(self):
        self.axes.figure.canvas.draw()

    def profile_data(self, xlim=None):
        if self._x is None or self._y is None:
            raise ValueError("Must set profile first")

        x = self._x
        y = self._y
        if xlim is not None:
            mask = (min(xlim) <= x) & (x <= max(xlim))
            x = x[mask]
            y = y[mask]

        return x, y

    def fit(self, fitter, xlim=None):
        try:
            x, y = self.profile_data(xlim)
            dy = None
        except ValueError:
            raise ValueError("Must set profile before fitting")

        result = fitter.build_and_fit(x, y)

        return result, x, y, dy

    def plot_fit(self, fitter, fit_result):
        self.clear_fit()
        x = self._x
        y = fitter.predict(fit_result, x)
        self._fit_artists = fitter.plot(fit_result, self.axes, x)
        resid = self._y - y
        self._resid = resid
        self._resid_artist, = self.resid_axes.plot(x, resid, 'k')
        self.autoscale_ylim()
        self._relayout()

    def new_value_grip(self, callback=None):
        """
        Create and return new ValueGrip

        :param callback: A callback function to be invoked
        whenever the grip.value property changes
        """
        result = self.value_cls(self)
        result.value = self._center[0]

        if callback is not None:
            add_callback(result, 'value', callback)
        self.grips.append(result)
        self.active_grip = result
        return result

    def new_range_grip(self, callback=None):
        """
        Create and return new RangeGrip

        :param callback: A callback function to be invoked
        whenever the grip.range property changes
        """
        result = self.range_cls(self)
        center = self._center[0]
        width = self._width
        result.range = center - width / 4, center + width / 4

        if callback is not None:
            add_callback(result, 'range', callback)

        self.grips.append(result)
        self.active_grip = result

        return result

    @property
    def _center(self):
        """Return the data coordinates of the axes center, as (x, y)"""
        xy = self.axes.transAxes.transform([(.5, .5)])
        xy = self.axes.transData.inverted().transform(xy)
        return tuple(xy.ravel())

    @property
    def _width(self):
        """Return the X-width of axes in data units"""
        xlim = self.axes.get_xlim()
        return xlim[1] - xlim[0]

    def pick_grip(self, x, y):
        """
        Given a coordinate in Data units,
        return the enabled Grip object nearest
        that point, or None if none are nearby
        """
        grips = [h for h in self.grips if h.enabled]
        if not grips:
            return

        dist, grip = min((h.pick_dist(x, y), h)
                         for h in grips)

        if dist < PICK_THRESH:
            return grip
