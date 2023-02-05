from glue.viewers.matplotlib.mpl_axes import init_mpl

from matplotlib.figure import Figure
import numpy as np


class ProjectionMplTransform(object):
    def __init__(self, projection, x_lim, y_lim, x_scale, y_scale):
        self._state = {'projection': projection, 'x_lim': x_lim, 'y_lim': y_lim,
                       'x_scale': x_scale, 'y_scale': y_scale}
        _, axes = init_mpl(Figure(), projection=self._state['projection'])
        axes.set_xscale(self._state['x_scale'])
        axes.set_yscale(self._state['y_scale'])
        if self._state['projection'] not in ['aitoff', 'hammer', 'lambert', 'mollweide']:
            axes.set_xlim(self._state['x_lim'])
            axes.set_ylim(self._state['y_lim'])
        self._transform = (axes.transData + axes.transAxes.inverted()).frozen()

    def __call__(self, x, y):
        points = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        res = self._transform.transform(points)
        out = np.hsplit(res, 2)
        return out[0].reshape(x.shape), out[1].reshape(y.shape)

    def __gluestate__(self, context):
        return dict(state=context.do(self._state))

    @classmethod
    def __setgluestate__(cls, rec, context):
        state = context.object(rec['state'])
        return cls(state['projection'], state['x_lim'], state['y_lim'],
                   state['x_scale'], state['y_scale'])


class RadianTransform(object):
    # We define 'next_transform' so that this pre-transform can
    # be chained together with another transformation, if desired
    def __init__(self, coords=[], next_transform=None):
        self._next_transform = next_transform
        self._coords = coords
        self._state = {"coords": coords, "next_transform": next_transform}

    def __call__(self, x, y):
        if 'x' in self._coords:
            x = np.deg2rad(x)
        if 'y' in self._coords:
            y = np.deg2rad(y)
        if self._next_transform is not None:
            return self._next_transform(x, y)
        else:
            return x, y

    def __gluestate__(self, context):
        return dict(state=context.do(self._state))

    @classmethod
    def __setgluestate__(cls, rec, context):
        state = context.object(rec['state'])
        return cls(state['coords'], state['next_transform'])


class FullSphereLongitudeTransform(object):

    def __init__(self, next_transform=None):
        self._next_transform = next_transform
        self._state = {"next_transform": next_transform}

    def __call__(self, x, y):
        x = np.mod(x + np.pi, 2 * np.pi) - np.pi
        if self._next_transform is not None:
            return self._next_transform(x, y)
        else:
            return x, y

    @classmethod
    def __setgluestate__(cls, rec, context):
        state = context.object(rec['state'])
        return cls(state['next_transform'])

    def __gluestate__(self, context):
        return dict(state=context.do(self._state))
