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
