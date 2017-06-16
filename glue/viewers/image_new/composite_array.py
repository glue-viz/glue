# This artist can be used to deal with the sampling of the data as well as any
# RGB blending.

from __future__ import absolute_import

import numpy as np

from matplotlib.transforms import TransformedBbox
from matplotlib.colors import ColorConverter, Colormap
from astropy.visualization import LinearStretch, ManualInterval, ContrastBiasStretch

__all__ = ['CompositeArray']

COLOR_CONVERTER = ColorConverter()


class CompositeArray(object):

    def __init__(self, ax, **kwargs):

        self.axes = ax

        # We keep a dictionary of layers. The key should be the UUID of the
        # layer artist, and the values should be dictionaries that contain
        # 'zorder', 'visible', 'array', 'color', and 'alpha'.
        self.layers = {}

        self._first = True

        # ax.set_ylim((df[y].min(), df[y].max()))
        # ax.set_xlim((df[x].min(), df[x].max()))
        # self.set_array([[1, 1], [1, 1]])

    def allocate(self, uuid):
        self.layers[uuid] = {'zorder': 0,
                             'visible': True,
                             'array': None,
                             'color': '0.5',
                             'alpha': 1,
                             'clim': (0, 1),
                             'contrast': 1,
                             'bias': 0.5,
                             'stretch': LinearStretch()}

    def deallocate(self, uuid):
        self.layers.pop(uuid)
        if len(self.layers) == 0:
            self.shape = None

    def set(self, uuid, **kwargs):
        for key, value in kwargs.items():
            if key not in self.layers[uuid]:
                raise KeyError("Unknown key: {0}".format(key))
            else:
                self.layers[uuid][key] = value

    @property
    def shape(self):
        if len(self.layers) > 0:
            first_layer = list(self.layers.values())[0]
            if callable(first_layer['array']):
                array = first_layer['array']()
            else:
                array = first_layer['array']
            return array.shape
        else:
            return None

    def __getitem__(self, view):

        img = None
        visible_layers = 0

        for uuid in sorted(self.layers, key=lambda x: self.layers[x]['zorder']):

            layer = self.layers[uuid]

            if not layer['visible']:
                continue

            interval = ManualInterval(*layer['clim'])
            contrast_bias = ContrastBiasStretch(layer['contrast'], layer['bias'])

            if callable(layer['array']):
                array = layer['array']()
            else:
                array = layer['array']

            array_sub = array[view]
            if np.isscalar(array_sub):
                scalar = True
                array_sub = np.atleast_2d(array_sub)
            else:
                scalar = False

            data = layer['stretch'](contrast_bias(interval(array_sub)))

            if isinstance(layer['color'], Colormap):

                if img is None:
                    img = np.ones(data.shape + (4,))

                # Compute colormapped image
                plane = layer['color'](data)

                # Use traditional alpha compositing
                img *= (1 - layer['alpha'])
                plane = plane * layer['alpha']

            else:

                if img is None:
                    img = np.zeros(data.shape + (4,))

                # Get color and pre-multiply by alpha values
                color = COLOR_CONVERTER.to_rgba_array(layer['color'])[0]
                color *= layer['alpha']

                plane = data[:, :, np.newaxis] * color
                plane[:, :, 3] = 1

                visible_layers += 1

            if scalar:
                plane = plane[0, 0]

            img += plane

        if img is None:
            img = np.zeros(self.shape + (4,))

        img = np.clip(img, 0, 1)

        return img

    @property
    def dtype(self):
        return np.float

    @property
    def ndim(self):
        return 2

    @property
    def size(self):
        return np.product(self.shape)
