# This artist can be used to deal with the sampling of the data as well as any
# RGB blending.

from __future__ import absolute_import

import numpy as np

from matplotlib.colors import ColorConverter, Colormap
from astropy.visualization import (LinearStretch, SqrtStretch, AsinhStretch,
                                   LogStretch, ManualInterval, ContrastBiasStretch)

__all__ = ['CompositeArray']

COLOR_CONVERTER = ColorConverter()

STRETCHES = {
    'linear': LinearStretch,
    'sqrt': SqrtStretch,
    'arcsinh': AsinhStretch,
    'log': LogStretch
}

class CompositeArray(object):

    def __init__(self, **kwargs):

        # We keep a dictionary of layers. The key should be the UUID of the
        # layer artist, and the values should be dictionaries that contain
        # 'zorder', 'visible', 'array', 'color', and 'alpha'.
        self.layers = {}

        self._first = True

    def allocate(self, uuid):
        self.layers[uuid] = {'zorder': 0,
                             'visible': True,
                             'array': None,
                             'color': '0.5',
                             'alpha': 1,
                             'clim': (0, 1),
                             'contrast': 1,
                             'bias': 0.5,
                             'stretch': 'linear'}

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
        for layer in self.layers.values():
            if callable(layer['array']):
                array = layer['array']()
            else:
                array = layer['array']
            if array is not None:
                return array.shape
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

            if array is None:
                continue

            array_sub = array[view]
            if np.isscalar(array_sub):
                scalar = True
                array_sub = np.atleast_2d(array_sub)
            else:
                scalar = False

            data = STRETCHES[layer['stretch']]()(contrast_bias(interval(array_sub)))

            if isinstance(layer['color'], Colormap):

                if img is None:
                    img = np.ones(data.shape + (4,))

                # Compute colormapped image
                plane = layer['color'](data)

                # Use traditional alpha compositing
                plane[:, :, 0] = plane[:, :, 0] * layer['alpha'] * plane[:, :, 3]
                plane[:, :, 1] = plane[:, :, 1] * layer['alpha'] * plane[:, :, 3]
                plane[:, :, 2] = plane[:, :, 2] * layer['alpha'] * plane[:, :, 3]

                img[:, :, 0] *= (1 - plane[:, :, 3])
                img[:, :, 1] *= (1 - plane[:, :, 3])
                img[:, :, 2] *= (1 - plane[:, :, 3])
                img[:, :, 3] = 1


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
