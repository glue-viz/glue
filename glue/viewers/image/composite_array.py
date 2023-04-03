# This artist can be used to deal with the sampling of the data as well as any
# RGB blending.

import warnings

import numpy as np

from glue.config import colormaps

from matplotlib.colors import ColorConverter, Colormap
from astropy.visualization import (LinearStretch, SqrtStretch, AsinhStretch,
                                   LogStretch, ManualInterval, ContrastBiasStretch)


__all__ = ['CompositeArray']

COLOR_CONVERTER = ColorConverter()

CMAP_SAMPLING = np.linspace(0, 1, 256)

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
        self._mode = 'color'

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value not in ['color', 'colormap']:
            raise ValueError("mode should be one of 'color' or 'colormap'")
        self._mode = value

    def allocate(self, uuid):
        self.layers[uuid] = {'zorder': 0,
                             'visible': True,
                             'array': None,
                             'shape': None,
                             'color': '0.5',
                             'cmap': colormaps.members[0][1],
                             'alpha': 1,
                             'clim': (0, 1),
                             'contrast': 1,
                             'bias': 0.5,
                             'stretch': 'linear'}

    def deallocate(self, uuid):
        self.layers.pop(uuid)

    def set(self, uuid, **kwargs):
        for key, value in kwargs.items():
            if key not in self.layers[uuid]:
                raise KeyError("Unknown key: {0}".format(key))
            elif key == 'color' and isinstance(value, Colormap):
                warnings.warn('Setting colormap using "color" key is deprecated, use "cmap" instead.', UserWarning)
                self.layers[uuid]['cmap'] = value
            else:
                self.layers[uuid][key] = value

    @property
    def shape(self):
        for layer in self.layers.values():
            if callable(layer['shape']):
                shape = layer['shape']()
            elif layer['shape'] is not None:
                shape = layer['shape']
            elif callable(layer['array']):
                array = layer['array']()
                if array is None:
                    return None
                else:
                    shape = array.shape
            else:
                shape = layer['array'].shape
            if shape is not None:
                return shape
        return None

    def __getitem__(self, item):
        return self()[item]

    def __call__(self, bounds=None):

        img = None
        visible_layers = 0

        # Get a sorted list of UUIDs with the top layers last
        sorted_uuids = sorted(self.layers, key=lambda x: self.layers[x]['zorder'])

        # We first check that layers are either all colormaps or all single colors.
        # In the case where we are dealing with colormaps, we can start from
        # the last layer that has an opacity of 1 because layers below will not
        # affect the output, assuming also that the colormaps do not change the
        # alpha
        if self.mode == 'colormap':
            for i in range(len(sorted_uuids) - 1, -1, -1):
                layer = self.layers[sorted_uuids[i]]
                if layer['visible']:
                    if layer['alpha'] == 1 and layer['cmap'](CMAP_SAMPLING)[:, 3].min() == 1:
                        sorted_uuids = sorted_uuids[i:]
                        break

        for uuid in sorted_uuids:

            layer = self.layers[uuid]

            if not layer['visible']:
                continue

            interval = ManualInterval(*layer['clim'])
            contrast_bias = ContrastBiasStretch(layer['contrast'], layer['bias'])
            stretch = STRETCHES[layer['stretch']]()

            if callable(layer['array']):
                array = layer['array'](bounds=bounds)
            else:
                array = layer['array']

            if array is None:
                continue

            if np.isscalar(array):
                scalar = True
                array = np.atleast_2d(array)
            else:
                scalar = False

            data = interval(array)
            data = contrast_bias(data, out=data)
            data = stretch(data, out=data)
            data[np.isnan(data)] = 0

            if self.mode == 'colormap':

                if img is None:
                    img = np.ones(data.shape + (4,))

                # Compute colormapped image
                plane = layer['cmap'](data)

                # Check what the smallest colormap alpha value for this layer is
                # - if it is 1 then this colormap does not change transparency,
                # and this allows us to speed things up a little.

                if layer['cmap'](CMAP_SAMPLING)[:, 3].min() == 1:

                    if layer['alpha'] == 1:
                        img[...] = 0
                    else:
                        plane *= layer['alpha']
                        img *= (1 - layer['alpha'])

                else:

                    # Use traditional alpha compositing

                    alpha_plane = layer['alpha'] * plane[:, :, 3]

                    plane[:, :, 0] = plane[:, :, 0] * alpha_plane
                    plane[:, :, 1] = plane[:, :, 1] * alpha_plane
                    plane[:, :, 2] = plane[:, :, 2] * alpha_plane

                    img[:, :, 0] *= (1 - alpha_plane)
                    img[:, :, 1] *= (1 - alpha_plane)
                    img[:, :, 2] *= (1 - alpha_plane)

                img[:, :, 3] = 1

            else:

                if img is None:
                    img = np.zeros(data.shape + (4,))

                # Get color and pre-multiply by alpha values
                color = COLOR_CONVERTER.to_rgba_array(layer['color'])[0]
                color *= layer['alpha']

                # We should treat NaN values as zero (post-stretch), which means
                # that those pixels don't contribute towards the final image.
                reset = np.isnan(data)
                if np.any(reset):
                    data[reset] = 0.

                plane = data[:, :, np.newaxis] * color
                plane[:, :, 3] = 1

                visible_layers += 1

            if scalar:
                plane = plane[0, 0]

            img += plane

        if img is None:
            return None
        else:
            img = np.clip(img, 0, 1, out=img)

        return img

    @property
    def dtype(self):
        return np.dtype(float)

    @property
    def ndim(self):
        return 2

    @property
    def size(self):
        return np.prod(self.shape)

    def __contains__(self, item):
        return item in self.layers
