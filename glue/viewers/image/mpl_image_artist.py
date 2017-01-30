# This artist can be used to deal with the sampling of the data as well as any
# RGB blending.

from __future__ import absolute_import

import numpy as np

import matplotlib.image as mimage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.colors import ColorConverter

__all__ = ['CompositeImageArtist']

COLOR_CONVERTER = ColorConverter()


class CompositeImageArtist(object):

    def __init__(self, ax, **kwargs):

        self.axes = ax

        # We create an image artist that remains invisible for now
        self.image = ax.imshow([[0]], interpolation='nearest')

        # We keep a dictionary of layers. The key should be the UUID of the
        # layer artist, and the values should be dictionaries that contain
        # 'zorder', 'visible', 'array', 'color', and 'alpha'.
        self.layers = {}

        self.shape = None

        self._first = True

        # ax.set_ylim((df[y].min(), df[y].max()))
        # ax.set_xlim((df[x].min(), df[x].max()))
        # self.set_array([[1, 1], [1, 1]])

    # TODO: maybe we should make this a dict sub-class...

    def allocate(self, uuid):
        self.layers[uuid] = {'zorder': 0,
                             'visible': True,
                             'array': None,
                             'color': None,
                             'alpha': 1,
                             'vmin': 0,
                             'vmax': 1}

    def deallocate(self, uuid):
        self.layers.pop(uuid)
        if len(self.layers) == 0:
            self.shape = None

    def set(self, uuid, **kwargs):
        for key, value in kwargs.items():
            if key not in self.layers[uuid]:
                raise KeyError("Unknown key: {0}".format(key))
            else:
                if key == 'array':
                    if self.shape is None:
                        self.shape = value.shape
                    else:
                        if value.shape != self.shape:
                            raise ValueError("data shape should be {0}".format(self.shape))
                if key == 'vmin' and value is None:
                    raise ValueError('who is setting vmin to None??')
                self.layers[uuid][key] = value
        self.update()

    def update(self):

        if self.shape is None:
            return

        # Construct image
        img = np.zeros(self.shape + (4,))
        for uuid in sorted(self.layers, key=lambda x: self.layers[x]['zorder']):

            layer = self.layers[uuid]

            # FIXME: the reason we check for vmin/vmax here is because when vmin
            # is first set, this is triggered before vmax is set. Need to avoid that.
            if not layer['visible'] or layer['vmin'] is None or layer['vmax'] is None:
                continue

            # Get color and pre-multiply by alpha values
            color = COLOR_CONVERTER.to_rgba_array(layer['color'])[0]
            color *= layer['alpha']

            # TODO: Here we could use the astropy normalization classes
            plane = color * np.ones(layer['array'].shape + (4,))
            data = (layer['array'] - layer['vmin']) / (layer['vmax'] - layer['vmin'])
            plane[:, :, 3] *= data

            print(plane)

            img += plane

        img = np.clip(img, 0, 1)

        self.image.set_clim(0, 1)
        self.image.set_array(img)
        self.image.set_extent([-0.5, self.shape[1] - 0.5, -0.5, self.shape[0] - 0.5])
        if self._first:
            self.axes.set_xlim(-0.5, self.shape[1] - 0.5)
            self.axes.set_ylim(-0.5, self.shape[0] - 0.5)
            self._first = False
