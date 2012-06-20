import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

import glue
from glue import VizClient
from glue.exceptions import IncompatibleAttribute


class InvNormalize(Normalize):
    """ Simple wrapper to matplotlib Normalize object, that
    handles the case where vmax <= vmin """
    def __call__(self, value):
        if self.vmax <= self.vmin:
            self.vmax, self.vmin = self.vmin, self.vmax
            result = 1 - Normalize.__call__(self, value)
            self.vmax, self.vmin = self.vmin, self.vmax
        else:
            result = Normalize.__call__(self, value)
        return result


class LayerManager(object):
    def __init__(self, layer, axes):
        self.layer = layer
        self.artist = None
        self.component_id = None
        self._ax = axes

    def set_visible(self, state):
        raise NotImplementedError

    def delete_artist(self):
        raise NotImplementedError

    def update_artist(self, image):
        raise NotImplementedError

    def __del__(self):
        self.delete_artist()


class DataLayerManager(LayerManager):
    def __init__(self, layer, axes):
        super(DataLayerManager, self).__init__(layer, axes)
        self.cmap = plt.cm.gray
        self.norm = InvNormalize()

    def update_artist(self, image):
        self.delete_artist()
        self.artist = self._ax.imshow(image, cmap=self.cmap,
                                      norm=self.norm,
                                      interpolation='nearest',
                                      origin='lower')

    def set_visible(self, state):
        if self.artist is None:
            return
        self.artist.set_visible(state)

    def set_norm(self, data, vmin=None, vmax=None):
        self.norm.autoscale(data)
        if vmin is not None:
            self.norm.vmin = vmin
        if vmax is not None:
            self.norm.vmax = vmax

    def delete_artist(self):
        if self.artist is None:
            return
        self.artist.remove()
        self.artist = None


class SubsetLayerManager(LayerManager):
    def __init__(self, layer, axes, area_style='filled'):
        super(SubsetLayerManager, self).__init__(layer, axes)
        self.mask = None
        self.area_style = area_style

    def set_visible(self, state):
        if self.artist is None:
            return
        for item in self.artist.collections:
            item.set_visible(state)

    def delete_artist(self):
        if self.artist is None:
            return
        for item in self.artist.collections:
            item.remove()
        self.artist = None

    def update_artist(self, mask):
        self.delete_artist()
        if self.area_style == 'filled':
            self.artist = self._ax.contourf(mask.astype(float),
                                            levels=[0.5, 1.0],
                                            alpha=0.3,
                                            colors=self.layer.style.color)
        else:
            self.artist = self._ax.contour(mask.astype(float),
                                           levels=[0.5],
                                           colors=self.layer.style.color)


class ImageClient(VizClient):

    def __init__(self, data, figure=None, axes=None, area_style='filled'):

        if axes is not None and figure is not None and \
                axes.figure is not figure:
            raise Exception("Axes and figure are incompatible")

        VizClient.__init__(self, data)

        self.layers = {}

        self.display_data = None
        self.display_attribute = None
        self._slice_ori = 2
        self._slice_ind = 0
        self._image = None

        if axes is not None:
            self._ax = axes
            self._figure = axes.figure
        else:
            if figure is None:
                self._figure = plt.figure()
            self._ax = self._figure.add_subplot(1, 2, 1)

        if area_style in ['contour', 'filled']:
            self.area_style = area_style
        else:
            raise Exception("area_style should be one of contour/filled")

    @property
    def is_3D(self):
        if not self.display_data:
            return False
        return len(self.display_data.shape) == 3

    @property
    def slice_ind(self):
        if self.is_3D:
            return self._slice_ind
        return None

    @property
    def image(self):
        return self._image

    @slice_ind.setter
    def slice_ind(self, value):
        if self.is_3D:
            self._slice_ind = value
            self._update_data_plot()
            self._redraw()
        else:
            raise IndexError("Cannot set slice for 2D image")

    def can_handle_data(self, data):
        return data.ndim in [2, 3]

    def _ensure_data_present(self, data):
        if data not in self.layers:
            self.add_layer(data)

    def set_data(self, data, attribute=None):
        self._ensure_data_present(data)

        if attribute:
            self.layers[data].component_id = attribute
        elif self.layers[data].component_id is None:
            self.layers[data].component_id = data.component_ids()[0]
        attribute = self.layers[data].component_id

        self.display_data = data
        self.display_attribute = attribute
        self._update_data_plot(relim=True)
        self._update_visibilities()
        self._redraw()

    def slice_bounds(self):
        if not self.is_3D:
            return (0, 0)
        if self._slice_ori == 2:
            return (0, self.display_data.shape[2] - 1)
        if self._slice_ori == 1:
            return (0, self.display_data.shape[1] - 1)
        if self._slice_ori == 0:
            return (0, self.display_data.shape[0] - 1)

    def set_slice_ori(self, ori):
        if not self.is_3D:
            raise IndexError("Cannot set orientation of 2D image")
        if ori not in [0, 1, 2]:
            raise TypeError("Orientation must be 0, 1, or 2")
        self._slice_ori = ori
        self.slice_ind = min(self.slice_ind, self.slice_bounds()[1])
        self.slice_ind = max(self.slice_ind, self.slice_bounds()[0])
        self._update_data_plot(relim=True)
        for sub in self.display_data.subsets:
            self._update_subset_plot(sub)

        self._redraw()

    def set_attribute(self, attribute):
        if not self.display_data or \
                attribute not in self.display_data.component_ids():
            raise IncompatibleAttribute(
                "Attribute not in data's attributes: %s" % attribute)
        self.display_attribute = attribute
        self.layers[self.display_data].component_id = attribute
        self._update_data_plot()
        self._redraw()

    def _redraw(self):
        """
        Re-render the screen
        """
        self._ax.figure.canvas.draw()

    def set_norm(self, vmin=None, vmax=None):
        if not self.display_data:
            return
        data = self.display_data[self.display_attribute]
        self.layers[self.display_data].set_norm(data, vmin, vmax)
        self._update_data_plot()
        self._redraw()

    def set_cmap(self, cmap):
        if not self.display_data:
            return
        self.layers[self.display_data].cmap = cmap
        self._update_data_plot()
        self._redraw()

    def _update_subset_plot(self, s):
        if s not in self.layers:
            return

        mask = self.layers[s].mask
        mask = self._extract_slice_from_data(data=mask)
        self.layers[s].update_artist(mask)

    def _extract_slice_from_data(self, data=None):
        if data is None:
            result = self.display_data[self.display_attribute]
        else:
            result = data

        if not self.is_3D:
            return result
        if self._slice_ori == 2:
            result = result[:, :, self.slice_ind]
        elif self._slice_ori == 1:
            result = result[:, self.slice_ind, :]
        else:
            result = result[self.slice_ind, :, :]

        return result

    def _update_data_plot(self, relim=False):
        """
        Sync the location of the scatter points to
        reflect what components are being plotted
        """

        if not self.display_data:
            return

        self._image = self._extract_slice_from_data()
        self.layers[self.display_data].update_artist(self._image)

        if relim:
            self.relim()

    def _update_visibilities(self):
        for layer in self.layers:
            self.layers[layer].set_visible(layer.data is self.display_data)

    def relim(self):
        self._ax.set_xlim(0, self._image.shape[1])
        self._ax.set_ylim(0, self._image.shape[0])

    def _update_axis_labels(self):
        self._ax.set_xlabel('X')
        self._ax.set_ylabel('Y')

    def _update_subset_single(self, s):
        """
        Update the location and visual properties
        of each point in a single subset

        Parameters:
        ----------
        s: A subset instance
        The subset to refresh.

        """
        if self.display_data is None:
            return

        data = self.display_data
        if s.data is not data:
            return

        try:
            mask = s.to_mask()
        except IncompatibleAttribute:
            mask = np.zeros(s.data.shape, dtype=bool)

        assert mask.shape == s.data.shape
        self.layers[s].mask = mask
        self._update_subset_plot(s)

    def _apply_roi(self, roi):
        # XXX this will only work for 2D images right now
        data = self.display_data
        if data is None:
            return

        subset_state = glue.subset.RoiSubsetState()
        xroi, yroi = roi.to_polygon()
        x, y = self._get_axis_components()
        subset_state.xatt = x
        subset_state.yatt = y
        subset_state.roi = glue.roi.PolygonalROI(xroi, yroi)
        data.edit_subset.subset_state = subset_state

    def _horizontal_axis_index(self):
        """Which index (in numpy convention - zyx) does the horizontal
        axis coorespond to?"""
        if not self.is_3D or self._slice_ori == 2:
            return 1
        return 2

    def _vertical_axis_index(self):
        """Which index (in numpy convention - zyx) does the vertical
        axis coorespond to?"""
        if self.is_3D and self._slice_ori == 0:
            return 1
        return 0

    def _get_axis_components(self):
        data = self.display_data
        ids = [self._horizontal_axis_index(), self._vertical_axis_index()]
        return map(data.get_pixel_component_id, ids)

    def _remove_subset(self, message):
        self.delete_layer(message.sender)

    def delete_layer(self, layer):
        if layer not in self.layers:
            return
        manager = self.layers.pop(layer)
        del manager
        if layer is self.display_data:
            self.display_data = None

        if isinstance(layer, glue.Data):
            for subset in layer.subsets:
                self.delete_layer(subset)

        self._redraw()

    def _remove_data(self, message):
        self.delete_layer(message.data)
        for s in message.data.subsets:
            self.delete_layer(s)

    def init_layer(self, layer):
        self.add_layer(layer)

    def add_layer(self, layer):
        if layer in self.layers:
            return

        if layer.data not in self.data:
            raise TypeError("Data not managed by client's data collection")

        if not self.can_handle_data(layer.data):
            return

        if isinstance(layer, glue.Data):
            self.layers[layer] = DataLayerManager(layer, self._ax)
            for s in layer.subsets:
                self.add_layer(s)
        elif isinstance(layer, glue.Subset):
            self.layers[layer] = SubsetLayerManager(layer, self._ax)
            self._update_subset_single(layer)
        else:
            raise TypeError("Unrecognized layer type: %s" % type(layer))
