from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

import glue
from glue import VizClient


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


class ImageClient(VizClient):

    def __init__(self, data, figure=None, axes=None, area_style='filled'):

        if axes is not None and figure is not None and \
                axes.figure is not figure:
            raise Exception("Axes and figure are incompatible")

        VizClient.__init__(self, data)

        # layers dict is keyed by data sets. Values are dicts with keys:
        # 'artist' : mpl artist
        # 'att': attribute to display
        # 'cmap' : the color map object
        self.layers = defaultdict(dict)
        self.draw_colorbar = False
        self.colorbar = None

        self.display_data = None
        self.display_attribute = None
        self.slice_ori = 2
        self._slice_ind = 0

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
        if not self.display_data: return False
        return len(self.display_data.shape) == 3

    @property
    def slice_ind(self):
        return self._slice_ind

    @slice_ind.setter
    def slice_ind(self, value):
        self._slice_ind = value
        self._update_data_plot()
        self._redraw()

    def set_data(self, data, attribute=None):
        if data not in self.data:
            raise TypeError("Data is not in parent DataCollection")
        if len(data.shape) not in [2, 3]:
            raise TypeError("Data must be 2- or 3-dimensional")

        if attribute:
            self.layers[data]['att'] = attribute
        elif data not in self.layers:
            self.layers[data]['att'] = data.component_ids()[0]
        attribute = self.layers[data]['att']

        #pick which attribute to show
        self.display_data = data
        self.display_attribute = attribute
        self._update_data_plot(relim=True, reset=True)
        self._redraw()

    def slice_bounds(self):
        if not self.is_3D: return (0,0)
        if self.slice_ori == 2:
            return (0, self.display_data.shape[2]-1)
        if self.slice_ori == 1:
            return (0, self.display_data.shape[1]-1)
        if self.slice_ori == 0:
            return (0, self.display_data.shape[0]-1)

    def set_slice_ori(self, ori):
        if ori not in [0,1,2]:
            raise TypeError("Orientation must be 0, 1, or 2")
        self.slice_ori = ori
        self.slice_ind = min(self.slice_ind, self.slice_bounds()[1])
        self.slice_ind = max(self.slice_ind, self.slice_bounds()[0])
        self._update_data_plot(relim=True, reset=True)
        self._redraw()

    def set_attribute(self, attribute):
        if not self.display_data or \
                attribute not in self.display_data.component_ids():
            raise TypeError("Attribute not in data's attributes: %s" % attribute)
        self.display_attribute = attribute
        self._update_data_plot()
        self._redraw()

    def _redraw(self):
        """
        Re-render the screen
        """
        self._ax.figure.canvas.draw()

    def set_colorbar_visible(self, state):
        self.draw_colorbar = state
        self.refresh()

    def toggle_colorbar(self):
        self.set_colorbar_visible(not self.draw_colorbar)

    def set_norm(self, vmin=None, vmax=None):
        if not self.display_data:
            return
        n = self.layers[self.display_data]['norm']
        if not n:
            n = InvNormalize()
            n.autoscale(self.display_data[self.display_attribute])

        if vmin: n.vmin = vmin
        if vmax: n.vmax = vmax
        self.layers[self.display_data]['norm'] = n
        self._update_data_plot()
        self._redraw()

    def set_cmap(self, cmap):
        if not self.display_data:
            return
        self.layers[self.display_data]['cmap'] = cmap
        self._update_data_plot()
        self._redraw()

    def _update_subset_plot(self, s):
        if s in self.layers and 'artist' in self.layers[s]:
            for item in self.layers[s]['artist'].collections:
                item.remove()

        mask = self.layers[s]['mask']

        # Extract 2D image
        if self.is_3D:
            if self.slice_ori == 2:
                mask = mask[:,:,self.slice_ind]
            elif self.slice_ori == 1:
                mask = mask[:,self.slice_ind,:]
            else:
                mask = mask[self.slice_ind, :, :]

        if self.area_style == 'contour':
            self.layers[s]['artist'] = self._ax.contour(mask.astype(float),
                                                        levels=[0.5],
                                                        colors=s.style.color)
        else:
            self.layers[s]['artist'] = self._ax.contourf(mask.astype(float),
                                                         levels=[0.5, 1.0], alpha=0.3,
                                                         colors = s.style.color)


    def _update_data_plot(self, relim=False, reset=False):
        """
        Sync the location of the scatter points to
        reflect what components are being plotted
        """

        if not self.display_data:
            return
        data = self.display_data[self.display_attribute]

        if not self.is_3D:
            self._image = data
        else:
            if self.slice_ori == 2:
                self._image = data[:,:,self.slice_ind]
            elif self.slice_ori == 1:
                self._image = data[:, self.slice_ind, :]
            else:
                self._image = data[self.slice_ind, :, :]

        data = self.display_data
        if 'cmap' not in self.layers[data]:
            self.layers[data]['cmap'] = plt.cm.gray
        if  'norm' not in self.layers[data]:
            self.layers[data]['norm'] = InvNormalize()
            self.layers[data]['norm'].autoscale(self._image)

        cmap = self.layers[data]['cmap']
        norm = self.layers[data]['norm']

        if reset:
            if 'artist' in self.layers[data]:
                self.layers[data]['artist'].remove()
                self.layers[data].pop('artist')

        if 'artist' not in self.layers[data]:
            plot = self._ax.imshow(self._image, cmap=cmap, norm=norm,
                                   interpolation='nearest', origin='lower')
            self.layers[data]['artist'] = plot
        else:
            self.layers[data]['artist'].set_data(self._image)
            self.layers[data]['artist'].set_norm(norm)
            self.layers[data]['artist'].set_cmap(cmap)

        artist = self.layers[data]['artist']

        for layer in self.layers:
            if layer is data:
                self.layers[layer]['artist'].set_visible(True)
                continue
            if isinstance(layer, glue.Data):
                self.layers[layer]['artist'].set_visible(False)
        if relim:
            self.relim()

    def relim(self):
        print self._image.shape, self.display_data.shape
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

        # XXX need to specify which direction the ROI 'cuts through'
        # for cubes
        mask = s.to_mask()
        assert mask.shape == s.data.shape
        self.layers[s]['mask'] = mask

        self._update_subset_plot(s)

    def _apply_roi(self, roi):
        data = self.display_data
        subset_state = glue.subset.RoiSubsetState(data)
        subset_state.xatt = 'XPIX'
        subset_state.yatt = 'YPIX'
        subset_state.roi = roi
        data.edit_subset.subset_state = subset_state

    def _remove_subset(self, message):
        self.delete_layer(message.sender)

    def delete_layer(self, layer):
        if 'artist' not in self.layers[layer] or not \
                self.layers[layer]['artist']:
            return
        if isinstance(layer, glue.Data):
            self.layers[layer]['artist'].remove()
        else:
            for item in self.layers[s].collections:
                item.remove()
        self.layers[layer].remove('artist')
        self._redraw()

    def _remove_data(self, message):
        self.delete_layer(message.sender)
        for s in message.sender.subsets:
            self.delete_layer(s)

    def init_layer(self, layer):
        if isinstance(layer, glue.Subset):
            self._update_subset_single(layer)

