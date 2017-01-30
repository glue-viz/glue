from __future__ import absolute_import, division, print_function

import uuid

from glue.utils import nonpartial

from glue.viewers.image.state import ImageLayerState
from glue.core.layer_artist import LayerArtistBase
from glue.viewers.image.mpl_image_artist import CompositeImageArtist
from glue.viewers.common.mpl_layer_artist import MatplotlibLayerArtist
from matplotlib.colors import ColorConverter

COLOR_CONVERTER = ColorConverter()


# NOTE: we use LayerArtistBase here instead of MatplotlibLayerArtist since we
#       are not making use of normal Matplotlib artists but a special object
#       to do the image compositing.


class ImageLayerArtist(LayerArtistBase):

    def __init__(self, layer, axes, viewer_state):

        super(ImageLayerArtist, self).__init__(layer)

        self.axes = axes
        self.viewer_state = viewer_state

        # Set up a state object for the layer artist
        self.layer_state = ImageLayerState(layer=layer)
        self.viewer_state.layers.append(self.layer_state)

        # Watch for changes in the viewer state which would require the
        # layers to be redrawn
        # TODO: don't connect to ALL signals here
        # self.viewer_state.connect_all(nonpartial(self.update))
        self.viewer_state.add_callback('xcoord', nonpartial(self.update))
        self.viewer_state.add_callback('ycoord', nonpartial(self.update))
        self.viewer_state.add_callback('att', nonpartial(self.update))

        self.layer_state.add_callback('*', nonpartial(self.update))

        # TODO: following is temporary
        self.layer_state.data_collection = self.viewer_state.data_collection
        self.data_collection = self.viewer_state.data_collection

        # TODO: set up matplotlib artist that can take multiple arrays and
        # colors. This could act like for the 3D viewers where each layer artist
        # has a unique ID. In future, this kind of artist could even deal with
        # the reprojection on-the-fly.

        # We use a custom object to deal with the compositing of images, and we
        # store it as a private attribute of the axes to make sure it is
        # accessible for all layer artists.
        self.uuid = str(uuid.uuid4())
        if not hasattr(self.axes, '_image'):
            self.axes._image = CompositeImageArtist(self.axes)
        self.axes._image.allocate(self.uuid)

    def clear(self):
        self.axes._image.deallocate(self.uuid)

    @property
    def zorder(self):
        # FIXME: need a public API for following
        return self.axes._image.layers[self.uuid]['zorder']

    @zorder.setter
    def zorder(self, value):
        return self.axes._image.set(self.uuid, zorder=value)

    @property
    def visible(self):
        # FIXME: need a public API for following
        return self.axes._image.layers[self.uuid]['visible']

    @visible.setter
    def visible(self, value):
        return self.axes._image.set(self.uuid, visible=value)

    def redraw(self):
        self.axes.figure.canvas.draw()

    def update(self):

        if self.layer_state.att is None:
            return

        data = self.layer[self.layer_state.att[0]]

        self.axes._image.set(self.uuid, array=data, color=self.layer_state.color,
                             vmin=self.layer_state.vmin, vmax=self.layer_state.vmax,
                             alpha=self.layer_state.alpha)

        # Reset the axes stack so that pressing the home button doesn't go back
        # to a previous irrelevant view.
        self.axes.figure.canvas.toolbar.update()

        self.redraw()


class ImageSubsetLayerArtist(MatplotlibLayerArtist):

    def __init__(self, layer, axes, viewer_state):

        super(ImageSubsetLayerArtist, self).__init__(layer, axes, viewer_state)

        # Set up a state object for the layer artist
        self.layer_state = ImageLayerState(layer=layer)
        self.viewer_state.layers.append(self.layer_state)

        # Watch for changes in the viewer state which would require the
        # layers to be redrawn
        # TODO: don't connect to ALL signals here
        # self.viewer_state.connect_all(nonpartial(self.update))
        self.viewer_state.add_callback('xcoord', nonpartial(self.update))
        self.viewer_state.add_callback('ycoord', nonpartial(self.update))
        self.viewer_state.add_callback('att', nonpartial(self.update))

        self.layer_state.add_callback('*', nonpartial(self.update))

        # TODO: following is temporary
        self.layer_state.data_collection = self.viewer_state.data_collection
        self.data_collection = self.viewer_state.data_collection

        # TODO: set up matplotlib artist that can take multiple arrays and
        # colors. This could act like for the 3D viewers where each layer artist
        # has a unique ID. In future, this kind of artist could even deal with
        # the reprojection on-the-fly.

        # We use a custom object to deal with the compositing of images, and we
        # store it as a private attribute of the axes to make sure it is
        # accessible for all layer artists.
        self.uuid = str(uuid.uuid4())
        if not hasattr(self.axes, '_image'):
            self.axes._image = CompositeImageArtist(self.axes)
        self.axes._image.allocate(self.uuid)

    def clear(self):
        self.axes._image.deallocate(self.uuid)

    @property
    def zorder(self):
        # FIXME: need a public API for following
        return self.axes._image.layers[self.uuid]['zorder']

    @zorder.setter
    def zorder(self, value):
        return self.axes._image.set(self.uuid, zorder=value)

    @property
    def visible(self):
        # FIXME: need a public API for following
        return self.axes._image.layers[self.uuid]['visible']

    @visible.setter
    def visible(self, value):
        return self.axes._image.set(self.uuid, visible=value)

    def redraw(self):
        self.axes.figure.canvas.draw()

    def update(self):

        if self.layer_state.att is None:
            return

        data = self.layer[self.layer_state.att[0]]

        self.axes._image.set(self.uuid, array=data, color=self.layer_state.color,
                             vmin=self.layer_state.vmin, vmax=self.layer_state.vmax,
                             alpha=self.layer_state.alpha)

        # Reset the axes stack so that pressing the home button doesn't go back
        # to a previous irrelevant view.
        self.axes.figure.canvas.toolbar.update()

        self.redraw()
