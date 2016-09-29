from __future__ import absolute_import, division, print_function

import numpy as np

from glue.external.echo import add_callback
from glue.utils import nonpartial
from glue.viewers.common.qt.mpl_toolbar import MatplotlibViewerToolbar
from glue.core.edit_subset_mode import EditSubsetMode
from glue.core import Data

from glue.viewers.common.qt.mpl_data_viewer import MatplotlibDataViewer
from glue.viewers.image.qt.layer_style_editor import ImageLayerStyleEditor
from glue.viewers.image.layer_artist import ImageLayerArtist
from glue.viewers.image.qt.options_widget import ImageOptionsWidget
from glue.viewers.image.state import ImageViewerState

__all__ = ['ImageViewer']


class ImageViewer(MatplotlibDataViewer):

    LABEL = 'New image viewer'
    _toolbar_cls = MatplotlibViewerToolbar
    _layer_style_widget_cls = ImageLayerStyleEditor
    _state_cls = ImageViewerState
    _options_cls = ImageOptionsWidget
    _data_artist_cls = ImageLayerArtist
    _subset_artist_cls = ImageLayerArtist

    tools = ['select:rectangle', 'select:xrange',
             'select:yrange', 'select:circle',
             'select:polygon']

    def __init__(self, session, parent=None):

        super(ImageViewer, self).__init__(session, parent)

        add_callback(self.viewer_state, 'xcoord', nonpartial(self.update_labels))
        add_callback(self.viewer_state, 'ycoord', nonpartial(self.update_labels))
        add_callback(self.viewer_state, 'aspect', nonpartial(self.update_aspect))

        self.update_labels()
        self.update_aspect()

    def update_labels(self):
        # if self.viewer_state.xcoord is not None:
        #     self.axes.set_xlabel(self.viewer_state.xcoord[0])
        # if self.viewer_state.ycoord is not None:
        #     self.axes.set_ylabel(self.viewer_state.ycoord[0])
        pass

    def update_aspect(self):
        self.axes.set_aspect(self.viewer_state.aspect)
        self.axes.figure.canvas.draw()

    def apply_roi(self, roi):

        pass
        # TODO: add back command stack here?
        # cmd = command.ApplyROI(client=self.client, roi=roi)
        # self._session.command_stack.do(cmd)

        # Does subset get applied to all data or just visible data?

        # for layer_artist in self._layer_artist_container:
        #
        #     if not isinstance(layer_artist.layer, Data):
        #         continue
        #
        #     x_comp = layer_artist.layer.get_component(self.viewer_state.xatt[0])
        #     y_comp = layer_artist.layer.get_component(self.viewer_state.yatt[0])
        #
        #     subset_state = x_comp.subset_from_roi(self.viewer_state.xatt[0], roi,
        #                                           other_comp=y_comp,
        #                                           other_att=self.viewer_state.yatt[0],
        #                                           coord='x')
        #
        #     mode = EditSubsetMode()
        #     mode.update(self._data, subset_state, focus_data=layer_artist.layer)
