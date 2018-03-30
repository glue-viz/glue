from __future__ import absolute_import, division, print_function

from qtpy import QtWidgets
from qtpy.QtCore import Qt

from glue.viewers.matplotlib.qt.toolbar import MatplotlibViewerToolbar
from glue.viewers.matplotlib.qt.data_viewer import MatplotlibDataViewer
from glue.viewers.profile.qt.layer_style_editor import ProfileLayerStyleEditor
from glue.viewers.profile.layer_artist import ProfileLayerArtist
from glue.viewers.profile.qt.options_widget import ProfileOptionsWidget
from glue.viewers.profile.state import ProfileViewerState
from glue.viewers.profile.qt.profile_tools import ProfileTools

from glue.viewers.common.qt import toolbar_mode  # noqa

__all__ = ['ProfileViewer']


class ProfileViewer(MatplotlibDataViewer):

    LABEL = '1D Profile'
    _toolbar_cls = MatplotlibViewerToolbar
    _layer_style_widget_cls = ProfileLayerStyleEditor
    _state_cls = ProfileViewerState
    _options_cls = ProfileOptionsWidget
    _data_artist_cls = ProfileLayerArtist
    _subset_artist_cls = ProfileLayerArtist

    tools = ['select:xrange', 'profile-analysis']

    def __init__(self, session, parent=None, state=None):
        super(ProfileViewer, self).__init__(session, parent, state=state)
        self.state.add_callback('x_att', self._update_axes)
        self.state.add_callback('y_att', self._update_axes)

    def _update_axes(self, *args):

        if self.state.x_att is not None:
            self.state.x_axislabel = self.state.x_att.label

        if self.state.y_att is not None:
            self.state.y_axislabel = self.state.y_att.label

        self.axes.figure.canvas.draw()

    def _roi_to_subset_state(self, roi):
        x_comp = self.state.x_att.parent.get_component(self.state.x_att)
        return x_comp.subset_from_roi(self.state.x_att, roi, coord='x')
