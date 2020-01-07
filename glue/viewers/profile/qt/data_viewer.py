from glue.utils import defer_draw, decorate_all_methods

from glue.viewers.matplotlib.qt.data_viewer import MatplotlibDataViewer
from glue.viewers.profile.qt.layer_style_editor import ProfileLayerStyleEditor
from glue.viewers.profile.qt.layer_artist import QThreadedProfileLayerArtist
from glue.viewers.profile.qt.options_widget import ProfileOptionsWidget
from glue.viewers.profile.state import ProfileViewerState

from glue.viewers.profile.qt.profile_tools import ProfileAnalysisTool  # noqa
from glue.viewers.profile.viewer import MatplotlibProfileMixin

__all__ = ['ProfileViewer']


@decorate_all_methods(defer_draw)
class ProfileViewer(MatplotlibProfileMixin, MatplotlibDataViewer):

    LABEL = '1D Profile'
    _layer_style_widget_cls = ProfileLayerStyleEditor
    _state_cls = ProfileViewerState
    _options_cls = ProfileOptionsWidget
    _data_artist_cls = QThreadedProfileLayerArtist
    _subset_artist_cls = QThreadedProfileLayerArtist

    large_data_size = 1e8

    allow_duplicate_data = True

    tools = ['select:xrange', 'profile-analysis']

    def __init__(self, session, parent=None, state=None):
        MatplotlibDataViewer.__init__(self, session, parent=parent, state=state)
        MatplotlibProfileMixin.setup_callbacks(self)
