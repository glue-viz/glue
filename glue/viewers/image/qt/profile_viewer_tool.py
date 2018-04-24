from glue.config import viewer_tool
from glue.viewers.common.qt.tool import Tool


@viewer_tool
class ProfileViewerTool(Tool):

    icon = 'glue_spectrum'
    tool_id = 'profile-viewer'

    def __init__(self, viewer):
        super(ProfileViewerTool, self).__init__(viewer)

    def activate(self):
        from glue.viewers.profile.qt import ProfileViewer
        profile_viewer = self.viewer.session.application.new_data_viewer(ProfileViewer)
        for data in self.viewer.session.data_collection:
            if data in self.viewer._layer_artist_container:
                profile_viewer.add_data(data)
        profile_viewer.state.reference_data = self.viewer.state.reference_data
