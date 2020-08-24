from glue.config import viewer_tool
from glue.viewers.common.tool import Tool
from glue.core.qt.dialogs import info, warn
from glue.core.component_id import PixelComponentID


@viewer_tool
class ProfileViewerTool(Tool):

    icon = 'glue_spectrum'
    tool_id = 'profile-viewer'

    @property
    def profile_viewers_exist(self):
        from glue.viewers.profile.qt import ProfileViewer
        for tab in self.viewer.session.application.viewers:
            for viewer in tab:
                if isinstance(viewer, ProfileViewer):
                    return True
        return False

    def activate(self):

        if self.profile_viewers_exist:

            proceed = warn('A profile viewer was already created',
                           'Do you really want to create a new one?',
                           default='Cancel', setting='show_warn_profile_duplicate')

            if not proceed:
                return

        else:

            proceed = info('Creating a profile viewer',
                           'Note: profiles are '
                           'computed from datasets and subsets collapsed along all but one '
                           'dimension. To view the profile of part of the data, once you '
                           'click OK you can draw and update a subset in the current '
                           'image viewer and the profile will update accordingly.', setting='show_info_profile_open')

            if not proceed:
                return

        from glue.viewers.profile.qt import ProfileViewer
        profile_viewer = self.viewer.session.application.new_data_viewer(ProfileViewer)
        any_added = False
        for data in self.viewer.session.data_collection:
            if data in self.viewer._layer_artist_container:
                result = profile_viewer.add_data(data)
                any_added = any_added or result

        if not any_added:
            profile_viewer.close()
            return
