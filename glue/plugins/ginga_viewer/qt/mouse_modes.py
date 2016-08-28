from __future__ import absolute_import, division, print_function

import os
import sys

from ginga import cmap as ginga_cmap

from qtpy import QtGui, QtWidgets
from glue.config import toolbar_mode
from glue.viewers.common.qt.mode import CheckableMode, NonCheckableMode
from glue.plugins.ginga_viewer.qt.utils import cmap2pixmap, ginga_graphic_to_roi
from glue.utils import nonpartial
from glue.plugins.tools.spectrum_tool.qt import SpectrumTool
from glue.plugins.tools.pv_slicer.qt import PVSlicerMode


# Find out location of ginga module so we can some of its icons
GINGA_HOME = os.path.split(sys.modules['ginga'].__file__)[0]
GINGA_ICON_DIR = os.path.join(GINGA_HOME, 'icons')

print("RUNNING GINGA MODES")

@toolbar_mode
class RectangleROIMode(CheckableMode):

    mode_id = 'Ginga rectangle'
    icon = 'glue_square'
    tooltip = 'Rectangle'

    def activate(self):
        self.viewer._set_roi_mode('rectangle', True)

    def deactivate(self):
        self.viewer._set_roi_mode('rectangle', False)


@toolbar_mode
class CircleROIMode(CheckableMode):

    mode_id = 'Ginga circle'
    icon = 'glue_circle'
    tooltip = 'Circle'

    def activate(self):
        self.viewer._set_roi_mode('circle', True)

    def deactivate(self):
        self.viewer._set_roi_mode('circle', False)


@toolbar_mode
class PolygonROIMode(CheckableMode):

    mode_id = 'Ginga polygon'
    icon = 'glue_lasso'
    tooltip = 'Polygon'

    def activate(self):
        self.viewer._set_roi_mode('polygon', True)

    def deactivate(self):
        self.viewer._set_roi_mode('polygon', False)


@toolbar_mode
class PanMode(CheckableMode):

    mode_id = 'Ginga pan'
    icon = 'glue_move'
    tooltip = 'Pan'

    def activate(self):
        self.viewer.mode_cb('pan', True)

    def deactivate(self):
        self.viewer.mode_cb('pan', False)


@toolbar_mode
class FreePanMode(CheckableMode):

    mode_id = 'Ginga free pan'
    icon = os.path.join(GINGA_ICON_DIR, 'hand_48.png')
    tooltip = 'Free Pan'

    def activate(self):
        self.viewer.mode_cb('freepan', True)

    def deactivate(self):
        self.viewer.mode_cb('freepan', False)


@toolbar_mode
class RotateMode(CheckableMode):

    mode_id = 'Ginga rotate'
    icon = os.path.join(GINGA_ICON_DIR, 'rotate_48.png')
    tooltip = 'Rotate'

    def activate(self):
        self.viewer.mode_cb('rotate', True)

    def deactivate(self):
        self.viewer.mode_cb('rotate', False)


@toolbar_mode
class ContrastMode(CheckableMode):

    mode_id = 'Ginga contrast'
    icon = 'glue_contrast'
    tooltip = 'Rotate'

    def activate(self):
        self.viewer.mode_cb('contrast', True)

    def deactivate(self):
        self.viewer.mode_cb('contrast', False)


@toolbar_mode
class CutsMode(CheckableMode):

    mode_id = 'Ginga cuts'
    icon = os.path.join(GINGA_ICON_DIR, 'cuts_48.png')
    tooltip = 'Cuts'

    def activate(self):
        self.viewer.mode_cb('cuts', True)

    def deactivate(self):
        self.viewer.mode_cb('cuts', False)


class ColormapAction(QtWidgets.QAction):

    def __init__(self, label, cmap, parent):
        super(ColormapAction, self).__init__(label, parent)
        self.cmap = cmap
        pm = cmap2pixmap(cmap)
        self.setIcon(QtGui.QIcon(pm))


@toolbar_mode
class ColormapMode(NonCheckableMode):

    icon = 'glue_rainbow'
    mode_id = 'Ginga colormap'
    action_text = 'Set color scale'
    tool_tip = 'Set color scale'

    def menu_actions(self):
        acts = []
        for label in ginga_cmap.get_names():
            cmap = ginga_cmap.get_cmap(label)
            a = ColormapAction(label, cmap, self.viewer)
            a.triggered.connect(nonpartial(self.viewer.client.set_cmap, cmap))
            acts.append(a)
        return acts


class GingaMode(CheckableMode):
    label = None
    icon = None
    shape = 'polygon'
    color = 'red'
    linestyle = 'solid'

    def __init__(self, viewer):

        super(CheckableMode, self).__init__(viewer)

        self.parent_canvas = self.viewer.canvas
        self._shape_tag = None

        self.parent_canvas.add_callback('draw-event', self._extract_callback)
        self.parent_canvas.add_callback('draw-down', self._clear_shape_cb)

    def _set_path_mode(self, enable):
        self.parent_canvas.enable_draw(True)
        self.parent_canvas.draw_context = self

        self.parent_canvas.set_drawtype(self.shape, color=self.color,
                                        linestyle=self.linestyle)
        bm = self.parent_canvas.get_bindmap()
        bm.set_mode('draw', mode_type='locked')

    def _clear_shape_cb(self, *args):
        try:
            self.parent_canvas.deleteObjectByTag(self._shape_tag)
        except:
            pass

    _clear_path = _clear_shape_cb


@toolbar_mode
class GingaPVSlicerMode(GingaMode):

    icon = 'glue_slice'
    mode_id = 'Ginga slicer'
    action_text = 'Slice Extraction'
    tool_tip = 'Extract a slice from an arbitrary path'

    shape = 'path'

    def _extract_callback(self, canvas, tag):
        if self.parent_canvas.draw_context is not self:
            return

        self._shape_tag = tag
        obj = self.parent_canvas.getObjectByTag(tag)
        vx, vy = zip(*obj.points)
        return self._build_from_vertices(vx, vy)

    _build_from_vertices = PVSlicerMode._build_from_vertices


@toolbar_mode
class GingaSpectrumMode(GingaMode, SpectrumTool):

    icon = 'glue_spectrum'
    mode_id = 'Ginga spectrum'
    action_text = 'Spectrum'
    tool_tip = 'Extract a spectrum from the selection'

    shape = 'rectangle'

    def __init__(self, widget=None):
        GingaMode.__init__(self, widget.canvas)
        SpectrumTool.__init__(self, widget, self)
        self._release_callback = self._update_profile
        self._move_callback = self._move_profile

    def _extract_callback(self, canvas, tag):
        if self.parent_canvas.draw_context is not self:
            return
        self._shape_tag = tag
        obj = self.parent_canvas.getObjectByTag(tag)
        roi = ginga_graphic_to_roi(obj)
        return self._update_from_roi(roi)

    def clear(self):
        pass
