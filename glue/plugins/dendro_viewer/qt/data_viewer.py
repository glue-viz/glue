from __future__ import absolute_import, division, print_function

import numpy as np

from qtpy.QtWidgets import QMessageBox

from glue.viewers.matplotlib.qt.toolbar import MatplotlibViewerToolbar
from glue.core.edit_subset_mode import EditSubsetMode
from glue.core.roi import PointROI
from glue.core import command
from glue.core.subset import CategorySubsetState

from glue.plugins.dendro_viewer.dendro_helpers import _substructures
from glue.viewers.matplotlib.qt.data_viewer import MatplotlibDataViewer
from glue.plugins.dendro_viewer.layer_artist import DendrogramLayerArtist
from glue.plugins.dendro_viewer.qt.options_widget import DendrogramOptionsWidget
from glue.plugins.dendro_viewer.state import DendrogramViewerState
from glue.plugins.dendro_viewer.qt.layer_style_editor import DendrogramLayerStyleEditor

__all__ = ['DendrogramViewer']


class DendrogramViewer(MatplotlibDataViewer):

    LABEL = 'Dendrogram'

    _toolbar_cls = MatplotlibViewerToolbar
    _layer_style_widget_cls = DendrogramLayerStyleEditor
    _state_cls = DendrogramViewerState
    _options_cls = DendrogramOptionsWidget
    _data_artist_cls = DendrogramLayerArtist
    _subset_artist_cls = DendrogramLayerArtist

    tools = ['select:pick']

    def __init__(self, *args, **kwargs):
        super(DendrogramViewer, self).__init__(*args, **kwargs)
        self.axes.set_xticks([])
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['bottom'].set_visible(False)
        self.state.add_callback('_layout', self._update_limits)

    def _update_limits(self, layout):

        if self.state._layout is None:
            return

        x, y = self.state._layout.xy
        x, y = x[::3], y[::3]
        xlim = np.array([x.min(), x.max()])
        xpad = .05 * xlim.ptp()
        xlim[0] -= xpad
        xlim[1] += xpad

        ylim = np.array([y.min(), y.max()])
        if self.state.y_log:
            ylim = np.maximum(ylim, 1e-5)
            pad = 1.05 * ylim[1] / ylim[0]
            ylim[0] /= pad
            ylim[1] *= pad
        else:
            pad = .05 * ylim.ptp()
            ylim[0] -= pad
            ylim[1] += pad

        self.axes.set_xlim(*xlim)
        self.axes.set_ylim(*ylim)

    def initialize_toolbar(self):

        super(DendrogramViewer, self).initialize_toolbar()

        def on_move(mode):
            if mode._drag:
                self.apply_roi(mode.roi())

        self.toolbar.tools['select:pick']._move_callback = on_move

    def add_data(self, data):
        if data.ndim != 1:
            QMessageBox.critical(self, "Error", "Only 1-D data can be added to "
                                 "the dendrogram viewer (tried to add a {}-D "
                                 "dataset)".format(data.ndim),
                                 buttons=QMessageBox.Ok)
            return False
        return super(DendrogramViewer, self).add_data(data)

    # TODO: move some of the ROI stuff to state class?

    def apply_roi(self, roi):
        if len(self.layers) > 0:
            cmd = command.ApplyROI(data_collection=self._data,
                                   roi=roi, apply_func=self._apply_roi)
            self._session.command_stack.do(cmd)
        else:
            # Make sure we force a redraw to get rid of the ROI
            self.axes.figure.canvas.draw()

    def _apply_roi(self, roi):

        # TODO Does subset get applied to all data or just visible data?

        if self.state._layout is None:
            return

        if not roi.defined():
            return

        if isinstance(roi, PointROI):

            x, y = roi.x, roi.y

            xs, ys = self.state._layout.xy
            parent_ys = ys[1::3]
            xs, ys = xs[::3], ys[::3]

            delt = np.abs(x - xs)
            delt[y > ys] = np.nan
            delt[y < parent_ys] = np.nan

            if np.isfinite(delt).any():
                select = np.nanargmin(delt)
                if self.state.select_substruct:
                    parent = self.state.reference_data[self.state.parent_att]
                    select = _substructures(parent, select)
                select = np.asarray(select, dtype=np.int)
            else:
                select = np.array([], dtype=np.int)

            subset_state = CategorySubsetState(self.state.reference_data.pixel_component_ids[0],
                                               select)

        else:

            raise TypeError("Only PointROI selections are supported")

        mode = EditSubsetMode()
        mode.update(self._data, subset_state)

    # @staticmethod
    # def update_viewer_state(rec, context):
    #     return update_histogram_viewer_state(rec, context)
