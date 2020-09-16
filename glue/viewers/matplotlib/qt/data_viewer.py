from qtpy.QtCore import QTimer

from glue.core.message import ComputationStartedMessage
from glue.viewers.common.qt.data_viewer import DataViewer
from glue.viewers.matplotlib.qt.widget import MplWidget
from glue.viewers.matplotlib.mpl_axes import init_mpl
from glue.utils import defer_draw, decorate_all_methods
from glue.viewers.matplotlib.state import MatplotlibDataViewerState
from glue.viewers.matplotlib.viewer import MatplotlibViewerMixin

# The following import is required to register the viewer tools
from glue.viewers.matplotlib.qt import toolbar  # noqa

__all__ = ['MatplotlibDataViewer']


@decorate_all_methods(defer_draw)
class MatplotlibDataViewer(MatplotlibViewerMixin, DataViewer):

    _state_cls = MatplotlibDataViewerState

    tools = ['mpl:home', 'mpl:pan', 'mpl:zoom']
    subtools = {'save': ['mpl:save']}

    def __init__(self, session, parent=None, wcs=None, state=None, projection=None):

        super(MatplotlibDataViewer, self).__init__(session, parent=parent, state=state)

        # Use MplWidget to set up a Matplotlib canvas inside the Qt window
        self.mpl_widget = MplWidget()
        self.setCentralWidget(self.mpl_widget)

        # TODO: shouldn't have to do this
        self.central_widget = self.mpl_widget

        self.figure, self.axes = init_mpl(self.mpl_widget.canvas.fig, wcs=wcs, projection=projection)

        MatplotlibViewerMixin.setup_callbacks(self)

        self.central_widget.resize(600, 400)
        self.resize(self.central_widget.size())

        self._monitor_computation = QTimer()
        self._monitor_computation.setInterval(500)
        self._monitor_computation.timeout.connect(self._update_computation)

    def _update_computation(self, message=None):

        # If we get a ComputationStartedMessage and the timer isn't currently
        # active, then we start the timer but we then return straight away.
        # This is to avoid showing the 'Computing' message straight away in the
        # case of reasonably fast operations.
        if isinstance(message, ComputationStartedMessage):
            if not self._monitor_computation.isActive():
                self._monitor_computation.start()
            return

        for layer_artist in self.layers:
            if layer_artist.is_computing:
                self.loading_rectangle.set_visible(True)
                text = self.loading_text.get_text()
                if text.count('.') > 2:
                    text = 'Computing'
                else:
                    text += '.'
                self.loading_text.set_text(text)
                self.loading_text.set_visible(True)
                self.redraw()
                return

        self.loading_rectangle.set_visible(False)
        self.loading_text.set_visible(False)
        self.redraw()

        # If we get here, the computation has stopped so we can stop the timer
        self._monitor_computation.stop()
