from __future__ import absolute_import, division, print_function

from glue.viewers.common.qt.data_viewer import DataViewer
from glue.viewers.common.qt.mpl_widget import MplWidget
from glue.viewers.common.viz_client import init_mpl
from glue.external.echo import add_callback
from glue.utils import nonpartial, avoid_circular
from glue.viewers.common.qt.mpl_toolbar import MatplotlibViewerToolbar
from glue.core import message as msg

__all__ = ['MatplotlibDataViewer']


class MatplotlibDataViewer(DataViewer):

    _toolbar_cls = MatplotlibViewerToolbar

    def __init__(self, session, parent=None):

        super(MatplotlibDataViewer, self).__init__(session, parent)

        # Use MplWidget to set up a Matplotlib canvas inside the Qt window
        self.mpl_widget = MplWidget()
        self.setCentralWidget(self.mpl_widget)

        # TODO: shouldn't have to do this
        self.central_widget = self.mpl_widget

        self.figure, self._axes = init_mpl(self.mpl_widget.canvas.fig)

        # Set up the state which will contain everything needed to represent
        # the current state of the viewer
        self.viewer_state = self._state_cls()
        self.viewer_state.data_collection = session.data_collection

        # Set up the options widget, which will include options that control the
        # viewer state
        self.options = self._options_cls(viewer_state=self.viewer_state,
                                         session=session)

        add_callback(self.viewer_state, 'x_min', nonpartial(self.limits_to_mpl))
        add_callback(self.viewer_state, 'x_max', nonpartial(self.limits_to_mpl))
        add_callback(self.viewer_state, 'y_min', nonpartial(self.limits_to_mpl))
        add_callback(self.viewer_state, 'y_max', nonpartial(self.limits_to_mpl))

        self.axes.callbacks.connect('xlim_changed', nonpartial(self.limits_from_mpl))
        self.axes.callbacks.connect('ylim_changed', nonpartial(self.limits_from_mpl))

        self.axes.set_autoscale_on(False)

    def update_labels(self):
        if self.viewer_state.xatt is not None:
            self.axes.set_xlabel(self.viewer_state.xatt[0])
        if self.viewer_state.yatt is not None:
            self.axes.set_ylabel(self.viewer_state.yatt[0])

    @avoid_circular
    def limits_from_mpl(self):
        # TODO: delay callbacks here
        self.viewer_state.x_min, self.viewer_state.x_max = self.axes.get_xlim()
        self.viewer_state.y_min, self.viewer_state.y_max = self.axes.get_ylim()

    @avoid_circular
    def limits_to_mpl(self):
        self.axes.set_xlim(self.viewer_state.x_min, self.viewer_state.x_max)
        self.axes.set_ylim(self.viewer_state.y_min, self.viewer_state.y_max)
        self.axes.figure.canvas.draw()

    # TODO: shouldn't need this!
    @property
    def axes(self):
        return self._axes

    def add_data(self, data):

        # Create layer artist and add to container
        layer = self._data_artist_cls(data, self._axes, self.viewer_state)
        self._layer_artist_container.append(layer)
        layer.update()

        return True

    def add_subset(self, subset):

        # Create scatter layer artist and add to container
        layer = self._subset_artist_cls(subset, self._axes, self.viewer_state)
        self._layer_artist_container.append(layer)
        layer.update()

        return True

    def _add_subset(self, message):
        self.add_subset(message.subset)

    def _update_subset(self, message):
        if message.subset in self._layer_artist_container:
            for layer_artist in self._layer_artist_container[message.subset]:
                layer_artist.update()
            self.axes.figure.canvas.draw()

    def _remove_subset(self, message):
        if message.subset in self._layer_artist_container:
            self._layer_artist_container.pop(message.subset)
            self.axes.figure.canvas.draw()

    def options_widget(self):
        return self.options

    def register_to_hub(self, hub):

        super(MatplotlibDataViewer, self).register_to_hub(hub)

        def subset_has_data(x):
            return x.sender.data in self._layer_artist_container.layers

        def has_data(x):
            return x.sender in self._layer_artist_container.layers

        hub.subscribe(self, msg.SubsetCreateMessage,
                      handler=self._add_subset,
                      filter=subset_has_data)

        hub.subscribe(self, msg.SubsetUpdateMessage,
                      handler=self._update_subset,
                      filter=subset_has_data)

        hub.subscribe(self, msg.SubsetDeleteMessage,
                      handler=self._remove_subset,
                      filter=subset_has_data)

        hub.subscribe(self, msg.NumericalDataChangedMessage,
                      handler=self._update_subset,
                      filter=has_data)

        # hub.subscribe(self, msg.ComponentsChangedMessage,
        #               handler=self._update_data,
        #               filter=has_data)
