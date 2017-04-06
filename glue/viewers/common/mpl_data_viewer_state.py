from glue.core.state_objects import State

from glue.external.echo import CallbackProperty, ListCallbackProperty


class MatplotlibDataViewerState(State):

    x_min = CallbackProperty()
    x_max = CallbackProperty()

    y_min = CallbackProperty()
    y_max = CallbackProperty()

    log_x = CallbackProperty()
    log_y = CallbackProperty()

    layers = ListCallbackProperty()
