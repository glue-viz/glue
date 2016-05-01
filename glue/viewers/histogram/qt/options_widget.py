import os

from glue.external.qt import QtGui

from glue.utils.qt.widget_properties import CurrentComboProperty, FloatLineProperty, ButtonProperty
from glue.utils.qt import load_ui
from glue.viewers.common.qt.attribute_limits_helper import AttributeLimitsHelper
from glue.core.qt.data_combo_helper import ComponentIDComboHelper

__all__ = ["HistogramOptionsWidget"]


class HistogramOptionsWidget(QtGui.QWidget):

    x_att = CurrentComboProperty('ui.combo_x_attribute')
    x_log = ButtonProperty('ui.button_x_log')
    x_min = FloatLineProperty('ui.value_x_min')
    x_max = FloatLineProperty('ui.value_x_max')

    y_normalized = ButtonProperty('ui.button_y_normalized')
    y_cumulative = ButtonProperty('ui.button_y_cumulative')
    y_log = ButtonProperty('ui.button_y_log')
    y_min = FloatLineProperty('ui.value_y_min')
    y_max = FloatLineProperty('ui.value_y_max')

    hidden = FloatLineProperty('ui.checkbox_hidden')

    def __init__(self, parent=None, data_collection=None):

        super(HistogramOptionsWidget, self).__init__(parent=parent)

        self.ui = load_ui('options_widget.ui', self,
                          directory=os.path.dirname(__file__))

        self.x_att_helper = ComponentIDComboHelper(self.ui.combo_x_attribute, data_collection)

        self.x_limits_helper = AttributeLimitsHelper(self.ui.combo_x_attribute,
                                                     self.ui.value_x_min,
                                                     self.ui.value_x_max,
                                                     flip_button=self.ui.button_flip_x,
                                                     log_button=self.ui.button_x_log)

    def append(self, data):
        self.x_att_helper.append(data)

    def remove(self, data):
        self.x_att_helper.remove(data)


if __name__ == "__main__":

    from glue.external.qt import get_qapp
    from glue.core import Data, DataCollection

    data1 = Data(x=[1,2,3], y=[4,5,6], z=[7,8,9])
    data2 = Data(a=[1,2,3], b=[4,5,6], c=[7,8,9])
    dc = DataCollection([data1, data2])

    app = get_qapp()

    widget = HistogramOptionsWidget(data_collection=dc)
    widget.append(data1)
    widget.append(data2)
    widget.show()
    widget.raise_()

    app.exec_()
