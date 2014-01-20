#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
import pytest

import numpy as np
import matplotlib.pyplot as plt
from mock import MagicMock

from ...tests import example_data
from ... import core

from ..beeswarm_client import BeeSwarmClient
from test_scatter_client import TestScatterClient

# share matplotlib instance, and disable rendering, for speed
FIGURE = plt.figure()
FIGURE.canvas.draw = lambda: 0
plt.close('all')


class TestBeeSwarmClient(TestScatterClient):
    """
    The BeeHive should be able to do everything Scatter does.
    """

    def setup_method(self, method):
        self.data = example_data.test_categorical_data()
        self.ids = [self.data[0].find_component_id('x1'),
                    self.data[0].find_component_id('y1'),
                    self.data[1].find_component_id('x2'),
                    self.data[1].find_component_id('y2')]
        self.roi_limits = (0.5, 0.5, 4, 4)
        self.roi_points = (np.array([1]), np.array([3]))
        self.hub = core.hub.Hub()
        self.collect = core.data_collection.DataCollection()

        FIGURE.clf()
        axes = FIGURE.add_subplot(111)
        self.client = BeeSwarmClient(self.collect, axes=axes)

        self.connect()

    def test_change_axis_labels(self):

        self.add_data()
        self.client._set_xydata('x', 'x1')
        nticks = [label.get_text() for label in self.client.axes.get_xticklabels()]
        assert nticks == ['a', 'b']

    def test_axis_labels_sync_with_setters(self):
        layer = self.add_data()
        self.client.xatt = self.ids[0]
        assert self.client.axes.get_xlabel() == self.ids[0].label
        self.client.yatt = self.ids[1]
        assert self.client.axes.get_ylabel() == self.ids[1].label
        nticks = [label.get_text() for label in self.client.axes.get_xticklabels()]
        assert nticks == ['a', 'b']
