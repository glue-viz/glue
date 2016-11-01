import numpy as np
from numpy.testing import assert_allclose

from glue.external.echo import CallbackProperty
from glue.core import Data, DataCollection

from ..state_objects import StateAttributeLimitsHelper


class TestStateAttributeLimitsHelper():

    def setup_method(self, method):

        self.data = Data(x=np.linspace(-100, 100, 10000),
                         y=np.linspace(2, 3, 10000), label='test_data')

        self.data_collection = DataCollection([self.data])

        class SimpleState(object):

            layer = CallbackProperty()
            comp = CallbackProperty()
            lower = CallbackProperty()
            upper = CallbackProperty()
            log = CallbackProperty()
            scale = CallbackProperty()

        self.state = SimpleState()

        self.helper = StateAttributeLimitsHelper(self.state, 'comp',
                                                 'lower', 'upper',
                                                 percentile='scale', vlog='log')

        self.state.data = self.data
        self.state.comp = (self.data.id['x'], self.data)

        self.x_id = self.data.visible_components[0]
        self.y_id = self.data.visible_components[1]

    def test_minmax(self):
        assert self.helper.vlo == -100
        assert self.helper.vhi == +100

    def test_change_attribute(self):
        self.helper.attribute = self.y_id, self.data
        assert self.helper.vlo == 2
        assert self.helper.vhi == 3
        self.helper.attribute = self.x_id, self.data
        assert self.helper.vlo == -100
        assert self.helper.vhi == +100

    def test_change_percentile(self):

        # Changing scale mode updates the limits
        self.helper.percentile = 99.5
        assert_allclose(self.helper.vlo, -99.5)
        assert_allclose(self.helper.vhi, +99.5)
        self.helper.percentile = 99
        assert_allclose(self.helper.vlo, -99)
        assert_allclose(self.helper.vhi, +99)
        self.helper.percentile = 90
        assert_allclose(self.helper.vlo, -90)
        assert_allclose(self.helper.vhi, +90)

        # When switching to custom, the last limits are retained
        self.helper.percentile = None
        assert_allclose(self.helper.vlo, -90)
        assert_allclose(self.helper.vhi, +90)

    def test_percentile_cached(self):
        # Make sure that if we change scale and change attribute, the scale
        # modes are cached on a per-attribute basis.
        self.helper.percentile = 99.5
        self.state.comp = self.y_id, self.data
        assert self.helper.percentile == 100
        self.helper.percentile = 99
        self.state.comp = self.x_id, self.data
        assert self.helper.percentile == 99.5
        self.state.comp = self.y_id, self.data
        assert self.helper.percentile == 99

    def test_flip_button(self):

        self.helper.flip_limits()

        assert self.helper.vlo == +100
        assert self.helper.vhi == -100

        # Make sure that values were re-cached when flipping
        self.state.comp = self.y_id, self.data
        assert self.helper.vlo == 2
        assert self.helper.vhi == 3
        self.state.comp = self.x_id, self.data
        assert self.helper.vlo == +100
        assert self.helper.vhi == -100

    def test_manual_edit(self):

        # Make sure that values are re-cached when edited manually
        self.helper.percentile = None
        self.state.lower = -122
        self.state.upper = 234
        self.helper.vlog = True
        assert self.helper.vlo == -122
        assert self.helper.vhi == 234
        assert self.helper.vlog
        self.state.comp = self.y_id, self.data
        assert self.helper.vlo == 2
        assert self.helper.vhi == 3
        assert not self.helper.vlog
        self.state.comp = self.x_id, self.data
        assert self.helper.vlo == -122
        assert self.helper.vhi == 234
        assert self.helper.vlog
