import numpy as np
from numpy.testing import assert_allclose

from glue.external.echo import CallbackProperty, ListCallbackProperty
from glue.core import Data, DataCollection

from .test_state import clone
from ..state_objects import State, StateAttributeLimitsHelper, StateAttributeSingleValueHelper


class SimpleTestState(State):
    a = CallbackProperty()
    b = CallbackProperty()
    flat = ListCallbackProperty()
    nested = ListCallbackProperty()


def test_state_serialization():

    state1 = SimpleTestState()
    state1.a = 2
    state1.b = 'hello'
    state1.flat = [1, 3, 4]

    sub_state = SimpleTestState()
    sub_state.a = 3
    sub_state.b = 'blah'
    sub_state.flat = [1, 2]
    sub_state.nested = []

    state1.nested = [1, 3, sub_state]

    state2 = clone(state1)

    assert state2.a == 2
    assert state2.b == 'hello'
    assert state2.flat == [1, 3, 4]
    assert state2.nested[0:2] == [1, 3]
    assert state2.nested[2].a == 3
    assert state2.nested[2].b == 'blah'
    assert state2.nested[2].flat == [1, 2]
    assert state2.nested[2].nested == []


class TestStateAttributeLimitsHelper():

    def setup_method(self, method):

        self.data = Data(x=np.linspace(-100, 100, 10000),
                         y=np.linspace(2, 3, 10000), label='test_data')

        self.data_collection = DataCollection([self.data])

        class SimpleState(State):

            layer = CallbackProperty()
            comp = CallbackProperty()
            lower = CallbackProperty()
            upper = CallbackProperty()
            log = CallbackProperty(False)
            scale = CallbackProperty(100)

        self.state = SimpleState()

        self.helper = StateAttributeLimitsHelper(self.state, attribute='comp',
                                                 lower='lower', upper='upper',
                                                 percentile='scale', log='log')
        self.state.data = self.data
        self.state.comp = self.data.id['x']

        self.x_id = self.data.visible_components[0]
        self.y_id = self.data.visible_components[1]

    def test_minmax(self):
        assert self.helper.lower == -100
        assert self.helper.upper == +100

    def test_change_attribute(self):
        self.helper.attribute = self.y_id
        assert self.helper.lower == 2
        assert self.helper.upper == 3
        self.helper.attribute = self.x_id
        assert self.helper.lower == -100
        assert self.helper.upper == +100

    def test_change_percentile(self):

        # Changing scale mode updates the limits
        self.helper.percentile = 99.5
        assert_allclose(self.helper.lower, -99.5)
        assert_allclose(self.helper.upper, +99.5)
        self.helper.percentile = 99
        assert_allclose(self.helper.lower, -99)
        assert_allclose(self.helper.upper, +99)
        self.helper.percentile = 90
        assert_allclose(self.helper.lower, -90)
        assert_allclose(self.helper.upper, +90)

        # When switching to custom, the last limits are retained
        self.helper.percentile = "Custom"
        assert_allclose(self.helper.lower, -90)
        assert_allclose(self.helper.upper, +90)

    def test_percentile_cached(self):
        # Make sure that if we change scale and change attribute, the scale
        # modes are cached on a per-attribute basis.
        self.helper.percentile = 99.5
        self.state.comp = self.y_id
        assert self.helper.percentile == 100
        self.helper.percentile = 99
        self.state.comp = self.x_id
        assert self.helper.percentile == 99.5
        self.state.comp = self.y_id
        assert self.helper.percentile == 99

    def test_flip_button(self):

        self.helper.flip_limits()

        assert self.helper.lower == +100
        assert self.helper.upper == -100

        # Make sure that values were re-cached when flipping
        self.state.comp = self.y_id
        assert self.helper.lower == 2
        assert self.helper.upper == 3
        self.state.comp = self.x_id
        assert self.helper.lower == +100
        assert self.helper.upper == -100

    def test_manual_edit(self):

        # Make sure that values are re-cached when edited manually
        self.helper.percentile = "Custom"
        self.state.lower = -122
        self.state.upper = 234
        self.helper.log = True
        assert self.helper.lower == -122
        assert self.helper.upper == 234
        assert self.helper.log
        self.state.comp = self.y_id
        assert self.helper.lower == 2
        assert self.helper.upper == 3
        assert not self.helper.log
        self.state.comp = self.x_id
        assert self.helper.lower == -122
        assert self.helper.upper == 234
        assert self.helper.log


class TestStateAttributeSingleValueHelper():

    def setup_method(self, method):

        self.data = Data(x=np.linspace(-100, 30, 9999),
                         y=np.linspace(2, 3, 9999), label='test_data')

        self.data_collection = DataCollection([self.data])

        class SimpleState(State):

            layer = CallbackProperty()
            comp = CallbackProperty()
            val = CallbackProperty()

        self.state = SimpleState()

        self.helper = StateAttributeSingleValueHelper(self.state, attribute='comp',
                                                      function=np.nanmedian, value='val')

        self.state.data = self.data

        self.state.comp = self.data.id['x']

        self.x_id = self.data.visible_components[0]
        self.y_id = self.data.visible_components[1]

    def test_value(self):
        assert self.helper.value == -35.

    def test_change_attribute(self):
        self.helper.attribute = self.y_id
        assert self.helper.value == 2.5
        self.helper.attribute = self.x_id
        assert self.helper.value == -35

    def test_manual_edit(self):
        self.state.val = 42.
        assert self.helper.value == 42
        self.state.comp = self.y_id
        assert self.helper.value == 2.5
        self.state.comp = self.x_id
        assert self.helper.value == 42


def test_limits_helper_initial_values():

    # Regression test for a bug that occurred if the limits cache was empty
    # but some attributes were set to values - in this case we don't want to
    # override the existing values.

    data = Data(x=np.linspace(-100, 100, 10000),
                y=np.linspace(2, 3, 10000), label='test_data')

    class SimpleState(State):

        layer = CallbackProperty()
        comp = CallbackProperty()
        lower = CallbackProperty()
        upper = CallbackProperty()

    state = SimpleState()
    state.lower = 1
    state.upper = 2
    state.comp = data.id['x']

    helper = StateAttributeLimitsHelper(state, attribute='comp',
                                        lower='lower', upper='upper')

    assert helper.lower == 1
    assert helper.upper == 2
