import pytest

from astropy.visualization import LinearStretch, LogStretch

from glue.core.state_objects import State
from glue.viewers.common.stretch_state_mixin import StretchStateMixin


class ExampleStateWithStretch(State, StretchStateMixin):
    pass


def test_not_set_up():
    state = ExampleStateWithStretch()
    with pytest.raises(Exception, match="setup_stretch_callback has not been called"):
        state.stretch_object


class TestStretchStateMixin:
    def setup_method(self, method):
        self.state = ExampleStateWithStretch()
        self.state.setup_stretch_callback()

    def test_defaults(self):
        assert self.state.stretch == "linear"
        assert len(self.state.stretch_parameters) == 0
        assert isinstance(self.state.stretch_object, LinearStretch)

    def test_change_stretch(self):
        self.state.stretch = "log"
        assert self.state.stretch == "log"
        assert len(self.state.stretch_parameters) == 0
        assert isinstance(self.state.stretch_object, LogStretch)

    def test_invalid_parameter(self):
        with pytest.raises(
            ValueError, match="Stretch object LinearStretch has no attribute foo"
        ):
            self.state.stretch_parameters["foo"] = 1

    def test_set_parameter(self):
        self.state.stretch = "log"

        assert self.state.stretch_object.exp == 1000

        # Setting the stretch parameter 'exp' is synced with the stretch object attribute
        self.state.stretch_parameters["exp"] = 200
        assert self.state.stretch_object.exp == 200

        # Changing stretch resets the stretch parameter dictionary
        self.state.stretch = "linear"
        assert len(self.state.stretch_parameters) == 0

        # And there is no memory of previous parameters
        self.state.stretch = "log"
        assert self.state.stretch_object.exp == 1000
