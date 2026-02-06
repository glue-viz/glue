import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from glue.core.tests.test_state import clone

from ..viewer_state import ViewerState3D


class TestViewerState3D:

    def setup_method(self, method):
        self.state = ViewerState3D()

    def test_flip_x(self):
        self.state.set_limits(0, 10, 0, 20, 0, 30)
        self.state.flip_x()
        assert self.state.x_min == 10
        assert self.state.x_max == 0

    def test_flip_y(self):
        self.state.set_limits(0, 10, 0, 20, 0, 30)
        self.state.flip_y()
        assert self.state.y_min == 20
        assert self.state.y_max == 0

    def test_flip_z(self):
        self.state.set_limits(0, 10, 0, 20, 0, 30)
        self.state.flip_z()
        assert self.state.z_min == 30
        assert self.state.z_max == 0

    def test_aspect_no_native(self):
        self.state.native_aspect = False
        assert_array_equal(self.state.aspect, [1, 1, 1])

    def test_aspect_native_equal_ranges(self):
        self.state.native_aspect = True
        self.state.set_limits(0, 10, 0, 10, 0, 10)
        assert_array_equal(self.state.aspect, [1, 1, 1])

    def test_aspect_native_unequal_ranges(self):
        self.state.native_aspect = True
        self.state.set_limits(0, 10, 0, 20, 0, 30)
        aspect = self.state.aspect
        # x=10, y=20, z=30 -> ratios are 1:2:3 -> normalized to max gives 1/3:2/3:1
        assert_allclose(aspect[0], 1.0 / 3.0)
        assert_allclose(aspect[1], 2.0 / 3.0)
        assert_allclose(aspect[2], 1.0)

    def test_clip_limits_property(self):
        self.state.set_limits(1, 5, 2, 6, 3, 7)
        assert self.state.clip_limits == (1, 5, 2, 6, 3, 7)

    def test_serialization(self):
        self.state.x_min = -10
        self.state.x_max = 10
        self.state.y_min = -20
        self.state.y_max = 20
        self.state.z_min = -30
        self.state.z_max = 30
        self.state.x_stretch = 1.5
        self.state.y_stretch = 2.0
        self.state.z_stretch = 0.5
        self.state.visible_axes = False
        self.state.perspective_view = True
        self.state.clip_data = False
        self.state.native_aspect = True
        self.state.line_width = 2.5

        new_state = clone(self.state)

        assert new_state.x_min == -10
        assert new_state.x_max == 10
        assert new_state.y_min == -20
        assert new_state.y_max == 20
        assert new_state.z_min == -30
        assert new_state.z_max == 30
        assert new_state.x_stretch == 1.5
        assert new_state.y_stretch == 2.0
        assert new_state.z_stretch == 0.5
        assert new_state.visible_axes is False
        assert new_state.perspective_view is True
        assert new_state.clip_data is False
        assert new_state.native_aspect is True
        assert new_state.line_width == 2.5
