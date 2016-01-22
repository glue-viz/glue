from __future__ import absolute_import, division, print_function

import pytest

from ..simpleforms import IntOption, FloatOption, BoolOption


class Stub(object):
    int_opt = IntOption(min=0, max=10, default=3)
    float_opt = FloatOption(min=1, max=2, default=1.5)
    bool_opt = BoolOption()


class TestSimpleForms(object):

    def test_get_set_int(self):
        assert Stub.int_opt.min == 0
        assert Stub.int_opt.max == 10
        assert Stub().int_opt == 3

    def test_get_set_bool(self):
        s = Stub()
        assert s.bool_opt is False
        s.bool_opt = True
        assert s.bool_opt

    def test_get_set_float(self):

        s = Stub()
        assert s.float_opt == 1.5

        s.float_opt = 1
        assert s.float_opt == 1.0
        assert isinstance(s.float_opt, float)

    def test_invalid_int(self):

        s = Stub()
        s.int_opt = 4

        with pytest.raises(ValueError):
            s.int_opt = -1

        with pytest.raises(ValueError):
            s.int_opt = 11

        with pytest.raises(ValueError):
            s.int_opt = 2.5

    def test_invalid_float(self):
        s = Stub()

        with pytest.raises(ValueError):
            s.float_opt = -0.1

        with pytest.raises(ValueError):
            s.float_opt = 10.1

    def test_invalid(self):
        s = Stub()

        with pytest.raises(ValueError):
            s.bool_opt = 3
