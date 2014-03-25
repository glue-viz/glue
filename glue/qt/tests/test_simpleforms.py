from ..simpleforms import build_form_item, FloatOption, IntOption, BoolOption


class Stub(object):
    i = IntOption(label="int", min=0, max=3, default=2)
    f = FloatOption(label="x", min=0, max=1, default=0.5)
    b = BoolOption(label="y", default=True)


class TestBuildFormItem(object):

    def test_int(self):
        s = Stub()
        w = build_form_item(s, 'i')
        assert w.label == "int"
        assert w.widget.value() == 2
        assert w.widget.minimum() == 0
        assert w.widget.maximum() == 3
        assert w.value == 2

    def test_float(self):
        s = Stub()
        w = build_form_item(s, 'f')
        assert w.label == "x"
        assert w.value == 0.5
        assert w.widget.minimum() == 0
        assert w.widget.maximum() == 1

    def test_bool(self):
        s = Stub()
        w = build_form_item(s, 'b')

        assert w.label == 'y'
        assert w.value is True
        assert w.widget.isChecked()
