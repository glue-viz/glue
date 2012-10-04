#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103,W0612
import pytest

import numpy as np

from ..data import ComponentID, Data, Component
from ..component_link import ComponentLink


class TestComponentLink(object):

    def toy_data(self):
        data = Data()
        from_comp = Component(np.array([1, 2, 3]))
        to_comp = Component(np.array([4, 5, 6]))
        return data, from_comp, to_comp

    def test_valid_init(self):
        link = ComponentLink([ComponentID('from')], ComponentID('to'))

    def test_valid_init_using(self):
        data, from_, to_ = self.toy_data()
        using = lambda x: x
        link = ComponentLink([ComponentID('from')], ComponentID('to'), using)

    def test_invalid_init_multi_from_no_using(self):
        with pytest.raises(TypeError) as exc:
            ComponentLink([ComponentID('a'), ComponentID('b')],
                          ComponentID('c'))
        assert exc.value.args[0] == ("comp_from must have only 1 element, "
                                     "or a 'using' function must be provided")

    def test_invalid_init_scalar_from(self):
        with pytest.raises(TypeError) as exc:
            ComponentLink(ComponentID('from'), ComponentID('to'))
        assert exc.value.args[0].startswith("comp_from must be a list")

    def test_compute_direct(self):
        data, from_, to_ = self.toy_data()
        from_id = data.add_component(from_, 'from_label')
        to_id = ComponentID('to_label')
        link = ComponentLink([from_id], to_id)

        result = link.compute(data)
        expected = from_.data
        np.testing.assert_array_equal(result, expected)

    def test_compute_using(self):
        data, from_, to_ = self.toy_data()
        from_id = data.add_component(from_, 'from_label')
        to_id = ComponentID('to_label')
        using = lambda x: 3 * x
        link = ComponentLink([from_id], to_id, using)

        result = link.compute(data)
        expected = from_.data * 3
        np.testing.assert_array_equal(result, expected)

    def test_getters(self):
        data, from_, to_ = self.toy_data()
        from_id = data.add_component(from_, 'from_label')
        to_id = ComponentID('to_label')
        using = lambda x: 3 * x
        link = ComponentLink([from_id], to_id, using)

        assert link.get_from_ids()[0] is from_id
        assert link.get_to_id() is to_id
        assert link.get_using() is using

    def test_str(self):
        """ str method returns without error """
        data, from_, to_ = self.toy_data()
        from_id = data.add_component(from_, 'from_label')
        to_id = ComponentID('to_label')
        link = ComponentLink([from_id], to_id)
        str(link)
        link = ComponentLink([from_id], to_id, using=lambda x: 3 * x)
        str(link)

    def test_repr(self):
        """ repr method returns without error """
        data, from_, to_ = self.toy_data()
        from_id = data.add_component(from_, 'from_label')
        to_id = ComponentID('to_label')
        link = ComponentLink([from_id], to_id)
        repr(link)

    def test_type_check(self):
        """Should raise an exception if non ComponentIDs are passed as input"""
        cid = ComponentID('test')
        with pytest.raises(TypeError) as exc:
            ComponentLink([None], cid)
        assert exc.value.args[0] == \
            'from argument is not a list of ComponentIDs'

        with pytest.raises(TypeError) as exc:
            ComponentLink([cid], None)
        assert exc.value.args[0] == 'to argument is not a ComponentID'

        with pytest.raises(TypeError) as exc:
            ComponentLink([cid, None], None, using=lambda x, y: None)
        assert exc.value.args[0] == \
            'from argument is not a list of ComponentIDs'
