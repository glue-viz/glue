import unittest

import numpy as np

from glue.data import ComponentID, Data, Component
from glue.component_link import ComponentLink

class TestComponentLink(unittest.TestCase):

    def toy_data(self):
        data = Data()
        from_comp = Component( np.array([1,2,3]))
        to_comp = Component( np.array([4,5,6]))
        return data, from_comp, to_comp

    def test_valid_init(self):
        data, from_, to_ = self.toy_data()
        link = ComponentLink([from_], to_)

    def test_valid_init_using(self):
        data, from_, to_ = self.toy_data()
        using = lambda x: x
        link = ComponentLink([from_], to_, using)

    def test_invalid_init_multi_from_no_using(self):
        data, from_, to_ = self.toy_data()
        using = lambda x: x
        self.assertRaises(TypeError,
                          ComponentLink,
                          [from_, from_],
                          to_)

    def test_invalid_init_scalar_from(self):
        data, from_, to_ = self.toy_data()
        self.assertRaises(TypeError,
                          ComponentLink,
                          from_, to_)

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

        self.assertIs(link.get_from_ids()[0], from_id)
        self.assertIs(link.get_to_id(), to_id)
        self.assertIs(link.get_using(), using)

    def test_str(self):
        """ str method returns without error """
        data, from_, to_ = self.toy_data()
        from_id = data.add_component(from_, 'from_label')
        to_id = ComponentID('to_label')
        link = ComponentLink([from_id], to_id)
        str(link)
        link = ComponentLink([from_id], to_id, using=lambda x:3*x)
        str(link)

    def test_repr(self):
        """ repr method returns without error """
        data, from_, to_ = self.toy_data()
        from_id = data.add_component(from_, 'from_label')
        to_id = ComponentID('to_label')
        link = ComponentLink([from_id], to_id)
        repr(link)



if __name__ == "__main__":
    unittest.main()