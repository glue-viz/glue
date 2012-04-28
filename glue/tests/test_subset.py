import unittest

import numpy as np
from mock import MagicMock

import glue
from glue.subset import Subset, SubsetState
from glue.subset import CompositeSubsetState
from glue.subset import OrState
from glue.subset import AndState
from glue.subset import XorState
from glue.subset import InvertState

class TestSubset(unittest.TestCase):
    def setUp(self):
        self.data = None

    def test_subset_mask_wraps_state(self):
        s = Subset(self.data)
        state = MagicMock(spec=SubsetState)
        s.subset_state = state
        s.to_mask()
        state.to_mask.assert_called_once_with()

    def test_subset_index_wraps_state(self):
        s = Subset(self.data)
        state = MagicMock(spec=SubsetState)
        s.subset_state = state
        s.to_index_list()
        state.to_index_list.assert_called_once_with()

    def test_set_label(self):
        s = Subset(self.data, label = 'hi')
        self.assertEquals(s.label, 'hi')

    def test_set_color(self):
        s = Subset(self.data, color = 'blue')
        self.assertEquals(s.style.color, 'blue')

    def test_subset_state_reparented_on_assignment(self):
        s = Subset(self.data)
        state = SubsetState()
        s.subset_state = state
        self.assertTrue(state.parent is s)

    def test_paste_returns_copy_of_state(self):
        s = Subset(self.data)
        state1 = MagicMock(spec=SubsetState)
        state1_copy = MagicMock()
        state1.copy.return_value = state1_copy
        s.subset_state = state1

        s2 = Subset(self.data)

        s2.paste(s)
        self.assertTrue(s2.subset_state is state1_copy)

class TestSubsetStateCombinations(unittest.TestCase):
    def setUp(self):
        self.data = None

    def test_or(self):
        s1 = Subset(self.data)
        s2 = Subset(self.data)
        s3 = s1.subset_state | s2.subset_state
        self.assertTrue(isinstance(s3, OrState))

    def test_and(self):
        s1 = Subset(self.data)
        s2 = Subset(self.data)
        s3 = s1.subset_state & s2.subset_state
        self.assertTrue(isinstance(s3, AndState))

    def test_invert(self):
        s1 = Subset(self.data)
        s3 = ~s1.subset_state
        self.assertTrue(isinstance(s3, InvertState))

    def test_xor(self):
        s1 = Subset(self.data)
        s2 = Subset(self.data)
        s3 = s1.subset_state ^ s2.subset_state
        self.assertTrue(isinstance(s3, XorState))

class TestCompositeSubsetStates(unittest.TestCase):
    def setUp(self):
        self.sub1 = MagicMock(spec=SubsetState)
        self.sub1.to_mask.return_value = np.array([1, 1, 0, 0], dtype='bool')
        self.sub2 = MagicMock(spec=SubsetState)
        self.sub2.to_mask.return_value = np.array([1, 0, 1, 0], dtype='bool')

    def test_or(self):
        s3 = OrState(self.sub1, self.sub2)
        answer = s3.to_mask()
        expected = np.array([True, True, True, False])
        np.testing.assert_array_equal(answer, expected)

    def test_and(self):
        s3 = AndState(self.sub1, self.sub2)
        answer = s3.to_mask()
        expected = np.array([True, False, False, False])
        np.testing.assert_array_equal(answer, expected)

    def test_xor(self):
        s3 = XorState(self.sub1, self.sub2)
        answer = s3.to_mask()
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(answer, expected)

    def test_invert(self):
        s3 = InvertState(self.sub1)
        answer = s3.to_mask()
        expected = np.array([False, False, True, True])
        np.testing.assert_array_equal(answer, expected)

    def test_multicomposite(self):
        s3 = AndState(self.sub1, self.sub2)
        s4 = XorState(s3, self.sub1)
        answer = s4.to_mask()
        expected = np.array([False, True, False, False])


if __name__ == "__main__":
    unittest.main()