import unittest

import numpy as np

import glue
from glue.util import *

class TestRelim(unittest.TestCase):
    pass

class TestGlue(unittest.TestCase):
    def test_1to1(self):
        d1 = glue.Data(label='d1')
        d2 = glue.Data(label='d2')
        c1 = glue.Component(np.zeros(3))
        c2 = glue.Component(np.zeros(3))
        id1 = d1.add_component(c1, 'label1')
        id2 = d2.add_component(c2, 'label2')

        glue_components_1to1(d1, id1, d2, id2)

        self.assertIn(id2, d1.derived_components)
        self.assertIn(id1, d2.derived_components)

if __name__ == "__main__":
    unittest.main()