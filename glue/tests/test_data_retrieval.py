import unittest

import numpy as np

import glue

class TestDataRetrieval(unittest.TestCase):
    def setUp(self):
        data1 = glue.Data()
        comp1 = glue.data.Component(np.arange(5))
        id1 = data1.add_component(comp1, 'comp_1')
        comp2 = glue.data.Component(np.arange(5)*2)
        id2 = data1.add_component(comp2, 'comp_2')

        data2 = glue.Data()
        comp3 = glue.data.Component(np.arange(5) * 3)
        id3 = data2.add_component(comp3, 'comp_3')
        comp4 = glue.data.Component(np.arange(5) * 4)
        id4 = data2.add_component(comp4, 'comp_4')

        self.data = [data1, data2]
        self.components = [comp1, comp2, comp3, comp4]
        self.component_ids = [id1, id2, id3, id4]

    def test_direct_get(self):
        self.assertIs(self.data[0][self.component_ids[0]],
                      self.components[0].data)
        self.assertIs(self.data[0][self.component_ids[1]],
                      self.components[1].data)
        self.assertIs(self.data[1][self.component_ids[2]],
                      self.components[2].data)
        self.assertIs(self.data[1][self.component_ids[3]],
                      self.components[3].data)

if __name__ == "__main__":
    unittest.main()




