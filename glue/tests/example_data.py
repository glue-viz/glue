import numpy as np

from glue.core.component import Component, CategoricalComponent
from glue.core.data import Data


def test_histogram_data():
    data = Data(label="Test Data")
    comp_a = Component(np.random.uniform(size=500))
    comp_b = Component(np.random.normal(size=500))
    data.add_component(comp_a, 'uniform')
    data.add_component(comp_b, 'normal')
    return data


def test_data():
    data = Data(label="Test Data 1")
    data2 = Data(label="Teset Data 2")

    comp_a = Component(np.array([1, 2, 3]))
    comp_b = Component(np.array([1, 2, 3]))
    comp_c = Component(np.array([2, 4, 6]))
    comp_d = Component(np.array([1, 3, 5]))
    data.add_component(comp_a, 'a')
    data.add_component(comp_b, 'b')
    data2.add_component(comp_c, 'c')
    data2.add_component(comp_d, 'd')
    return data, data2


def test_categorical_data():

    data = Data(label="Test Cat Data 1")
    data2 = Data(label="Teset Cat Data 2")

    comp_x1 = CategoricalComponent(np.array(['a', 'a', 'b']))
    comp_y1 = Component(np.array([1, 2, 3]))
    comp_x2 = CategoricalComponent(np.array(['c', 'a', 'b']))
    comp_y2 = Component(np.array([1, 3, 5]))
    data.add_component(comp_x1, 'x1')
    data.add_component(comp_y1, 'y1')
    data2.add_component(comp_x2, 'x2')
    data2.add_component(comp_y2, 'y2')
    return data, data2


def test_image():
    data = Data(label="Test Image")
    comp_a = Component(np.ones((25, 25)))
    data.add_component(comp_a, 'test_1')
    comp_b = Component(np.zeros((25, 25)))
    data.add_component(comp_b, 'test_2')
    return data


def test_cube():
    data = Data(label="Test Cube")
    comp_a = Component(np.ones((16, 16, 16)))
    data.add_component(comp_a, 'test_3')
    return data
