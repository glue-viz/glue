from __future__ import absolute_import, division, print_function

import pkgutil

import numpy as np

import glue


def test_histogram_data():
    data = glue.core.data.Data(label="Test Data")
    comp_a = glue.core.data.Component(np.random.uniform(size=500))
    comp_b = glue.core.data.Component(np.random.normal(size=500))
    data.add_component(comp_a, 'uniform')
    data.add_component(comp_b, 'normal')
    return data


def test_data():
    data = glue.core.data.Data(label="Test Data 1")
    data2 = glue.core.data.Data(label="Test Data 2")

    comp_a = glue.core.data.Component(np.array([1, 2, 3]))
    comp_b = glue.core.data.Component(np.array([1, 2, 3]))
    comp_c = glue.core.data.Component(np.array([2, 4, 6]))
    comp_d = glue.core.data.Component(np.array([1, 3, 5]))
    data.add_component(comp_a, 'a')
    data.add_component(comp_b, 'b')
    data2.add_component(comp_c, 'c')
    data2.add_component(comp_d, 'd')
    return data, data2


def test_categorical_data():

    data = glue.core.data.Data(label="Test Cat Data 1")
    data2 = glue.core.data.Data(label="Test Cat Data 2")

    comp_x1 = glue.core.data.CategoricalComponent(np.array(['a', 'a', 'b']))
    comp_y1 = glue.core.data.Component(np.array([1, 2, 3]))
    comp_x2 = glue.core.data.CategoricalComponent(np.array(['c', 'a', 'b']))
    comp_y2 = glue.core.data.Component(np.array([1, 3, 5]))
    data.add_component(comp_x1, 'x1')
    data.add_component(comp_y1, 'y1')
    data2.add_component(comp_x2, 'x2')
    data2.add_component(comp_y2, 'y2')
    return data, data2


def test_date_data():
    data = glue.core.data.Data(label="Test Time Data 1")
    data2 = glue.core.data.Data(label="Test Time Data 2")

    dates1 = np.array(['2006-01-13', '2007-07-13', '2010-08-13'], dtype='datetime64')
    dates2 = np.array(['2010-12-14', '2013-11-20', '2014-12-25'], dtype='datetime64')

    comp_dates1 = glue.core.data.Component(dates1)
    comp_y1 = glue.core.data.Component(np.array([1, 3, 9]))
    comp_dates2 = glue.core.data.Component(dates2)
    comp_y2 = glue.core.data.Component(np.array([2, 4, 8]))

    data.add_component(comp_dates1, 'dates1')
    data.add_component(comp_y1, 'y1')
    data2.add_component(comp_dates2, 'dates2')
    data2.add_component(comp_y2, 'y2')
    return data, data2

def test_image():
    data = glue.core.data.Data(label="Test Image")
    comp_a = glue.core.data.Component(np.ones((25, 25)))
    data.add_component(comp_a, 'test_1')
    comp_b = glue.core.data.Component(np.zeros((25, 25)))
    data.add_component(comp_b, 'test_2')
    return data


def test_cube():
    data = glue.core.data.Data(label="Test Cube")
    comp_a = glue.core.data.Component(np.ones((16, 16, 16)))
    data.add_component(comp_a, 'test_3')
    return data
