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
    data2 = glue.core.data.Data(label="Teset Data 2")

    comp_a = glue.core.data.Component(np.array([1, 2, 3]))
    comp_b = glue.core.data.Component(np.array([1, 2, 3]))
    comp_c = glue.core.data.Component(np.array([2, 4, 6]))
    comp_d = glue.core.data.Component(np.array([1, 3, 5]))
    data.add_component(comp_a, 'a')
    data.add_component(comp_b, 'b')
    data2.add_component(comp_c, 'c')
    data2.add_component(comp_d, 'd')
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


def pipe():

    # terrible. Must fix
    ysos = pkgutil.get_data(__name__, 'examples/pipe_yso.txt')
    cores = pkgutil.get_data(__name__, 'examples/pipe_cores.vot')
    with open('.__junk1', 'w') as out:
        out.write(ysos)
    with open('.__junk2', 'w') as out:
        out.write(cores)

    data = glue.core.data.TabularData(label="Pipe YSOs")
    data2 = glue.core.data.TabularData(label="Pipe Cores")

    data.read_data('.__junk1',
                   type='ascii',
                   delimiter='\t', data_start=2)
    s = glue.Subset(data, label="YSO subset")
    data2.read_data('.__junk2', type='vo')

    s2 = glue.Subset(data2, label="Core Subset")

    return data, data2, s, s2


def simple_image():
    data = glue.core.data.GriddedData(label='Pipe Extinction')
    data.read_data('../../tests/examples/Pipe.fits')
    return data


def simple_cube():
    data = glue.core.data.GriddedData(label="Dummy Cube")
    data.read_data('../../tests/examples/cps_12co_05.fits')
    return data
