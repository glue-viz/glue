import pkgutil

import numpy as np

import glue

def test_data():
    data = glue.Data(label="Test Data 1")
    data.ndim = 1
    data.shape = (3,)
    data2 = glue.Data(label="Teset Data 2")
    data2.ndim = 1
    data2.shape = (3,)

    comp_a = glue.Component(np.array([1,2,3]))
    comp_b = glue.Component(np.array([1,2,3]))
    comp_c = glue.Component(np.array([2,4,6]))
    comp_d = glue.Component(np.array([1,3,5]))
    data.add_component(comp_a, 'a')
    data.add_component(comp_b, 'b')
    data2.add_component(comp_c, 'c')
    data2.add_component(comp_d, 'd')
    return data, data2

def pipe():

    # terrible. Must fix
    ysos = pkgutil.get_data(__name__, 'examples/pipe_yso.txt')
    cores = pkgutil.get_data(__name__, 'examples/pipe_cores.vot')
    with open('.__junk1','w') as out:
        out.write(ysos)
    with open('.__junk2', 'w') as out:
        out.write(cores)

    data = glue.data.TabularData(label="Pipe YSOs")
    data2 = glue.data.TabularData(label="Pipe Cores")

    data.read_data('.__junk1',
                   type='ascii',
                   delimiter='\t', data_start = 2)
    s = glue.subset.RoiSubset(data, label="YSO subset")
    data2.read_data('.__junk2', type='vo')

    s2 = glue.subset.RoiSubset(data2, label="Core Subset")

    return data, data2, s, s2

def simple_image():
    data = glue.io.extract_data_fits('../examples/Pipe.fits')
    comp = glue.data.Component(data['PRIMARY'])
    comp2 = glue.data.Component(data['PRIMARY'] * -1)
    data = glue.Data(label="Pipe Extinction")
    data.add_component(comp, 'main')
    data.add_component(comp2, 'invert')
    data.shape = comp.data.shape
    data.ndim = len(data.shape)
    s = glue.subset.RoiSubset(data, label="Subset")
    return data, s


def simple_cube():
    data = glue.io.extract_data_fits('../examples/cps_12co_05.fits')
    comp = glue.data.Component(data['PRIMARY'])
    comp2 = glue.data.Component(data['PRIMARY'] * -1)
    data = glue.Data(label="Dummy Cube")
    data.add_component(comp, 'main')
    data.add_component(comp2, 'invert')
    data.shape = comp.data.shape
    data.ndim = len(data.shape)
    s = glue.subset.RoiSubset(data, label="Subset")
    return data, s

