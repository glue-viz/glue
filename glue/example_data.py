import glue
import pkgutil

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
    data.components['main'] = comp
    data.components['invert'] = comp2
    data.shape = comp.data.shape
    data.ndim = len(data.shape)
    s = glue.subset.RoiSubset(data, label="Subset")
    return data, s


def simple_cube():
    data = glue.io.extract_data_fits('../examples/cps_12co_05.fits')
    comp = glue.data.Component(data['PRIMARY'])
    comp2 = glue.data.Component(data['PRIMARY'] * -1)
    data = glue.Data(label="Dummy Cube")
    data.components['main'] = comp
    data.components['invert'] = comp2
    data.shape = comp.data.shape
    data.ndim = len(data.shape)
    s = glue.subset.RoiSubset(data, label="Subset")
    return data, s

