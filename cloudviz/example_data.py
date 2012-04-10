import cloudviz as cv

def pipe():
    data = cv.data.TabularData(label="Pipe YSOs")
    data2 = cv.data.TabularData(label="Pipe Cores")
    data.read_data('../examples/pipe_yso.txt', type='ascii',
                   delimiter='\t', data_start = 2)
    s = cv.subset.RoiSubset(data, label="YSO subset")
    data2.read_data('../examples/pipe_cores.vot', type='vo')
    s2 = cv.subset.RoiSubset(data2, label="Core Subset")

    return data, data2, s, s2

def simple_image():
    data = cv.io.extract_data_fits('../examples/Pipe.fits')
    comp = cv.data.Component(data['PRIMARY'])
    comp2 = cv.data.Component(data['PRIMARY'] * -1)
    data = cv.Data(label="Pipe Extinction")
    data.components['main'] = comp
    data.components['invert'] = comp2
    data.shape = comp.data.shape
    data.ndim = len(data.shape)
    s = cv.subset.RoiSubset(data, label="Subset")
    return data, s


def simple_cube():
    data = cv.io.extract_data_fits('../examples/cps_12co_05.fits')
    comp = cv.data.Component(data['PRIMARY'])
    comp2 = cv.data.Component(data['PRIMARY'] * -1)
    data = cv.Data(label="Dummy Cube")
    data.components['main'] = comp
    data.components['invert'] = comp2
    data.shape = comp.data.shape
    data.ndim = len(data.shape)
    s = cv.subset.RoiSubset(data, label="Subset")
    return data, s

