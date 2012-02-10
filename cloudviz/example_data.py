import cloudviz as cv

def pipe():
    data = cv.data.TabularData(label="Pipe YSOs")
    data2 = cv.data.TabularData(label="Pipe Cores")
    data.read_data('../examples/pipe_yso.txt', type='ascii', delimiter='\t', data_start = 2)
    data2.read_data('../examples/pipe_cores.vot')
    s = cv.subset.RoiSubset(data, label="YSO subset")
    s2 = cv.subset.RoiSubset(data2, label="Core Subset")

    return data, data2, s, s2
