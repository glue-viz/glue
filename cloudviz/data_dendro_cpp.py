import cloudviz as cv
from cloudviz.io import extract_data_fits


def data_dendro_cpp(file):
    """ Read a C++-generated dendrogram file into
    a cloudviz data structure

    Parameters:
    -----------
    file: Name of a file to read

    Outputs:
    --------
    A cloudviz data structure representing the file
    """

    data = extract_data_fits(file, use_hdu=[0, 1])
    m = extract_data_fits(file, use_hdu=[2])
    merge_list = m['CLUSTERS']
    merge_list = merge_list[(merge_list.shape[0] + 1) / 2:]

    im = data['INDEX_MAP']
    val = data['PRIMARY']

    c = cv.data.Component(val)

    result = cv.Data()
    result.components['PRIMARY'] = c
    result.shape = val.shape
    result.ndim = len(val.shape)
    result.tree = cv.tree.DendroMerge(merge_list, index_map=im)
    return result
