import glue
from glue.io import extract_data_fits


def data_dendro_cpp(file):
    """ Read a C++-generated dendrogram file into
    a glue data structure

    Parameters:
    -----------
    file: Name of a file to read

    Outputs:
    --------
    A glue data structure representing the file
    """

    data = extract_data_fits(file, use_hdu=[0, 1])
    m = extract_data_fits(file, use_hdu=[2])
    merge_list = m['CLUSTERS']
    merge_list = merge_list[(merge_list.shape[0] + 1) / 2:]

    im = data['INDEX_MAP']
    val = data['PRIMARY']

    c = glue.data.Component(val)

    result = glue.data.GriddedData()
    result.read_data(file, use_hdu=['PRIMARY', 'INDEX_MAP'])
    result.tree = glue.tree.DendroMerge(merge_list, index_map=im)
    return result
