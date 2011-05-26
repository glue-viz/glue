from cloudviz.io import extract_data_fits
from cloudviz.tree import DendroMerge


def read():
    file = 'dendro_oph.fits'
    data = extract_data_fits(file, use_hdu=[0, 1])
    m = extract_data_fits(file, use_hdu=[2])
    merge_list = m['CLUSTERS']
    merge_list = merge_list[(merge_list.shape[0] + 1) / 2:]
    tree = DendroMerge(merge_list, data['INDEX_MAP'])

    march_prefix(tree)

    print "Newick String:"
    print tree.to_newick()


def march_prefix(tree, depth=0):
    print ''.join([' |'] * depth) + ("%3.3i:" % tree.id)
    for c in tree.children:
        march_prefix(c, depth=depth + 1)

if __name__ == "__main__":
    read()
