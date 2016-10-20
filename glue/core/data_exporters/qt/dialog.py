from __future__ import absolute_import, division, print_function

from qtpy import compat
from glue import config


def export_data(data):

    exporters = {}
    for e in config.data_exporter:
        if e.extension == '':
            fltr = "{0} (*)".format(e.label)
        else:
            fltr = "{0} ({1})".format(e.label, ' '.join('*.' + ext for ext in e.extension))
        exporters[fltr] = e.function

    filters = ';;'.join(sorted(exporters))

    filename, fltr = compat.getsavefilename(caption="Choose an output filename",
                                            filters=filters)

    filename = str(filename)
    if not filename:
        return

    exporters[fltr](data, filename)
