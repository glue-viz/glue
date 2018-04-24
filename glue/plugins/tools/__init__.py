def setup():
    from glue.plugins.tools import python_export
    from glue.viewers.common.qt.data_viewer import DataViewer
    DataViewer.subtools['save'].append('save:python')
