from copy import deepcopy


def setup():

    from glue.plugins.tools import python_export  # noqa

    from glue.viewers.histogram.qt.data_viewer import HistogramViewer
    HistogramViewer.subtools = deepcopy(HistogramViewer.subtools)
    HistogramViewer.subtools['save'].append('save:python')

    from glue.viewers.image.qt.data_viewer import ImageViewer
    ImageViewer.subtools = deepcopy(ImageViewer.subtools)
    ImageViewer.subtools['save'].append('save:python')

    from glue.viewers.scatter.qt.data_viewer import ScatterViewer
    ScatterViewer.subtools = deepcopy(ScatterViewer.subtools)
    ScatterViewer.subtools['save'].append('save:python')

    from glue.viewers.profile.qt.data_viewer import ProfileViewer
    ProfileViewer.subtools = deepcopy(ProfileViewer.subtools)
    ProfileViewer.subtools['save'].append('save:python')
