"""
Backward-compatibility shim. The package has been renamed to
:mod:`glue.plugins.tools.path_slicer`; this module re-exports the
public symbols under their old import paths so released versions of
downstream packages (e.g. glue-qt) keep working.
"""
from glue.plugins.tools.path_slicer.path_sliced_data import (  # noqa: F401
    PathSlicedData, sample_points)
from glue.plugins.tools.path_slicer.path_sliced_data_links import (  # noqa: F401
    PathRelativeLink, link_path_sliced_to_parent,
    link_path_sliced_pair_paths, link_path_sliced_group)
