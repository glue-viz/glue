import os

from glue.core import data_factories as df
from glue.tests.helpers import requires_astropy


DATA = os.path.join(os.path.dirname(__file__), 'data')


@requires_astropy
def test_load_vot():
    # This checks that we can load a VO table which incidentally is a subset of
    # the one included in the tutorial.
    d_set = df.load_data(os.path.join(DATA, 'w5_subset.vot'))
    assert len(d_set.components) == 15
