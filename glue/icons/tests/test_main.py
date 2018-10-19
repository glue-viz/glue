import os
from .. import icon_path


def test_icon_path():

    path = icon_path('glue_replace')
    assert os.path.exists(path)

    path = icon_path('glue_replace', icon_format='png')
    assert os.path.exists(path)

    path = icon_path('glue_replace', icon_format='svg')
    assert os.path.exists(path)
