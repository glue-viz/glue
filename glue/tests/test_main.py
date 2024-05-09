from unittest.mock import patch

from glue.main import load_plugins


def test_no_duplicate_loading(capsys):
    """
    Regression test for duplicated loading of plugins
    on subsequent calls of `load_plugins()` after initial
    glue-qt startup.

    """
    from glue.logger import logger

    with patch.object(logger, 'info') as info:
        load_plugins()

    for acall in info.call_args_list:
        if 'Loading plugin' in acall[0][0]:
            assert 'failed' in acall[0][0]
