import pytest


def pytest_addoption(parser):
    parser.addoption('--qtapi', action='store_true',
                     help='Run Qt Binding tests')


def pytest_runtest_setup(item):
    if 'qtapi' in item.keywords and not item.config.getvalue('qtapi'):
        pytest.skip("Need --qtapi option to run")
