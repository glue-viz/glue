def pytest_addoption(parser):
    parser.addoption("--no-optional-skip", action="store_true", default=False,
                     help="don't skip any tests with optional dependencies")

