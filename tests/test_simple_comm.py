def load_tests(loader, tests, ignore):
    from doctest import DocTestSuite, NORMALIZE_WHITESPACE
    import unittest
    import simple_comm
    tests.addTests(DocTestSuite(simple_comm, 
                                optionflags=NORMALIZE_WHITESPACE))
    return tests


