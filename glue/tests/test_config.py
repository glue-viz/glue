from ..config import link_function, data_factory


def test_add_link_default():
    @link_function(info='maps x to y', output_labels=['y'])
    def foo(x):
        return 3
    val = (foo, 'maps x to y', ['y'], 'General')
    assert val in link_function


def test_add_data_factory():
    @data_factory('XYZ file', "*txt")
    def foo(x):
        pass
    assert (foo, 'XYZ file', '*txt', 0, False) in data_factory
