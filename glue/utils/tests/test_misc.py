from ..misc import as_variable_name


def test_as_variable_name():
    def check(input, expected):
        assert as_variable_name(input) == expected

    tests = [('x', 'x'),
             ('x2', 'x2'),
             ('2x', '_2x'),
             ('x!', 'x_'),
             ('x y z', 'x_y_z'),
             ('_XY', '_XY')
             ]
    for input, expected in tests:
        yield check, input, expected
