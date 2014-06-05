from ..layout import Rectangle, snap_to_grid


class TestSnap(object):

    @staticmethod
    def check(input, expected, **kwargs):
        result = snap_to_grid(input, **kwargs)
        for i, e in zip(input, expected):
            assert result[i] == e

    def test_2x2(self):

        rs = [Rectangle(-.2, -.1, .45, .52),
              Rectangle(.52, -.23, .49, .49),
              Rectangle(0, .45, .51, .53),
              Rectangle(.50, .45, .51, .53)]

        ex = [Rectangle(0, 0, .5, .5),
              Rectangle(.5, 0, .5, .5),
              Rectangle(0, .5, .5, .5),
              Rectangle(.5, .5, .5, .5)]

        self.check(rs, ex)

    def test_1x2(self):

        rs = [Rectangle(-.2, -.2, .95, .48),
              Rectangle(0, .45, .51, .53),
              Rectangle(.50, .45, .51, .53)]

        ex = [Rectangle(0, 0, 1, .5),
              Rectangle(0, .5, .5, .5),
              Rectangle(.5, .5, .5, .5)]

        self.check(rs, ex)

    def test_1x3(self):

        rs = [Rectangle(-.02, -.2, .95, .48),
              Rectangle(0.1, .51, 0.32, .53),
              Rectangle(0.32, .49, .30, .53),
              Rectangle(0.7, .52, .40, .53)]

        ex = [Rectangle(0, 0, 1, .5),
              Rectangle(0, .5, 1 / 3., .5),
              Rectangle(1 / 3., .5, 1 / 3., .5),
              Rectangle(2 / 3., .5, 1 / 3., .5)]

        self.check(rs, ex)

    def test_padding_1x2(self):

        rs = [Rectangle(0, 0, 1, .5),
              Rectangle(0, .5, .5, .5),
              Rectangle(.5, .5, .5, .5)]
        ex = [Rectangle(.1, .1, .8, .3),
              Rectangle(.1, .6, .3, .3),
              Rectangle(.6, .6, .3, .3)]

        self.check(rs, ex, padding=0.1)
