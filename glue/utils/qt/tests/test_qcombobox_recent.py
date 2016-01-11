from ..qcombobox_recent import QComboBoxRecent


def _items_as_string(box):
    items = [box._itemText(i) for i in range(box._count())]
    print(items)
    items[items.index('Recent items')] = "/"
    items[items.index('All items')] = "/"
    return "".join(items)


def test_qcombobox_recent_event():
    """
    We need to check that the index changed event is intercepted and also offset
    """

    class EventHandler(object):

        def __init__(self, box):
            self.box = box

        def on_change(self, idx):
            self.index = self.box.currentIndex()

    box = QComboBoxRecent()

    handler = EventHandler(box)

    box.show()
    for letter in 'abcdefghij':
        box.addItem(letter)
    box.currentIndexChanged.connect(handler.on_change)

    box.setCurrentIndex(7)

    assert handler.index == 7
    assert box.itemText(handler.index) == 'h'

    box.setCurrentIndex(7)

    assert handler.index == 7
    assert box.itemText(handler.index) == 'h'


def test_qcombobox_recent():

    box = QComboBoxRecent()
    box.show()

    # Should only include 'Recent items' and 'All items'
    # assert _items_as_string(box) == "//"

    for letter in 'abcdefghij':
        box.addItem(letter)

    # Initial recent list is empty
    # assert _items_as_string(box) == "//abcdefghij"

    # Selecting 'c' puts it in the recent list
    box._setCurrentIndex(4)
    assert _items_as_string(box) == "/c/abcdefghij"
    assert box._currentIndex() == 1

    # Selecting 'c' again doesn't change anything
    box._setCurrentIndex(5)
    assert _items_as_string(box) == "/c/abcdefghij"
    assert box._currentIndex() == 1

    # Selecting 'c' from recents list doesn't change anything
    box._setCurrentIndex(1)
    assert _items_as_string(box) == "/c/abcdefghij"
    assert box._currentIndex() == 1

    # Selecting 'a' puts it in the recent list
    box._setCurrentIndex(3)
    assert _items_as_string(box) == "/ac/abcdefghij"
    assert box._currentIndex() == 1

    # Selecting 'c' from recents list puts it back at front
    box._setCurrentIndex(2)
    assert _items_as_string(box) == "/ca/abcdefghij"
    assert box._currentIndex() == 1

    # Selecting 'a' from main list swaps recents list
    box._setCurrentIndex(4)
    assert _items_as_string(box) == "/ac/abcdefghij"
    assert box._currentIndex() == 1

    # Selecting 'f' from main list adds to recent list
    box._setCurrentIndex(9)
    assert _items_as_string(box) == "/fac/abcdefghij"
    assert box._currentIndex() == 1

    # Selecting 'g' from main list adds to recent list
    box._setCurrentIndex(11)
    assert _items_as_string(box) == "/gfac/abcdefghij"
    assert box._currentIndex() == 1

    # Selecting 'f' from main list moves around recent list
    box._setCurrentIndex(11)
    assert _items_as_string(box) == "/fgac/abcdefghij"
    assert box._currentIndex() == 1

    # Selecting 'a' from recents list moves around recent list
    box._setCurrentIndex(3)
    assert _items_as_string(box) == "/afgc/abcdefghij"
    assert box._currentIndex() == 1

    # Selecting 'h' from main list adds to recent list
    box._setCurrentIndex(13)
    assert _items_as_string(box) == "/hafgc/abcdefghij"
    assert box._currentIndex() == 1

    # Selecting 'b' from main list adds to recent list and drops one from recent list
    box._setCurrentIndex(8)
    assert _items_as_string(box) == "/bhafg/abcdefghij"
    assert box._currentIndex() == 1

    # Check that setting to -1 works
    box._setCurrentIndex(-1)
    assert box._currentIndex() == -1
    assert box.currentIndex() == -1

    # Selecting 'd' from main list adds to recent list and drops one from
    # recent list
    box._setCurrentIndex(10)
    assert _items_as_string(box) == "/dbhaf/abcdefghij"
    assert box._currentIndex() == 1

    # Inserting an item adds it at the start of the full data not the recent
    # list
    box.insertItem(0, 'z')
    assert _items_as_string(box) == "/dbhaf/zabcdefghij"
    assert box._currentIndex() == 1
