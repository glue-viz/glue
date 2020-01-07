from qtpy import QtWidgets, QtGui, QtCore


class HtmlItemDelegate(QtWidgets.QStyledItemDelegate):
    """
    An item delegate that can be used for e.g. QTreeView, QTreeWidget,
    QListView or QListWidget. This will automatically interpret any HTML that
    is inside the items in these views/widgets.

    This is more efficient than using e.g. QLabel instances embedded in the
    views/widgets, and is required for horizontal scroll bars to appear
    correctly.
    """

    # Implementation adapted based on solutions presented on StackOverflow:
    # https://stackoverflow.com/questions/1956542/how-to-make-item-view-render-rich-html-text-in-qt

    def paint(self, painter, option, index):

        options = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(options, index)

        painter.save()

        doc = QtGui.QTextDocument()
        doc.setTextWidth(options.rect.width())

        text_option = QtGui.QTextOption()
        text_option.setWrapMode(QtGui.QTextOption.NoWrap)
        text_option.setAlignment(options.displayAlignment)
        doc.setDefaultTextOption(text_option)

        doc.setHtml(options.text)

        options.text = ""
        options.widget.style().drawControl(QtWidgets.QStyle.CE_ItemViewItem, options, painter)

        iconSize = options.icon.actualSize(options.rect.size())
        painter.translate(options.rect.left() + iconSize.width(), options.rect.top())
        clip = QtCore.QRectF(0, 0, options.rect.width() + iconSize.width(), options.rect.height())

        doc.drawContents(painter, clip)

        painter.restore()

    def sizeHint(self, option, index):

        options = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(options, index)

        doc = QtGui.QTextDocument()

        text_option = QtGui.QTextOption()
        text_option.setWrapMode(QtGui.QTextOption.NoWrap)
        doc.setDefaultTextOption(text_option)

        doc.setHtml(options.text)
        doc.setTextWidth(options.rect.width())

        return QtCore.QSize(doc.idealWidth(), doc.size().height())
