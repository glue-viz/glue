from __future__ import absolute_import, division, print_function

import numpy as np

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QPainter, QTransform, QPen
from qtpy.QtWidgets import (QGraphicsView, QGraphicsScene, QApplication,
                            QGraphicsTextItem, QGraphicsEllipseItem,
                            QGraphicsLineItem)

from glue.utils.qt import mpl_to_qt_color, qt_to_mpl_color

COLOR_SELECTED = (0.2, 0.9, 0.2)
COLOR_CONNECTED = (0.6, 0.9, 0.9)
COLOR_DISCONNECTED = (0.9, 0.6, 0.6)


def get_pen(color, linewidth=1):
    color = mpl_to_qt_color(color)
    return QPen(color, linewidth, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)


class Edge(QGraphicsLineItem):

    def __init__(self, node_source, node_dest):
        self.linewidth = 3
        self.node_source = node_source
        self.node_dest = node_dest
        super(Edge, self).__init__(0, 0, 1, 1)
        self.setZValue(5)
        self.color = '0.5'

    def update_position(self):
        x0, y0 = self.node_source.node_position
        x1, y1 = self.node_dest.node_position
        self.setLine(x0, y0, x1, y1)

    @property
    def color(self):
        return qt_to_mpl_color(self.pen().color())

    @color.setter
    def color(self, value):
        self.setPen(get_pen(value, self.linewidth))

    def add_to_scene(self, scene):
        scene.addItem(self)

    def remove_from_scene(self, scene):
        scene.removeItem(self)

    def contains(self, point):
        return super(Edge, self).contains(self.mapFromScene(point))


class DataNode:

    def __init__(self, data, radius=15):

        self.data = data

        # Add circular node
        self.node = QGraphicsEllipseItem(0, 0, 1, 1)

        # Set radius
        self.radius = radius

        # Add text label
        self.label = QGraphicsTextItem(data.label)
        font = self.label.font()
        font.setPointSize(10)
        self.label.setFont(font)

        # Add line between label and node
        self.line1 = QGraphicsLineItem(0, 0, 1, 1)
        self.line2 = QGraphicsLineItem(0, 0, 1, 1)

        self.node.setZValue(20)
        self.label.setZValue(10)
        self.line1.setZValue(10)
        self.line2.setZValue(10)

        self.line1.setPen(get_pen('0.5'))
        self.line2.setPen(get_pen('0.5'))

        self.color = '0.8'

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value
        self.node.setRect(-value, -value, 2 * value, 2 * value)

    def contains(self, point):

        # Check label
        if self.label.contains(self.label.mapFromScene(point)):
            return True

        # Check node
        if self.node.contains(self.node.mapFromScene(point)):
            return True

        return False

    def update(self):
        self.node.update()

    def add_to_scene(self, scene):
        scene.addItem(self.node)
        scene.addItem(self.label)
        scene.addItem(self.line1)
        scene.addItem(self.line2)

    def remove_from_scene(self, scene):
        scene.removeItem(self.node)
        scene.removeItem(self.label)
        scene.removeItem(self.line1)
        scene.removeItem(self.line2)

    @property
    def node_position(self):
        pos = self.node.pos()
        return pos.x(), pos.y()

    @node_position.setter
    def node_position(self, value):
        self.node.setPos(value[0], value[1])
        self.update_lines()

    @property
    def label_position(self):
        pos = self.label.pos()
        return pos.x(), pos.y()

    @label_position.setter
    def label_position(self, value):
        self.label.setPos(value[0], value[1])
        self.update_lines()

    def update_lines(self):
        x0, y0 = self.label_position
        x2, y2 = self.node_position
        x1 = 0.5 * (x0 + x2)
        y1 = y0
        self.line1.setLine(x0, y0, x1, y1)
        self.line2.setLine(x1, y1, x2, y2)

    @property
    def color(self):
        return qt_to_mpl_color(self.node.brush().color())

    @color.setter
    def color(self, value):
        self.node.setBrush(mpl_to_qt_color(value))


def get_connections(dc_links):
    links = set()
    for link in dc_links:
        to_id = link.get_to_id()
        for from_id in link.get_from_ids():
            data1 = from_id.parent
            data2 = to_id.parent
            if data1 is data2:
                continue
            if (data1, data2) not in links and (data2, data1) not in links:
                links.add((data1, data2))
    return links


def layout_simple_circle(nodes, edges, center=None, radius=None, reorder=True):

    # Place nodes around a circle

    if reorder:
        nodes[:] = order_nodes_by_connections(nodes, edges)

    for i, node in enumerate(nodes):
        angle = 2 * np.pi * i / len(nodes)
        nx = radius * np.cos(angle) + center[0]
        ny = radius * np.sin(angle) + center[1]
        node.node_position = nx, ny


def order_nodes_by_connections(nodes, edges):

    search_nodes = list(nodes)
    sorted_nodes = []

    while len(search_nodes) > 0:

        lengths = []
        connections = []

        for node in search_nodes:
            direct, indirect = find_connections(node, search_nodes, edges)
            connections.append((indirect, direct))
            lengths.append((len(indirect), len(direct)))

        m = max(lengths)

        for i in range(len(lengths)):
            if lengths[i] == m:
                for node in connections[i][0] + connections[i][1]:
                    if node not in sorted_nodes:
                        sorted_nodes.append(node)

        search_nodes = [node for node in nodes if node not in sorted_nodes]

    return sorted_nodes


class DataGraphWidget(QGraphicsView):

    selection_changed = Signal()

    def __init__(self, parent=None):

        super(DataGraphWidget, self).__init__(parent=parent)

        # Set up scene

        self.scene = QGraphicsScene(self)
        self.scene.setItemIndexMethod(QGraphicsScene.NoIndex)
        self.scene.setSceneRect(0, 0, 800, 300)

        self.setScene(self.scene)

        self.setWindowTitle("Glue data graph")

        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)

        self.selection_level = 0

    def resizeEvent(self, event):
        self.scene.setSceneRect(0, 0, self.width(), self.height())
        self.relayout(reorder=False)

    def relayout(self, reorder=True):

        # Update radius
        for node in self.nodes:
            node.radius = self.height() / 30.

        layout_simple_circle(self.nodes, self.edges,
                             center=(self.width() / 2, self.height() / 2),
                             radius=self.height() / 3, reorder=reorder)

        # Update edge positions
        for edge in self.edges:
            edge.update_position()

        # Set up labels
        self.left_nodes = [node for node in self.nodes if node.node_position[0] < self.width() / 2]
        self.left_nodes = sorted(self.left_nodes, key=lambda x: x.node_position[1], reverse=True)

        self.right_nodes = [node for node in self.nodes if node not in self.left_nodes]
        self.right_nodes = sorted(self.right_nodes, key=lambda x: x.node_position[1], reverse=True)

        for i, node in enumerate(self.left_nodes):
            y = self.height() - (i + 1) / (len(self.left_nodes) + 1) * self.height()
            node.label_position = self.width() / 2 - self.height() / 2, y

        for i, node in enumerate(self.right_nodes):
            y = self.height() - (i + 1) / (len(self.right_nodes) + 1) * self.height()
            node.label_position = self.width() / 2 + self.height() / 2, y

    def set_data_collection(self, data_collection):

        # Get data and initialize nodes
        self.data_to_nodes = dict((data, DataNode(data)) for data in data_collection)
        self.nodes = list(self.data_to_nodes.values())

        # Get links and set up edges
        self.edges = [Edge(self.data_to_nodes[data1], self.data_to_nodes[data2])
                      for data1, data2 in get_connections(data_collection.external_links)]

        # Figure out positions
        self.relayout()

        # Add nodes and edges to graph

        for node in self.nodes:
            node.add_to_scene(self.scene)

        for edge in self.edges:
            edge.add_to_scene(self.scene)

        self.text_adjusted = False

        self.selected_edge = None
        self.selected_node1 = None
        self.selected_node2 = None

    def set_links(self, links):

        for edge in self.edges:
            edge.remove_from_scene(self.scene)

        self.edges = [Edge(self.data_to_nodes[data1], self.data_to_nodes[data2])
                      for data1, data2 in get_connections(links)]

        for edge in self.edges:
            edge.update_position()

        for edge in self.edges:
            edge.add_to_scene(self.scene)

        self._update_selected_edge()

        self._update_selected_colors()

    def paintEvent(self, event):

        super(DataGraphWidget, self).paintEvent(event)

        if not self.text_adjusted:

            for node in self.nodes:

                width = node.label.boundingRect().width()
                height = node.label.boundingRect().height()

                transform = QTransform()
                if node in self.left_nodes:
                    transform.translate(-width, -height / 2)
                else:
                    transform.translate(0, -height / 2)

                node.label.setTransform(transform)

            self.text_adjusted = True

    def manual_select(self, data1=None, data2=None):
        if data1 is None and data2 is not None:
            data1, data2 = data2, data1
        if data2 is not None:
            self.selection_level = 2
        elif data1 is not None:
            self.selection_level = 1
        else:
            self.selection_level = 0
        self.selected_node1 = self.data_to_nodes.get(data1, None)
        self.selected_node2 = self.data_to_nodes.get(data2, None)
        self._update_selected_edge()
        self._update_selected_colors()

    def find_object(self, event):
        for obj in list(self.nodes) + self.edges:
            if obj.contains(event.localPos()):
                return obj

    def mouseMoveEvent(self, event):

        # TODO: Don't update until the end
        # TODO: Only select object on top

        selected = self.find_object(event)

        if selected is None:

            if self.selection_level == 0:
                self.selected_node1 = None
                self.selected_node2 = None
                self._update_selected_edge()
            elif self.selection_level == 1:
                self.selected_node2 = None
                self._update_selected_edge()

        elif isinstance(selected, DataNode):

            if self.selection_level == 0:
                self.selected_node1 = selected
                self.selected_node2 = None
            elif self.selection_level == 1:
                if selected is not self.selected_node1:
                    self.selected_node2 = selected
                    self._update_selected_edge()

        elif isinstance(selected, Edge):

            if self.selection_level == 0:
                self.selected_edge = selected
                self.selected_node1 = selected.node_source
                self.selected_node2 = selected.node_dest

        self._update_selected_colors()

        self.selection_changed.emit()

    def mousePressEvent(self, event):

        # TODO: Don't update until the end
        # TODO: Only select object on top

        selected = self.find_object(event)

        if selected is None:

            self.selection_level = 0
            self.selected_node1 = None
            self.selected_node2 = None

            self._update_selected_edge()

        elif isinstance(selected, DataNode):

            if self.selection_level == 0:
                self.selected_node1 = selected
                self.selection_level += 1
            elif self.selection_level == 1:
                if selected is self.selected_node1:
                    self.selected_node1 = None
                    self.selection_level = 0
                else:
                    self.selected_node2 = selected
                    self.selection_level = 2
            elif self.selection_level == 2:
                if selected is self.selected_node2:
                    self.selected_node2 = None
                    self.selection_level = 1
                else:
                    self.selected_node1 = selected
                    self.selected_node2 = None
                    self.selection_level = 1

            self._update_selected_edge()

        elif isinstance(selected, Edge):

            if self.selected_edge is selected and self.selection_level == 2:
                self.selected_edge = None
                self.selected_node1 = None
                self.selected_node2 = None
                self.selection_level = 0
            else:
                self.selected_edge = selected
                self.selected_node1 = selected.node_source
                self.selected_node2 = selected.node_dest
                self.selection_level = 2

        self.mouseMoveEvent(event)

    def _update_selected_edge(self):
        for edge in self.edges:
            if (edge.node_source is self.selected_node1 and edge.node_dest is self.selected_node2 or
                    edge.node_source is self.selected_node2 and edge.node_dest is self.selected_node1):
                self.selected_edge = edge
                break
        else:
            self.selected_edge = None

    def _update_selected_colors(self):

        colors = {}

        if self.selected_node1 is not None and self.selection_level < 2:

            direct, indirect = find_connections(self.selected_node1, self.nodes, self.edges)

            for node in self.nodes:
                if node in direct or node in indirect:
                    colors[node] = COLOR_CONNECTED
                else:
                    colors[node] = COLOR_DISCONNECTED

            for edge in self.edges:
                if (edge.node_source is self.selected_node1 or
                        edge.node_dest is self.selected_node1):
                    colors[edge] = COLOR_CONNECTED

        if self.selected_edge is not None:
            colors[self.selected_edge] = COLOR_SELECTED

        if self.selected_node1 is not None:
            colors[self.selected_node1] = COLOR_SELECTED

        if self.selected_node2 is not None:
            colors[self.selected_node2] = COLOR_SELECTED

        self.set_colors(colors)

    def set_colors(self, colors):

        for obj in list(self.nodes) + self.edges:
            default_color = '0.8' if isinstance(obj, DataNode) else '0.5'
            obj.color = colors.get(obj, default_color)
            obj.update()


def find_connections(node, nodes, edges):

    direct = [node]
    indirect = []
    current = direct
    connected = [node]

    changed = True
    while changed:
        changed = False
        for edge in edges:
            source = edge.node_source
            dest = edge.node_dest
            if source in connected and dest not in connected:
                current.append(dest)
                changed = True
            if dest in connected and source not in connected:
                current.append(source)
                changed = True
        current = indirect
        connected.extend(current)
    return direct, indirect


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)

    from glue.core.state import load
    dc = load('links.glu')

    widget = DataGraphWidget(dc)
    widget.show()

    sys.exit(app.exec_())
