"""
AnnotationCanvas.py — QGraphicsView-based annotation canvas with zoom, pan,
crosshair, grid overlay, integrated zoom view, and annotation visualization.
"""

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush, QCursor, QPainterPath
from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsTextItem,
    QGraphicsLineItem, QWidget, QVBoxLayout, QLabel, QFrame,
    QGraphicsItem
)

from .UITheme import (
    Colors, ANNOTATION_COLORS_QT, ANNOTATION_COLORS_BRIGHT,
    POINT_RADIUS, POINT_RADIUS_SELECTED, BADGE_FONT_SIZE,
    ZOOM_VIEW_SIZE
)


def cv2_to_qimage(cv_img):
    """Convert an OpenCV BGR image (numpy array) to QImage (RGB)."""
    if cv_img is None:
        return QImage()
    if len(cv_img.shape) == 2:
        # Grayscale
        h, w = cv_img.shape
        return QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
    h, w, ch = cv_img.shape
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


def cv2_to_qpixmap(cv_img):
    """Convert an OpenCV BGR image to QPixmap."""
    return QPixmap.fromImage(cv2_to_qimage(cv_img))


class DraggablePointItem(QGraphicsEllipseItem):
    """An interactive annotation point that can be dragged."""

    def __init__(self, x, y, r, color, idx, is_negative, canvas):
        super().__init__(-r, -r, r * 2, r * 2)
        self.idx = idx
        self.canvas = canvas

        self.setBrush(QBrush(color))
        self.setPen(QPen(QColor(Colors.TEXT_BRIGHT), 1.5))
        
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setCursor(QCursor(Qt.SizeAllCursor))

        self.badge = QGraphicsTextItem(str(idx + 1), self)
        self.badge.setDefaultTextColor(QColor(Colors.TEXT_BRIGHT))
        font = QFont("Segoe UI", BADGE_FONT_SIZE)
        font.setBold(True)
        self.badge.setFont(font)
        self.badge.setPos(r + 2, -r - 4)

        if is_negative:
            pen = QPen(QColor(Colors.ACCENT_RED), 2)
            self.line1 = QGraphicsLineItem(-r, -r, r, r, self)
            self.line1.setPen(pen)
            self.line2 = QGraphicsLineItem(-r, r, r, -r, self)
            self.line2.setPen(pen)

        self.setPos(x, y)

    def boundingRect(self):
        # Expand hit area by 6 pixels for easier clicking
        r = self.rect().width() / 2
        pad = 6
        return QRectF(-r - pad, -r - pad, r * 2 + pad * 2, r * 2 + pad * 2)

    def shape(self):
        path = QPainterPath()
        path.addEllipse(self.boundingRect())
        return path

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged and hasattr(self, 'idx'):
            if not self.canvas._is_redrawing:
                self.canvas.point_dragging.emit(self.idx, value.x(), value.y())
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        modifiers = event.modifiers()
        button = event.button()

        if button == Qt.RightButton and not (modifiers & Qt.ControlModifier):
            # Delete on plain Right Click
            self.canvas.point_deleted.emit(self.idx)
            event.accept()
        elif button == Qt.LeftButton and modifiers & Qt.ShiftModifier:
            # Move on Shift + Left Click
            self._drag_start_pos = self.pos()
            super().mousePressEvent(event)
        else:
            # Ignore others (like plain Left Click or Ctrl+Click on point)
            event.ignore()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() != Qt.RightButton and hasattr(self, '_drag_start_pos') and self.pos() != self._drag_start_pos:
            pos = self.pos()
            self.canvas.point_moved.emit(
                self.idx, 
                self._drag_start_pos.x(), 
                self._drag_start_pos.y(), 
                pos.x(), 
                pos.y()
            )



class ZoomViewWidget(QFrame):
    """Integrated corner zoom view showing a magnified area around the cursor."""

    def __init__(self, parent=None, size=ZOOM_VIEW_SIZE):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._size = size
        self._pixmap = None
        self.setStyleSheet(f"""
            ZoomViewWidget {{
                border: 2px solid {Colors.ZOOM_BORDER};
                border-radius: 4px;
                background-color: {Colors.BG_DARKEST};
            }}
        """)

    def update_zoom(self, source_pixmap, center_x, center_y, zoom_factor=4):
        """Update the zoom view to show area around (center_x, center_y)."""
        if source_pixmap is None or source_pixmap.isNull():
            return
        src_w = source_pixmap.width()
        src_h = source_pixmap.height()
        half_view = self._size // (2 * zoom_factor)
        x1 = max(int(center_x) - half_view, 0)
        y1 = max(int(center_y) - half_view, 0)
        x2 = min(int(center_x) + half_view, src_w)
        y2 = min(int(center_y) + half_view, src_h)
        if x2 <= x1 or y2 <= y1:
            return
        cropped = source_pixmap.copy(x1, y1, x2 - x1, y2 - y1)
        self._pixmap = cropped.scaled(
            self._size, self._size,
            Qt.KeepAspectRatio, Qt.FastTransformation
        )
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self._pixmap and not self._pixmap.isNull():
            x = (self.width() - self._pixmap.width()) // 2
            y = (self.height() - self._pixmap.height()) // 2
            painter.drawPixmap(x, y, self._pixmap)
            # Crosshair in zoom view
            pen = QPen(QColor(Colors.ACCENT_GREEN), 1, Qt.DashLine)
            painter.setPen(pen)
            cx, cy = self.width() // 2, self.height() // 2
            painter.drawLine(cx, 0, cx, self.height())
            painter.drawLine(0, cy, self.width(), cy)
        else:
            painter.setPen(QColor(Colors.TEXT_MUTED))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Image")
        painter.end()


class AnnotationCanvas(QGraphicsView):
    """Main annotation canvas with zoom, pan, crosshair, and annotation rendering.

    Signals:
        point_clicked(float, float, int): Emitted on left/right click with
            (x, y, button) where button is Qt.LeftButton or Qt.RightButton.
        mouse_moved(float, float): Emitted on mouse move with image coordinates.
    """

    point_clicked = pyqtSignal(float, float, int)
    mouse_moved = pyqtSignal(float, float)
    point_moved = pyqtSignal(int, float, float, float, float)
    point_dragging = pyqtSignal(int, float, float)
    point_deleted = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Canvas settings
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QBrush(QColor(Colors.CANVAS_BG)))
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt.CrossCursor))

        # Image item
        self._image_item = QGraphicsPixmapItem()
        self._scene.addItem(self._image_item)

        # Overlay items (managed manually)
        self._overlay_items = []  # Annotation point graphics
        self._crosshair_items = []
        self._grid_items = []
        self._skeleton_items = []

        # State
        self._show_crosshair = True
        self._show_grid = False
        self._zoom_level = 1.0
        self._is_panning = False
        self._is_redrawing = False
        self._pan_start = QPointF()
        self._source_pixmap = None  # Unmodified pixmap for zoom view

        # Zoom view (corner widget)
        self.zoom_view = ZoomViewWidget(self)
        self.zoom_view.move(8, 8)
        self.zoom_view.setVisible(True)

    def set_image(self, cv_img):
        """Set the canvas image from an OpenCV BGR numpy array."""
        if cv_img is None:
            return
        pixmap = cv2_to_qpixmap(cv_img)
        self._source_pixmap = pixmap
        self._image_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))

    def update_image(self, cv_img):
        """Update the displayed image without resetting scene rect (for mask overlays)."""
        if cv_img is None:
            return
        pixmap = cv2_to_qpixmap(cv_img)
        self._image_item.setPixmap(pixmap)

    def get_source_pixmap(self):
        """Get the current source pixmap."""
        return self._source_pixmap

    def fit_image(self):
        """Fit the image to the view."""
        if self._source_pixmap and not self._source_pixmap.isNull():
            self.fitInView(self._image_item, Qt.KeepAspectRatio)
            self._zoom_level = self.transform().m11()

    # ─── Annotation Rendering ────────────────────────────────────────────────

    def clear_annotations(self):
        """Remove all annotation overlay items from the scene."""
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()
        self._clear_skeleton()

    def draw_annotations(self, points, labels, pose_keypoints=None, pose_coords=None, pose_config=None):
        """Draw annotation points with numbered badges and class colors."""
        self._is_redrawing = True
        try:
            self.clear_annotations()
            for idx, (pt, lbl) in enumerate(zip(points, labels)):
                x, y = pt[0], pt[1]
                class_id = abs(lbl) // 1000
                is_negative = lbl < 0
                color = ANNOTATION_COLORS_QT.get(class_id, QColor(Colors.TEXT_PRIMARY))
                if is_negative:
                    color = QColor(Colors.ACCENT_RED)
                    
                is_visible = True
                if pose_coords and idx < len(pose_coords):
                    is_visible = pose_coords[idx].get('visible', True)
                    
                if not is_visible:
                    color.setAlpha(120)

                r = POINT_RADIUS
                item = DraggablePointItem(x, y, r, color, idx, is_negative, self)
                if not is_visible:
                    item.setOpacity(0.5)
                self._scene.addItem(item)
                self._overlay_items.append(item)

            # Draw skeleton connecting lines between consecutive annotation points.
            # Always shown when 2+ points exist so the user can see the structure.
            if len(points) > 1:
                self._draw_skeleton(points, labels, pose_coords)
        finally:
            self._is_redrawing = False

    def update_skeleton(self, points, labels=None, pose_coords=None):
        """Optimized skeleton update that doesn't clear points."""
        self._draw_skeleton(points, labels, pose_coords)

    def _draw_skeleton(self, points, labels, pose_coords=None):
        """Draw connecting lines between consecutive annotation points.

        In pose mode these are keypoint skeleton lines (dashed cyan).
        In segment mode these are simple connector lines to show the order
        points were placed, matching the legacy stable behaviour.
        """
        self._clear_skeleton()
        if len(points) < 2:
            return
        pen = QPen(QColor(Colors.ACCENT_CYAN), 1.5, Qt.DashLine)
        pen.setCosmetic(True)

        pen_invisible = QPen(QColor(Colors.ACCENT_CYAN), 1.5, Qt.DotLine)
        pen_invisible.setCosmetic(True)
        c = QColor(Colors.ACCENT_CYAN)
        c.setAlpha(60)
        pen_invisible.setColor(c)

        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            # If pose_coords provided, use visibility; otherwise always visible
            vis1 = pose_coords[i].get('visible', True) if pose_coords and i < len(pose_coords) else True
            vis2 = pose_coords[i+1].get('visible', True) if pose_coords and (i + 1) < len(pose_coords) else True

            p = pen if (vis1 and vis2) else pen_invisible

            line = self._scene.addLine(x1, y1, x2, y2, p)
            self._skeleton_items.append(line)

    def _clear_skeleton(self):
        for item in self._skeleton_items:
            self._scene.removeItem(item)
        self._skeleton_items.clear()

    # ─── Crosshair & Grid ────────────────────────────────────────────────────

    def set_show_crosshair(self, show):
        self._show_crosshair = show
        self.viewport().update()

    def set_show_grid(self, show):
        self._show_grid = show
        self.viewport().update()

    def toggle_crosshair(self):
        self._show_crosshair = not self._show_crosshair
        self.viewport().update()
        return self._show_crosshair

    def toggle_grid(self):
        self._show_grid = not self._show_grid
        self.viewport().update()
        return self._show_grid

    # ─── Events ──────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        modifiers = event.modifiers()
        button = event.button()
        scene_pos = self.mapToScene(event.pos())

        # 1. Search for nearest point if Shift is held (50px threshold)
        target_item = None
        if modifiers & Qt.ShiftModifier and button == Qt.LeftButton:
            min_dist = 50.0  # Threshold in pixels
            for item in self._overlay_items:
                dist = (item.pos() - scene_pos).manhattanLength() # Fast check
                if dist < min_dist:
                    # More accurate Euclidean check
                    dist = np.sqrt((item.pos().x() - scene_pos.x())**2 + (item.pos().y() - scene_pos.y())**2)
                    if dist < min_dist:
                        min_dist = dist
                        target_item = item

        # If no nearest point found, check exactly under mouse for point or background
        if target_item is None:
            item = self.itemAt(event.pos())
            while item is not None:
                if isinstance(item, DraggablePointItem):
                    target_item = item
                    break
                item = item.parentItem()

        is_point = isinstance(target_item, DraggablePointItem)

        # Handle Pan and Add only if not clicking on a point (or grabbing a point via Shift)
        if not is_point:
            # 1. Pan View (Left Click or Middle Click)
            if button == Qt.MiddleButton or (button == Qt.LeftButton and not modifiers):
                self._is_panning = True
                self._pan_start = event.pos()
                self.setCursor(QCursor(Qt.ClosedHandCursor))
                event.accept()
                return

            # 2. Add Point (Ctrl + Click)
            if modifiers & Qt.ControlModifier:
                img_rect = self._image_item.boundingRect()
                if img_rect.contains(scene_pos):
                    if button in (Qt.LeftButton, Qt.RightButton):
                        self.point_clicked.emit(scene_pos.x(), scene_pos.y(), int(button))
                        event.accept()
                        return
        else:
            # If we identified a target point (either directly or via nearest search)
            # and we are in Move/Delete mode, we need to ensure the item gets the event.
            # QGraphicsScene usually handles this if itemAt finds it, but for our "nearest snap"
            # we might need to manually set focus or proxy the event.
            if target_item != self.itemAt(event.pos()):
                # snap interaction: modify event or call item directly
                # For snap, we'll just allow the item to capture the move if it's within range
                pass

        # 3. Pass to super (which sends to DraggablePointItem for Move/Delete)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        is_pan_release = event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self._is_panning)
        
        if is_pan_release:
            self._is_panning = False
            self.setCursor(QCursor(Qt.CrossCursor))
            event.accept()
            return
            
        item = self.itemAt(event.pos())
        is_point = False
        while item is not None:
            if isinstance(item, DraggablePointItem):
                is_point = True
                break
            item = item.parentItem()
            
        if is_point:
            event.accept()
            return
            
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
            return

        scene_pos = self.mapToScene(event.pos())
        self.mouse_moved.emit(scene_pos.x(), scene_pos.y())

        # Update zoom view
        if self.zoom_view.isVisible() and self._source_pixmap:
            self.zoom_view.update_zoom(
                self._source_pixmap, scene_pos.x(), scene_pos.y())

        self.viewport().update()  # Trigger crosshair repaint
        super().mouseMoveEvent(event)

    def wheelEvent(self, event):
        """Zoom in/out with scroll wheel."""
        factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(factor, factor)
            self._zoom_level *= factor
        else:
            self.scale(1 / factor, 1 / factor)
            self._zoom_level /= factor
        # Clamp zoom
        self._zoom_level = max(0.1, min(self._zoom_level, 20.0))

    def get_zoom_level(self):
        return self._zoom_level

    def zoom_in(self):
        self.scale(1.2, 1.2)
        self._zoom_level *= 1.2

    def zoom_out(self):
        self.scale(1 / 1.2, 1 / 1.2)
        self._zoom_level /= 1.2

    def drawForeground(self, painter, rect):
        """Draw crosshair and grid overlays."""
        super().drawForeground(painter, rect)

        # Grid
        if self._show_grid:
            pen = QPen(QColor(Colors.GRID), 0.5)
            pen.setCosmetic(True)
            painter.setPen(pen)
            img_rect = self._image_item.boundingRect()
            grid_step = 50
            x = img_rect.left()
            while x <= img_rect.right():
                painter.drawLine(QPointF(x, img_rect.top()), QPointF(x, img_rect.bottom()))
                x += grid_step
            y = img_rect.top()
            while y <= img_rect.bottom():
                painter.drawLine(QPointF(img_rect.left(), y), QPointF(img_rect.right(), y))
                y += grid_step

        # Crosshair at cursor
        if self._show_crosshair:
            cursor_pos = self.mapFromGlobal(QCursor.pos())
            scene_pos = self.mapToScene(cursor_pos)
            img_rect = self._image_item.boundingRect()
            if img_rect.contains(scene_pos):
                pen = QPen(QColor(Colors.CROSSHAIR), 0.8)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.drawLine(
                    QPointF(scene_pos.x(), img_rect.top()),
                    QPointF(scene_pos.x(), img_rect.bottom())
                )
                painter.drawLine(
                    QPointF(img_rect.left(), scene_pos.y()),
                    QPointF(img_rect.right(), scene_pos.y())
                )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Reposition zoom view at bottom-left
        if self.zoom_view:
            margin = 8
            self.zoom_view.move(
                margin,
                self.viewport().height() - self.zoom_view.height() - margin
            )
