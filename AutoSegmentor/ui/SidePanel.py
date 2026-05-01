"""
SidePanel.py — Right-side dockable panel with annotation info, keypoint progress,
and batch/frame status for the AutoSegmenter annotation UI.
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QGroupBox, QProgressBar, QFrame, QScrollArea,
    QSizePolicy, QCheckBox, QPushButton
)


from .UITheme import (
    Colors, ANNOTATION_COLORS_QT, Fonts, SIDEBAR_WIDTH, get_class_point_color
)


class SectionHeader(QLabel):
    """Styled section header with icon prefix."""

    def __init__(self, icon_char, text, parent=None):
        super().__init__(f" {icon_char}  {text}", parent)
        self.setObjectName("sectionHeader")
        self.setFont(Fonts.header())


class SeparatorLine(QFrame):
    """Thin horizontal separator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setStyleSheet(f"background-color: {Colors.BORDER}; max-height: 1px;")


class ColorDot(QLabel):
    """Small colored dot indicator."""

    def __init__(self, color, size=10, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, size, size)
        painter.end()
        self.setPixmap(pixmap)


class BatchInfoPanel(QGroupBox):
    """Displays current batch and frame information."""

    def __init__(self, parent=None):
        super().__init__("Navigation", parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 16, 8, 8)

        self.batch_label = QLabel("Batch: — / —")
        self.batch_label.setFont(Fonts.body())
        self.frame_label = QLabel("Frame: —")
        self.frame_label.setFont(Fonts.body())
        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setFont(Fonts.body())

        self.batch_progress = QProgressBar()
        self.batch_progress.setFixedHeight(14)
        self.batch_progress.setTextVisible(True)
        self.batch_progress.setFormat("%v / %m")

        layout.addWidget(self.batch_label)
        layout.addWidget(self.frame_label)
        layout.addWidget(self.zoom_label)
        layout.addWidget(self.batch_progress)

    def update_info(self, batch, total_batches, frame_idx, total_frames, zoom_level):
        self.batch_label.setText(f"Batch: {batch + 1} / {total_batches}")
        self.frame_label.setText(f"Frame: {frame_idx}")
        self.zoom_label.setText(f"Zoom: {zoom_level:.0%}")
        self.batch_progress.setMaximum(max(total_batches, 1))
        self.batch_progress.setValue(batch + 1)


class AnnotationListPanel(QGroupBox):
    """Displays list of current annotation points with class colors."""

    def __init__(self, parent=None):
        super().__init__("Annotations", parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 16, 8, 8)

        self.count_label = QLabel("0 points")
        self.count_label.setFont(Fonts.small())
        self.count_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")

        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(180)
        self.list_widget.setFont(Fonts.mono_small())

        layout.addWidget(self.count_label)
        layout.addWidget(self.list_widget)

    def update_annotations(self, points, labels):
        """Refresh the annotation list."""
        self.list_widget.clear()
        for idx, (pt, lbl) in enumerate(zip(points, labels)):
            class_id = abs(lbl) // 1000
            instance_id = abs(lbl) % 1000
            is_neg = lbl < 0
            sign = "−" if is_neg else "+"
            # Use per-class point color (complement of mask) matching the canvas
            color = QColor(Colors.ACCENT_RED) if is_neg else get_class_point_color(class_id)
            text = f" {sign} [{idx + 1}]  C{class_id}:I{instance_id}  ({int(pt[0])}, {int(pt[1])})"
            item = QListWidgetItem(text)
            item.setForeground(color)
            self.list_widget.addItem(item)		

        count = len(points)
        self.count_label.setText(f"{count} point{'s' if count != 1 else ''}")


class KeypointProgressPanel(QGroupBox):
    """Displays pose keypoint completion progress with checkmarks and visibility toggles."""

    visibility_toggled = pyqtSignal(int, bool)

    def __init__(self, parent=None):
        super().__init__("Keypoints", parent)
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(2)
        self._layout.setContentsMargins(8, 16, 8, 8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setTextVisible(True)
        self._layout.addWidget(self.progress_bar)

        self._keypoint_labels = []
        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setSpacing(1)
        self._container_layout.setContentsMargins(0, 4, 0, 0)
        self._layout.addWidget(self._container)

    def setup_keypoints(self, keypoint_names):
        """Initialize the keypoint list."""
        # Clear existing — _keypoint_labels holds (row, check, name_lbl, vis_cb) tuples
        for row, check, name_lbl, vis_cb in self._keypoint_labels:
            self._container_layout.removeWidget(row)
            row.deleteLater()
        self._keypoint_labels.clear()

        self.progress_bar.setMaximum(max(len(keypoint_names), 1))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"0 / {len(keypoint_names)}")

        for i, name in enumerate(keypoint_names):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(4, 1, 4, 1)
            row_layout.setSpacing(6)

            check = QLabel("○")
            check.setFont(Fonts.body())
            check.setFixedWidth(16)
            check.setStyleSheet(f"color: {Colors.TEXT_MUTED};")

            name_lbl = QLabel(name)
            name_lbl.setFont(Fonts.mono_small())
            name_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
            
            vis_cb = QCheckBox("Vis")
            vis_cb.setEnabled(False)
            vis_cb.setChecked(True)
            vis_cb.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
            vis_cb.toggled.connect(lambda checked, idx=i: self.visibility_toggled.emit(idx, checked))

            row_layout.addWidget(check)
            row_layout.addWidget(name_lbl)
            row_layout.addStretch()
            row_layout.addWidget(vis_cb)

            self._container_layout.addWidget(row)
            self._keypoint_labels.append((row, check, name_lbl, vis_cb))

    def update_progress(self, completed_count, pose_coords=None):
        """Update the keypoint completion status."""
        total = len(self._keypoint_labels)
        self.progress_bar.setValue(min(completed_count, total))
        self.progress_bar.setFormat(f"{min(completed_count, total)} / {total}")

        for i, (row, check, name_lbl, vis_cb) in enumerate(self._keypoint_labels):
            if i < completed_count:
                check.setText("✓")
                check.setStyleSheet(f"color: {Colors.SUCCESS};")
                name_lbl.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")
                vis_cb.setEnabled(True)
                if pose_coords and i < len(pose_coords):
                    vis_cb.blockSignals(True)
                    vis_cb.setChecked(pose_coords[i].get('visible', True))
                    vis_cb.blockSignals(False)
            else:
                check.setText("○")
                check.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
                name_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
                vis_cb.setEnabled(False)

    def set_current(self, index):
        """Highlight the current keypoint to click."""
        for i, (row, check, name_lbl, vis_cb) in enumerate(self._keypoint_labels):
            if i == index:
                check.setText("▸")
                check.setStyleSheet(f"color: {Colors.ACCENT_ORANGE};")
                name_lbl.setStyleSheet(f"color: {Colors.ACCENT_ORANGE}; font-weight: bold;")
            elif i < index:
                check.setText("✓")
                check.setStyleSheet(f"color: {Colors.SUCCESS};")
                name_lbl.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")
            else:
                check.setText("○")
                check.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
                name_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        self.progress_bar.setValue(min(index, len(self._keypoint_labels)))
        self.progress_bar.setFormat(
            f"{min(index, len(self._keypoint_labels))} / {len(self._keypoint_labels)}")


class ToolInfoPanel(QGroupBox):
    """Shows the current active tool mode and class/instance info."""

    def __init__(self, parent=None):
        super().__init__("Current Tool", parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 16, 8, 8)

        self.mode_label = QLabel("Mode: Segment")
        self.mode_label.setFont(Fonts.body())

        self.class_label = QLabel("Class: 1")
        self.class_label.setFont(Fonts.body())

        self.instance_label = QLabel("Instance: 1")
        self.instance_label.setFont(Fonts.body())

        self.class_color_dot = ColorDot(ANNOTATION_COLORS_QT.get(1, QColor(255, 80, 80)), 12)

        class_row = QWidget()
        class_layout = QHBoxLayout(class_row)
        class_layout.setContentsMargins(0, 0, 0, 0)
        class_layout.setSpacing(6)
        class_layout.addWidget(self.class_color_dot)
        class_layout.addWidget(self.class_label)
        class_layout.addStretch()

        layout.addWidget(self.mode_label)
        layout.addWidget(class_row)
        layout.addWidget(self.instance_label)

    def update_info(self, mode, class_id, instance_id):
        self.mode_label.setText(f"Mode: {mode}")
        self.class_label.setText(f"Class: {class_id}")
        self.instance_label.setText(f"Instance: {instance_id}")
        # Update color dot — show the per-class point color (complement of mask)
        color = get_class_point_color(class_id)
        pixmap = QPixmap(12, 12)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, 12, 12)
        painter.end()
        self.class_color_dot.setPixmap(pixmap)


class LiveConfigPanel(QGroupBox):
    """Live configuration controls for mask alpha and point size."""

    mask_alpha_changed = pyqtSignal(int)   # 0-100
    point_size_changed = pyqtSignal(int)   # px
    backward_tracking_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__("Live Config", parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 16, 8, 8)

        # Mask Alpha
        alpha_row = QWidget()
        alpha_hl = QHBoxLayout(alpha_row)
        alpha_hl.setContentsMargins(0, 0, 0, 0)
        alpha_lbl = QLabel("Mask α:")
        alpha_lbl.setFont(Fonts.body())
        alpha_lbl.setFixedWidth(52)
        from PyQt5.QtWidgets import QSlider, QSpinBox as _QSpinBox
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        self.alpha_value_lbl = QLabel("50%")
        self.alpha_value_lbl.setFont(Fonts.small())
        self.alpha_value_lbl.setFixedWidth(30)
        self.alpha_slider.valueChanged.connect(self._on_alpha_changed)
        alpha_hl.addWidget(alpha_lbl)
        alpha_hl.addWidget(self.alpha_slider, 1)
        alpha_hl.addWidget(self.alpha_value_lbl)
        layout.addWidget(alpha_row)

        # Point Size
        pt_row = QWidget()
        pt_hl = QHBoxLayout(pt_row)
        pt_hl.setContentsMargins(0, 0, 0, 0)
        pt_lbl = QLabel("Pt Size:")
        pt_lbl.setFont(Fonts.body())
        pt_lbl.setFixedWidth(52)
        self.pt_spinbox = _QSpinBox()
        self.pt_spinbox.setRange(2, 20)
        self.pt_spinbox.setValue(5)
        self.pt_spinbox.setFixedWidth(55)
        self.pt_spinbox.valueChanged.connect(self._on_pt_size_changed)
        pt_hl.addWidget(pt_lbl)
        pt_hl.addWidget(self.pt_spinbox)
        pt_hl.addStretch()
        layout.addWidget(pt_row)

        # Propagate Backward
        self.backward_cb = QCheckBox("Propagate Backward")
        self.backward_cb.setFont(Fonts.body())
        self.backward_cb.setChecked(True)
        self.backward_cb.toggled.connect(self.backward_tracking_toggled.emit)
        layout.addWidget(self.backward_cb)

    def _on_alpha_changed(self, val: int):
        self.alpha_value_lbl.setText(f"{val}%")
        self.mask_alpha_changed.emit(val)

    def _on_pt_size_changed(self, val: int):
        self.point_size_changed.emit(val)


class SidePanel(QScrollArea):
    """Right-side panel containing all info panels.

    Combines BatchInfoPanel, ToolInfoPanel, AnnotationListPanel,
    KeypointProgressPanel, and LiveConfigPanel into a scrollable sidebar.
    """

    export_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFixedWidth(SIDEBAR_WIDTH)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet(f"""
            QScrollArea {{
                border-left: 1px solid {Colors.BORDER};
                background-color: {Colors.BG_DARK};
            }}
        """)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)

        self.batch_info = BatchInfoPanel()
        self.tool_info = ToolInfoPanel()
        self.annotation_list = AnnotationListPanel()
        self.keypoint_progress = KeypointProgressPanel()
        self.live_config = LiveConfigPanel()

        layout.addWidget(self.batch_info)
        layout.addWidget(self.tool_info)
        layout.addWidget(self.annotation_list)
        layout.addWidget(self.keypoint_progress)
        layout.addWidget(self.live_config)
        
        layout.addStretch()

        # Export Button at the very bottom
        self.export_btn = QPushButton(" Export Dataset")
        self.export_btn.setFont(Fonts.header())
        self.export_btn.setFixedHeight(40)
        self.export_btn.setObjectName("acceptButton")
        self.export_btn.clicked.connect(self.export_requested.emit)
        layout.addWidget(self.export_btn)

        self.setWidget(container)

    def set_pose_mode(self, enabled, keypoints=None):
        """Show/hide the keypoint progress panel based on pose mode."""
        self.keypoint_progress.setVisible(enabled)
        if enabled and keypoints:
            self.keypoint_progress.setup_keypoints(keypoints)
            self.tool_info.update_info("Pose", 1, 1)
        elif not enabled:
            self.tool_info.update_info("Segment", 1, 1)

