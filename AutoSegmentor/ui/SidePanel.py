"""
SidePanel.py — Right-side dockable panel with annotation info, keypoint progress,
and batch/frame status for the AutoSegmenter annotation UI.
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QGroupBox, QProgressBar, QFrame, QScrollArea,
    QSizePolicy, QCheckBox, QPushButton, QTextEdit
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

    def update_annotations(self, points, labels, pose_coords=None):
        """Refresh the annotation list."""
        self.list_widget.clear()
        
        if pose_coords is not None:
            for pc in pose_coords:
                if not pc.get('visible', True):
                    continue
                lbl = pc.get('label', 1001)
                class_id = abs(lbl) // 1000
                instance_id = abs(lbl) % 1000
                is_neg = lbl < 0
                sign = "−" if is_neg else "+"
                color = QColor(Colors.ACCENT_RED) if is_neg else get_class_point_color(class_id)
                
                pt_name = pc.get('name')
                if pt_name == 'Negative_Point':
                    display_name = "Neg"
                else:
                    display_name = str(pc.get('point_id', 0) + 1)
                
                text = f" {sign} [{display_name}]  C{class_id}:I{instance_id}  ({int(pc['x'])}, {int(pc['y'])})"
                item = QListWidgetItem(text)
                item.setForeground(color)
                self.list_widget.addItem(item)
            
            count = sum(1 for pc in pose_coords if pc.get('visible', True))
            self.count_label.setText(f"{count} point{'s' if count != 1 else ''}")
        else:
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


class ProcessingStagePanel(QWidget):
    """Live SAM2/CoTracker batch-processing plan and per-stage timing.

    Lives in its own dock (see MainWindow._init_ui) rather than the right-hand
    Annotation sidebar, so it can be moved/floated independently — no
    QGroupBox chrome here since the dock's own title bar already labels it.

    Which stages run (and for how many frames) differs per batch — SAM-only,
    CoTracker-only, both, or a partial refinement from a mid-batch anchor
    frame — so the panel is driven by an explicit per-batch "plan" (a list of
    {key, label, frames} the current batch will actually execute) rather than
    a single flat average. Each stage gets its own row (pending ○ / running ▸
    / done ✓) showing its frame count and a time estimate; the estimate comes
    from a rolling per-frame-rate history for that specific stage, so it
    stays accurate as SAM2 (many frames) and CoTracker (fewer, and no
    per-frame progress hook — only a single blocking call) diverge over time.
    The overall progress bar is determinate when the active stage reports
    real per-frame progress (SAM2's propagate_in_video loop); otherwise it is
    animated toward the stage's own time estimate so the user still sees
    something moving instead of a static "busy" spinner.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        self.stage_label = QLabel("Idle")
        self.stage_label.setFont(Fonts.body())
        self.stage_label.setWordWrap(True)

        time_row = QWidget()
        time_layout = QHBoxLayout(time_row)
        time_layout.setContentsMargins(0, 0, 0, 0)
        self.elapsed_label = QLabel("")
        self.elapsed_label.setFont(Fonts.small())
        self.eta_label = QLabel("")
        self.eta_label.setFont(Fonts.small())
        self.eta_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        time_layout.addWidget(self.elapsed_label)
        time_layout.addStretch()
        time_layout.addWidget(self.eta_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setTextVisible(True)

        # Per-stage checklist rows, built fresh for each batch's plan (mirrors
        # KeypointProgressPanel's row layout for visual consistency).
        self._stage_rows = {}  # key -> (row_widget, icon_label, text_label, time_label)
        self._stage_container = QWidget()
        self._stage_layout = QVBoxLayout(self._stage_container)
        self._stage_layout.setSpacing(1)
        self._stage_layout.setContentsMargins(0, 2, 0, 2)

        layout.addWidget(self.stage_label)
        layout.addWidget(time_row)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self._stage_container)
        layout.addStretch()

        self.set_idle()

    def set_idle(self):
        self.stage_label.setText("Idle")
        self.elapsed_label.setText("")
        self.eta_label.setText("")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        self._clear_stage_rows()

    def _clear_stage_rows(self):
        for row, *_ in self._stage_rows.values():
            self._stage_layout.removeWidget(row)
            row.deleteLater()
        self._stage_rows.clear()

    def set_plan(self, stages):
        """Rebuild the checklist for the stages this batch will actually run.

        stages: list of {key, label, frames, est_seconds (or None)}.
        """
        self._clear_stage_rows()
        for stage in stages:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(2, 1, 2, 1)
            row_layout.setSpacing(6)

            icon = QLabel("○")
            icon.setFont(Fonts.body())
            icon.setFixedWidth(16)
            icon.setStyleSheet(f"color: {Colors.TEXT_MUTED};")

            frame_note = f" ({stage['frames']} frames)" if stage.get('frames') else ""
            text_lbl = QLabel(f"{stage['label']}{frame_note}")
            text_lbl.setFont(Fonts.mono_small())
            text_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
            text_lbl.setWordWrap(True)

            time_lbl = QLabel(self._format_estimate(stage.get('est_seconds')))
            time_lbl.setFont(Fonts.small())
            time_lbl.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
            time_lbl.setFixedWidth(50)
            time_lbl.setAlignment(Qt.AlignRight)

            row_layout.addWidget(icon)
            row_layout.addWidget(text_lbl, 1)
            row_layout.addWidget(time_lbl)

            self._stage_layout.addWidget(row)
            self._stage_rows[stage['key']] = (row, icon, text_lbl, time_lbl)

    @staticmethod
    def _format_estimate(seconds):
        if seconds is None:
            return "~?s"
        return f"~{seconds:.0f}s"

    def mark_stage_running(self, key):
        if key not in self._stage_rows:
            return
        _, icon, text_lbl, _ = self._stage_rows[key]
        icon.setText("▸")
        icon.setStyleSheet(f"color: {Colors.ACCENT_ORANGE};")
        text_lbl.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-weight: bold;")

    def mark_stage_frame_count(self, key, done, total):
        if key not in self._stage_rows:
            return
        _, _, _, time_lbl = self._stage_rows[key]
        time_lbl.setText(f"{done}/{total}")

    def mark_stage_done(self, key, actual_seconds):
        if key not in self._stage_rows:
            return
        _, icon, text_lbl, time_lbl = self._stage_rows[key]
        icon.setText("✓")
        icon.setStyleSheet(f"color: {Colors.SUCCESS};")
        text_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        time_lbl.setText(f"{actual_seconds:.1f}s")

    def set_current_stage_label(self, label):
        self.stage_label.setText(label)

    def start_frame_progress(self, total):
        """Switch the main bar to determinate mode for a stage with real per-frame progress."""
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"0 / {total} frames")

    def set_frame_progress(self, done, total):
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(done)
        self.progress_bar.setFormat(f"{done} / {total} frames")

    def start_estimated_progress(self, stage_name):
        """Main bar for a stage with no real per-frame hook (e.g. CoTracker's
        single blocking inference call) — animated via set_estimated_progress
        toward the stage's own historical time estimate, so it still visibly
        moves instead of sitting on a static spinner."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(stage_name)

    def set_estimated_progress(self, fraction):
        self.progress_bar.setValue(int(max(0.0, min(fraction, 1.0)) * 100))

    def set_elapsed(self, seconds):
        self.elapsed_label.setText(f"Elapsed: {seconds:.0f}s")

    def set_eta(self, text):
        self.eta_label.setText(text)


class LogPanel(QWidget):
    """Timestamped processing log — its own dock, VSCode Output-panel style.

    Kept separate from ProcessingStagePanel so it can default to a short
    strip at the bottom of the window instead of competing for space in a
    side panel. History accumulates across batches (not cleared on idle).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(Fonts.mono_small())
        self.log.setStyleSheet(f"background-color: {Colors.BG_DARKEST}; color: {Colors.TEXT_SECONDARY}; border: none;")
        layout.addWidget(self.log)

    def append_log(self, text):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {text}")


class ModelRoutingPanel(QGroupBox):
    """Controls which models receive new annotation points."""

    routing_changed = pyqtSignal(list)  # list of active model strings
    auto_shift_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__("Model Routing", parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 16, 8, 8)

        # Bulk selection buttons
        btn_layout = QHBoxLayout()
        self.btn_all = QPushButton("All")
        self.btn_all.setFixedHeight(22)
        self.btn_all.setFont(Fonts.small())
        self.btn_all.clicked.connect(self._select_all)
        
        self.btn_none = QPushButton("None")
        self.btn_none.setFixedHeight(22)
        self.btn_none.setFont(Fonts.small())
        self.btn_none.clicked.connect(self._select_none)
        
        btn_layout.addWidget(self.btn_all)
        btn_layout.addWidget(self.btn_none)
        layout.addLayout(btn_layout)

        # Checkboxes for models
        self.model_checks = {}
        for mid, name in [("sam", "Mask (SAM)"), ("pose", "Pose (CoTracker)")]:
            cb = QCheckBox(name)
            cb.setFont(Fonts.body())
            cb.toggled.connect(self._on_check_toggled)
            layout.addWidget(cb)
            self.model_checks[mid] = cb

        layout.addSpacing(4)
        layout.addWidget(SeparatorLine())
        layout.addSpacing(4)

        # Auto-shift toggle
        self.auto_shift_cb = QCheckBox("Auto-Shift Instance")
        self.auto_shift_cb.setFont(Fonts.body())
        self.auto_shift_cb.setToolTip("Automatically advance to next instance when Pose is full")
        self.auto_shift_cb.toggled.connect(self.auto_shift_toggled.emit)
        layout.addWidget(self.auto_shift_cb)

    def _select_all(self):
        for cb in self.model_checks.values():
            cb.setChecked(True)

    def _select_none(self):
        for cb in self.model_checks.values():
            cb.setChecked(False)

    def _on_check_toggled(self):
        active = [mid for mid, cb in self.model_checks.items() if cb.isChecked()]
        self.routing_changed.emit(active)

    def set_active_models(self, active_list):
        for mid, cb in self.model_checks.items():
            cb.blockSignals(True)
            cb.setChecked(mid in active_list)
            cb.blockSignals(False)

    def set_auto_shift(self, enabled):
        self.auto_shift_cb.blockSignals(True)
        self.auto_shift_cb.setChecked(enabled)
        self.auto_shift_cb.blockSignals(False)


class SidePanel(QScrollArea):
    """Annotation/dataset-label panel — lives in its own dock (see
    MainWindow._init_ui), defaulting to the right edge like before, but now
    freely resizable/movable/floatable rather than a fixed-width splitter pane.

    Combines BatchInfoPanel, ToolInfoPanel, AnnotationListPanel,
    KeypointProgressPanel, ModelRoutingPanel and LiveConfigPanel. Processing
    status/timing and the log live in their own separate docks (see
    ProcessingStagePanel / LogPanel) so they can be positioned independently.
    """

    export_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(SIDEBAR_WIDTH)
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
        self.model_routing = ModelRoutingPanel()
        self.live_config = LiveConfigPanel()

        layout.addWidget(self.batch_info)
        layout.addWidget(self.tool_info)
        layout.addWidget(self.annotation_list)
        layout.addWidget(self.keypoint_progress)
        layout.addWidget(self.model_routing)
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

