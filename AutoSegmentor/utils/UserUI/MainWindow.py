"""
MainWindow.py - Main PyQt dialog window for the AutoSegmenter annotation UI.
"""

import cv2
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QKeySequence, QFont
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QMenuBar, QMenu,
    QToolBar, QAction, QStatusBar, QLabel, QPushButton, QComboBox,
    QUndoStack, QShortcut, QWidget, QSizePolicy, QTextEdit, QDialogButtonBox
)

from .UITheme import DARK_STYLESHEET, SIDEBAR_WIDTH, TOOLBAR_HEIGHT, STATUSBAR_HEIGHT, Colors
from .AnnotationCanvas import AnnotationCanvas
from .SidePanel import SidePanel
from .NavigationManager import AddPointCommand, ResetPointsCommand, DeletePointCommand, SkipPointCommand, DragPointCommand, NavigationState
from .logger_config import logger

class BatchProcessorThread(QThread):
    finished_batch = pyqtSignal(int)
    
    def __init__(self, handler, batch, query_frame_idx=None):
        super().__init__()
        self.handler = handler
        self.batch = batch
        self.query_frame_idx = query_frame_idx
        
    def run(self):
        processor = self.handler.pipeline_processor
        
        batch_index = self.batch * processor.config.batch_size
        processor.frame_handler.move_and_copy_frames(batch_index, processor.frame_paths, processor.config.batch_size)
        
        if processor.config.sam_enabled:
            # SAM2 with multi-frame prompts
            processor.mask_processor.generate_mask(
                batch_number=self.batch,
                sam2_predictor=processor.sam2_predictor,
                temp_directory=processor.config.temp_directory,
                prompt_encoding=processor.prompt_encoding,
                auto_prompt_encoding=processor.auto_prompt_encoding,
                predictor_lock=processor._predictor_lock
            )
            
        if (processor.config.pose_config and processor.config.pose_config.get('enabled')
                and processor.config.pose_config.get('tracker', 'lk').lower() == 'cotracker'):
            # CoTracker with optional specific query frame and backward tracking (enabled in wrapper)
            processor._track_batch_inline(self.batch, query_frame_idx=self.query_frame_idx)
            
        self.finished_batch.emit(self.batch)

class AnnotationWindow(QDialog):
    """Main AutoSegmenter annotation window. Runs modally, handles bg processing."""

    def __init__(self, handler, config, parent=None):
        super().__init__(parent)
        self.handler = handler
        self.config = config
        self.is_processing = False
        
        # UI State
        self.nav_state = NavigationState()
        self.nav_state.show_crosshair = getattr(self.config, 'ui_show_crosshair', True)
        self.nav_state.show_grid = getattr(self.config, 'ui_show_grid', False)
        self.undo_stack = QUndoStack(self)
        self.undo_stack.setUndoLimit(30)
        
        self._init_ui()
        self._setup_shortcuts()
        self._connect_signals()

        # ── Hold-to-scroll timer ─────────────────────────────────────────────
        # Fires every 100 ms while A/D/←/→ is held down
        self._nav_timer = QTimer(self)
        self._nav_timer.setInterval(100)
        self._nav_timer.timeout.connect(self._on_nav_timer)
        self._nav_direction = 0   # -1 = prev, +1 = next, 0 = idle

    def _init_ui(self):
        """Initialize main layout and widgets."""
        self.setWindowTitle("AutoSegmenter Annotation Tool")
        self.setMinimumSize(1024, 768)
        self.setStyleSheet(DARK_STYLESHEET)
        
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.menu_bar = QMenuBar(self)
        self._build_menu()
        main_layout.addWidget(self.menu_bar)
        
        self.tool_bar = QToolBar("Main Tools", self)
        self.tool_bar.setFixedHeight(TOOLBAR_HEIGHT)
        self.tool_bar.setMovable(False)
        self._build_toolbar()
        main_layout.addWidget(self.tool_bar)
        
        self.splitter = QSplitter(Qt.Horizontal, self)
        
        self.canvas = AnnotationCanvas(self)
        self.canvas.set_show_crosshair(self.nav_state.show_crosshair)
        self.canvas.set_show_grid(self.nav_state.show_grid)
        
        # Loader label
        self.loader_label = QLabel("Processing batch (SAM2/CoTracker)...", self.canvas)
        self.loader_label.setStyleSheet("QLabel { background-color: rgba(0, 0, 0, 180); color: white; font-size: 24px; padding: 20px; border-radius: 10px; }")
        self.loader_label.setAlignment(Qt.AlignCenter)
        self.loader_label.resize(400, 100)
        self.loader_label.move(self.canvas.width() // 2 - 200, self.canvas.height() // 2 - 50)
        self.loader_label.hide()
        
        self.splitter.addWidget(self.canvas)
        
        self.sidebar = SidePanel(self)
        is_pose = self.handler.pose_mode
        self.sidebar.set_pose_mode(is_pose, self.handler.pose_keypoints if is_pose else None)
        self.splitter.addWidget(self.sidebar)
        
        self.splitter.setSizes([800, SIDEBAR_WIDTH])
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        main_layout.addWidget(self.splitter, 1)
        
        self.status_bar = QStatusBar(self)
        self.status_bar.setFixedHeight(STATUSBAR_HEIGHT)
        self._build_statusbar()
        main_layout.addWidget(self.status_bar)

    def _build_menu(self):
        file_menu = self.menu_bar.addMenu("&File")

        save_action = QAction("&Save Progress", self)
        save_action.setShortcut("Ctrl+S")
        save_action.setStatusTip("Save current annotations to disk")
        save_action.triggered.connect(self._on_save_progress)
        file_menu.addAction(save_action)

        export_action = QAction("&Export to YOLO Dataset", self)
        export_action.setShortcut("Ctrl+E")
        export_action.setStatusTip("Export verified annotations as a YOLO-format dataset")
        export_action.triggered.connect(self._on_export_yolo)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        finish_action = QAction("Finish Pipeline", self)
        finish_action.triggered.connect(self.accept)
        file_menu.addAction(finish_action)

        file_menu.addSeparator()

        quit_action = QAction("E&xit Without Saving", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.reject)
        file_menu.addAction(quit_action)

        edit_menu = self.menu_bar.addMenu("&Edit")
        undo_action = self.undo_stack.createUndoAction(self, "&Undo")
        undo_action.setShortcut("Ctrl+Z")
        edit_menu.addAction(undo_action)
        redo_action = self.undo_stack.createRedoAction(self, "&Redo")
        redo_action.setShortcut("Ctrl+Y")
        edit_menu.addAction(redo_action)
        edit_menu.addSeparator()
        reset_action = QAction("Re&set All Points", self)
        reset_action.setShortcut("R")
        reset_action.triggered.connect(self.reset_points)
        edit_menu.addAction(reset_action)

        view_menu = self.menu_bar.addMenu("&View")
        self.crosshair_action = QAction("Toggle &Crosshair", self, checkable=True)
        self.crosshair_action.setChecked(self.nav_state.show_crosshair)
        self.crosshair_action.setShortcut("C")
        self.crosshair_action.triggered.connect(self.toggle_crosshair)
        view_menu.addAction(self.crosshair_action)
        self.grid_action = QAction("Toggle &Grid", self, checkable=True)
        self.grid_action.setChecked(self.nav_state.show_grid)
        self.grid_action.setShortcut("G")
        self.grid_action.triggered.connect(self.toggle_grid)
        view_menu.addAction(self.grid_action)
        self.zoom_view_action = QAction("Toggle Corner &Zoom", self, checkable=True)
        self.zoom_view_action.setChecked(True)
        self.zoom_view_action.setShortcut("Z")
        self.zoom_view_action.triggered.connect(self.toggle_corner_zoom)
        view_menu.addAction(self.zoom_view_action)

        help_menu = self.menu_bar.addMenu("&Help")
        shortcuts_action = QAction("Show &Shortcuts  (H)", self)
        shortcuts_action.setShortcut("H")
        shortcuts_action.triggered.connect(self._show_help_overlay)
        help_menu.addAction(shortcuts_action)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'loader_label'):
            self.loader_label.move(self.canvas.width() // 2 - self.loader_label.width() // 2, 
                                   self.canvas.height() // 2 - self.loader_label.height() // 2)

    def _build_toolbar(self):
        # Navigation
        self.btn_prev_batch = QPushButton("<< Prev Batch")
        self.btn_prev_batch.clicked.connect(self.prev_batch)
        self.tool_bar.addWidget(self.btn_prev_batch)
        
        self.btn_prev_img = QPushButton("< Prev Img")
        self.btn_prev_img.clicked.connect(self.prev_image)
        self.tool_bar.addWidget(self.btn_prev_img)
        
        self.btn_next_img = QPushButton("Next Img >")
        self.btn_next_img.clicked.connect(self.next_image)
        self.tool_bar.addWidget(self.btn_next_img)
        
        self.btn_next_batch = QPushButton("Next Batch >>")
        self.btn_next_batch.clicked.connect(self.next_batch)
        self.tool_bar.addWidget(self.btn_next_batch)
        
        self.tool_bar.addSeparator()

        self.btn_reset = QPushButton("Reset (R)")
        self.btn_reset.setObjectName("resetButton")
        self.btn_reset.clicked.connect(self.reset_points)
        self.tool_bar.addWidget(self.btn_reset)

        self.btn_toggle_mask = QPushButton("Toggle Mask (M)")
        self.btn_toggle_mask.setObjectName("toggleMaskButton")
        self.btn_toggle_mask.setCheckable(True)
        self.btn_toggle_mask.setChecked(True)
        self.btn_toggle_mask.clicked.connect(self.toggle_mask)
        self.tool_bar.addWidget(self.btn_toggle_mask)
        
        self.tool_bar.addSeparator()
        
        self.tool_bar.addWidget(QLabel("  Class: "))
        self.class_combo = QComboBox()
        for i in range(1, 11):
            self.class_combo.addItem(f"Class {i}", i)
        self.class_combo.setCurrentIndex(self.handler.current_class_label - 1)
        self.class_combo.currentIndexChanged.connect(self.change_class)
        self.tool_bar.addWidget(self.class_combo)
        if self.handler.pose_mode:
            self.class_combo.setEnabled(False)
            
        self.tool_bar.addWidget(QLabel("  Inst: "))
        self.inst_lbl = QLabel(str(self.handler.current_instance_id))
        self.inst_lbl.setFixedWidth(20)
        self.tool_bar.addWidget(self.inst_lbl)
        
        from PyQt5.QtWidgets import QLineEdit
        self.tool_bar.addWidget(QLabel("  Jump: "))
        self.frame_jump_input = QLineEdit()
        self.frame_jump_input.setPlaceholderText("Frame #")
        self.frame_jump_input.setFixedWidth(60)
        self.frame_jump_input.returnPressed.connect(self.jump_to_frame)
        self.tool_bar.addWidget(self.frame_jump_input)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.tool_bar.addWidget(spacer)
        
        self.btn_process_batch = QPushButton("Process Batch")
        self.btn_process_batch.setObjectName("processBatchButton")
        self.btn_process_batch.clicked.connect(self.process_current_batch)
        self.btn_process_batch.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.tool_bar.addWidget(self.btn_process_batch)
        
    def _build_statusbar(self):
        self.coord_label = QLabel(" 📍 (0, 0)")
        self.coord_label.setObjectName("statusCoord")
        self.status_bar.addWidget(self.coord_label)
        self._save_status_label = QLabel("")
        self._save_status_label.setStyleSheet(f"color: {Colors.ACCENT_GREEN}; font-size: 8pt;")
        self.status_bar.addWidget(self._save_status_label)
        self.status_bar.addPermanentWidget(
            QLabel(" A/←: Prev  D/→: Next  [/]: Batch  R: Reset  M: Mask  H: Help  Ctrl+S: Save  Ctrl+E: Export  +/-: Zoom")
        )

    def _setup_shortcuts(self):
        for i in range(1, 10):
            QShortcut(QKeySequence(str(i)), self, lambda checked, idx=i: self.set_class(idx))

        QShortcut(QKeySequence("Tab"),       self, self.next_instance)
        QShortcut(QKeySequence("Shift+Tab"), self, self.prev_instance)
        QShortcut(QKeySequence("Space"),     self, self.skip_point)
        QShortcut(QKeySequence("+"),         self, self.canvas.zoom_in)
        QShortcut(QKeySequence("="),         self, self.canvas.zoom_in)
        QShortcut(QKeySequence("-"),         self, self.canvas.zoom_out)
        QShortcut(QKeySequence("M"),         self, self.btn_toggle_mask.animateClick)
        QShortcut(QKeySequence("U"),         self, self.undo_stack.undo)
        QShortcut(QKeySequence("Return"),    self, self.btn_process_batch.animateClick)
        QShortcut(QKeySequence("Enter"),     self, self.btn_process_batch.animateClick)

        # Image navigation — arrow keys AND A/D
        QShortcut(QKeySequence("Left"),  self, self.btn_prev_img.animateClick)
        QShortcut(QKeySequence("Right"), self, self.btn_next_img.animateClick)
        QShortcut(QKeySequence("A"),     self, self.btn_prev_img.animateClick)
        QShortcut(QKeySequence("D"),     self, self.btn_next_img.animateClick)

        # Batch navigation — [ and ]
        QShortcut(QKeySequence("["), self, self.btn_prev_batch.animateClick)
        QShortcut(QKeySequence("]"), self, self.btn_next_batch.animateClick)

        # Save / Export / Help
        QShortcut(QKeySequence("Ctrl+S"), self, self._on_save_progress)
        QShortcut(QKeySequence("Ctrl+E"), self, self._on_export_yolo)
        QShortcut(QKeySequence("H"),      self, self._show_help_overlay)

        # NOTE: A / D / ← / → are handled via keyPressEvent / keyReleaseEvent
        #       below so that holding the key scrolls at 0.1 s rate.

    def _connect_signals(self):
        self.canvas.point_clicked.connect(self.handle_canvas_click)
        self.canvas.mouse_moved.connect(self.handle_mouse_move)
        self.canvas.point_moved.connect(self.handle_point_moved)
        self.canvas.point_dragging.connect(self.handle_point_dragging)
        self.canvas.point_deleted.connect(self.handle_point_deleted)
        self.sidebar.keypoint_progress.visibility_toggled.connect(self.handle_visibility_toggled)
        self.undo_stack.indexChanged.connect(self._on_undo_stack_changed)

    # ─── Hold-to-scroll key handling ───────────────────────────────────────────

    _NAV_KEYS = {Qt.Key_Left, Qt.Key_Right, Qt.Key_A, Qt.Key_D}

    def _is_text_widget_focused(self) -> bool:
        """Return True when a text-entry widget has keyboard focus.

        Prevents A/D from scrolling images while the user is typing
        in the frame-jump input or any other QLineEdit/QComboBox.
        """
        from PyQt5.QtWidgets import QLineEdit, QComboBox, QSpinBox
        fw = self.focusWidget()
        return isinstance(fw, (QLineEdit, QComboBox, QSpinBox))

    def keyPressEvent(self, event):
        """Start the hold-to-scroll timer on A/D/←/→ press."""
        key = event.key()
        if key in self._NAV_KEYS and not event.isAutoRepeat() and not self._is_text_widget_focused():
            direction = -1 if key in (Qt.Key_Left, Qt.Key_A) else 1
            if self._nav_direction != direction:
                self._nav_direction = direction
                # Fire immediately for the first frame, then rely on timer
                self._on_nav_timer()
                self._nav_timer.start()
            return   # consumed
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Stop the hold-to-scroll timer when A/D/←/→ is released."""
        key = event.key()
        if key in self._NAV_KEYS and not event.isAutoRepeat():
            self._nav_timer.stop()
            self._nav_direction = 0
            return   # consumed
        super().keyReleaseEvent(event)

    def _on_nav_timer(self):
        """Called every 100 ms while a navigation key is held."""
        if self._nav_direction == -1:
            self.prev_image()
        elif self._nav_direction == 1:
            self.next_image()
        else:
            self._nav_timer.stop()

    # ─── Navigation Handlers ─────────────────────────────────────────────────
    def prev_image(self):
        idx = max(0, self.handler.current_frame_idx - 1)
        if self.handler.has_data_for_frame(idx) or idx == 0:
            self.handler.load_frame_for_ui(idx)
        else:
            logger.info(f"Navigation blocked: Frame {idx} has no points/results yet.")
        
    def next_image(self):
        idx = min(len(self.handler.frame_paths) - 1, self.handler.current_frame_idx + 1)
        if self.handler.has_data_for_frame(idx):
            self.handler.load_frame_for_ui(idx)
        else:
            logger.info(f"Navigation blocked: Frame {idx} has no points/results yet.")
        
    def prev_batch(self):
        idx = max(0, self.handler.current_frame_idx - self.config.batch_size)
        idx = (idx // self.config.batch_size) * self.config.batch_size
        self.handler.load_frame_for_ui(idx)
        
    def next_batch(self):
        idx = min(len(self.handler.frame_paths) - 1, self.handler.current_frame_idx + self.config.batch_size)
        idx = (idx // self.config.batch_size) * self.config.batch_size
        self.handler.load_frame_for_ui(idx)

    # ─── Event Handlers ──────────────────────────────────────────────────────
    def handle_canvas_click(self, x, y, button):
        full_label = self.handler.encode_label(self.handler.current_class_label, self.handler.current_instance_id)
        if button == Qt.RightButton:
            full_label *= -1
            
        pose_click = None
        if self.handler.pose_mode:
            if self.handler.current_keypoint_index < len(self.handler.pose_keypoints):
                kp_name = self.handler.pose_keypoints[self.handler.current_keypoint_index]
                pose_click = {
                    "name": kp_name,
                    "point_id": self.handler.current_keypoint_index,
                    "x": int(x),
                    "y": int(y),
                    "visible": True
                }

        cmd = AddPointCommand(self.handler, [x, y], full_label, pose_click)
        self.undo_stack.push(cmd)

    def handle_point_deleted(self, index):
        if index < len(self.handler.selected_points):
            cmd = DeletePointCommand(self.handler, index)
            self.undo_stack.push(cmd)

    def skip_point(self):
        if self.handler.pose_mode and self.handler.current_keypoint_index < len(self.handler.pose_keypoints):
            cmd = SkipPointCommand(self.handler)
            self.undo_stack.push(cmd)

    def handle_mouse_move(self, x, y):
        self.nav_state.update_mouse(int(x), int(y))
        self.coord_label.setText(f" 📍 ({int(x)}, {int(y)})")

    def handle_point_moved(self, index, old_x, old_y, new_x, new_y):
        if index < len(self.handler.selected_points):
            cmd = DragPointCommand(self.handler, index, [old_x, old_y], [new_x, new_y])
            self.undo_stack.push(cmd)

    def handle_point_dragging(self, index, x, y):
        if index < len(self.handler.selected_points):
            self.handler.selected_points[index] = [x, y]
            if self.handler.pose_mode and self.handler.pose_click_coords:
                if index < len(self.handler.pose_click_coords):
                    self.handler.pose_click_coords[index]['x'] = int(x)
                    self.handler.pose_click_coords[index]['y'] = int(y)
            self.canvas.update_skeleton(
                self.handler.selected_points,
                labels=self.handler.selected_labels,
                pose_coords=self.handler.pose_click_coords if self.handler.pose_mode else None
            )
            self.coord_label.setText(f" 📍 ({int(x)}, {int(y)})")

    def handle_visibility_toggled(self, index, is_visible):
        if self.handler.pose_mode and self.handler.pose_click_coords:
            if index < len(self.handler.pose_click_coords):
                self.handler.pose_click_coords[index]['visible'] = is_visible
                self._trigger_prompt_update()

    # ─── Frame Updates ───────────────────────────────────────────────────────
    def refresh_display(self):
        if self.btn_toggle_mask.isChecked():
            self.canvas.set_image(self.handler.current_frame_only_with_points)
        else:
            self.canvas.set_image(cv2.imread(self.handler.frame_paths[self.handler.current_frame_idx]))
        self._redraw_annotations()

    def _on_undo_stack_changed(self, idx):
        self._trigger_prompt_update()

    def _trigger_prompt_update(self):
        self.handler.user_prompt_adder_pyqt()
        
        if self.btn_toggle_mask.isChecked():
            self.canvas.update_image(self.handler.current_frame)
        else:
            self.canvas.update_image(cv2.imread(self.handler.frame_paths[self.handler.current_frame_idx]))
            
        self._redraw_annotations()
        self._update_sidebar()

    def _redraw_annotations(self):
        self.canvas.draw_annotations(
            self.handler.selected_points,
            self.handler.selected_labels,
            pose_keypoints=self.handler.pose_keypoints if self.handler.pose_mode else None,
            pose_coords=self.handler.pose_click_coords if self.handler.pose_mode else None,
            pose_config=self.config.pose_config if self.handler.pose_mode else None
        )
        
    def _update_sidebar(self):
        self.sidebar.annotation_list.update_annotations(
            self.handler.selected_points, 
            self.handler.selected_labels
        )
        
        mode_str = "Pose" if self.handler.pose_mode else "Segment"
        self.sidebar.tool_info.update_info(
            mode_str, 
            self.handler.current_class_label, 
            self.handler.current_instance_id
        )
        self.inst_lbl.setText(str(self.handler.current_instance_id))
        
        if self.handler.pose_mode:
            self.sidebar.keypoint_progress.update_progress(
                self.handler.current_keypoint_index,
                self.handler.pose_click_coords
            )
            self.sidebar.keypoint_progress.set_current(self.handler.current_keypoint_index)

    def set_batch_info(self, batch, total_batches, frame_idx, total_frames):
        self.nav_state.set_batch_info(batch, total_batches, frame_idx, total_frames)
        self.sidebar.batch_info.update_info(batch, total_batches, frame_idx, total_frames, 1.0)
        
        # Check if batch is processing
        if self.is_processing and self.processing_batch == batch:
            self.loader_label.show()
        else:
            self.loader_label.hide()

    # ─── UI Actions ──────────────────────────────────────────────────────────
    def toggle_mask(self):
        self.refresh_display()

    def set_class(self, class_id):
        if self.handler.pose_mode:
            return
        self.class_combo.setCurrentIndex(class_id - 1)

    def change_class(self, index):
        self.handler.change_class_label_pyqt(index + 1)
        self._update_sidebar()

    def next_instance(self):
        self.handler.current_instance_id += 1
        self._update_sidebar()
        
    def prev_instance(self):
        if self.handler.current_instance_id > 1:
            self.handler.current_instance_id -= 1
            self._update_sidebar()

    def reset_points(self):
        if self.handler.selected_points:
            cmd = ResetPointsCommand(self.handler)
            self.undo_stack.push(cmd)

    def toggle_crosshair(self):
        self.nav_state.show_crosshair = self.canvas.toggle_crosshair()

    def toggle_grid(self):
        self.nav_state.show_grid = self.canvas.toggle_grid()
        
    def toggle_corner_zoom(self):
        self.canvas.zoom_view.setVisible(self.zoom_view_action.isChecked())

    def jump_to_frame(self):
        text = self.frame_jump_input.text()
        try:
            val = int(text)
            if 0 <= val < len(self.handler.frame_paths):
                self.handler.load_frame_for_ui(val)
                self.frame_jump_input.clear()
            else:
                logger.warning(f"Invalid frame index: {val}")
        except ValueError:
            pass

    # ─── New Action Handlers ──────────────────────────────────────────────────

    def _on_save_progress(self):
        """Ctrl+S — save current annotation and flash status bar."""
        self.handler.save_current_annotation()
        self._save_status_label.setText("  ✓  Saved")
        QTimer.singleShot(2000, lambda: self._save_status_label.setText(""))
        logger.info("Progress saved by user (Ctrl+S).")

    def _on_export_yolo(self):
        """Ctrl+E — trigger YOLO dataset export (non-blocking)."""
        try:
            from DatasetManager.YolovDatasetManager.DatasetCreatere import DatasetCreator
            creator = DatasetCreator()
            wdir = self.config.working_dir_name if hasattr(self.config, 'working_dir_name') else "working_dir"
            creator.create_dataset(verified_img_dir=getattr(self.config, 'verified_img_dir', ''),
                                   verified_mask_dir=getattr(self.config, 'verified_mask_dir', ''))
            self._save_status_label.setText("  ✓  YOLO export done")
            QTimer.singleShot(3000, lambda: self._save_status_label.setText(""))
        except Exception as e:
            logger.warning(f"YOLO export failed: {e}")
            self._save_status_label.setText(f"  ✗  Export failed: {e}")
            QTimer.singleShot(4000, lambda: self._save_status_label.setText(""))

    def _show_help_overlay(self):
        """H — show shortcut cheatsheet popup."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard Shortcuts")
        dlg.setMinimumSize(500, 520)
        dlg.setStyleSheet(DARK_STYLESHEET)
        layout = QVBoxLayout(dlg)

        title = QLabel("⌨  AutoSegmentor Keyboard Shortcuts")
        title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {Colors.ACCENT_BLUE}; padding: 10px;")
        layout.addWidget(title)

        SHORTCUTS = """
<style>
  body  { background:#1e1e2e; color:#e0e0e0; font-family:'Segoe UI',sans-serif; font-size:10pt; }
  table { width:100%; border-collapse:collapse; }
  th    { background:#2b2b3c; color:#4fc3f7; padding:6px 10px; text-align:left; }
  td    { padding:5px 10px; border-bottom:1px solid #3d3d5c; }
  .key  { background:#3b3b4f; color:#ffa726; border-radius:3px;
           padding:1px 6px; font-family:'Cascadia Code',monospace; font-size:9pt; }
  .sec  { background:#1a1a2e; color:#66bb6a; font-weight:bold; }
</style>
<table>
  <tr><th>Action</th><th>Key(s)</th></tr>
  <tr class='sec'><td colspan='2'>🖼 Navigation</td></tr>
  <tr><td>Previous image</td>     <td><span class='key'>A</span> / <span class='key'>←</span></td></tr>
  <tr><td>Next image</td>         <td><span class='key'>D</span> / <span class='key'>→</span></td></tr>
  <tr><td>Previous batch</td>     <td><span class='key'>[</span></td></tr>
  <tr><td>Next batch</td>         <td><span class='key'>]</span></td></tr>
  <tr><td>Jump to frame</td>      <td>Type in Jump box + <span class='key'>Enter</span></td></tr>
  <tr class='sec'><td colspan='2'>✏️ Annotation</td></tr>
  <tr><td>Add foreground point</td><td><span class='key'>Ctrl+LClick</span></td></tr>
  <tr><td>Add background point</td><td><span class='key'>Ctrl+RClick</span></td></tr>
  <tr><td>Delete point</td>       <td><span class='key'>RClick</span> on point</td></tr>
  <tr><td>Move point</td>         <td><span class='key'>Shift+LClick</span> drag</td></tr>
  <tr><td>Undo</td>               <td><span class='key'>Ctrl+Z</span> / <span class='key'>U</span></td></tr>
  <tr><td>Redo</td>               <td><span class='key'>Ctrl+Y</span></td></tr>
  <tr><td>Reset all points</td>   <td><span class='key'>R</span></td></tr>
  <tr><td>Skip keypoint (pose)</td><td><span class='key'>Space</span></td></tr>
  <tr><td>Next instance</td>      <td><span class='key'>Tab</span></td></tr>
  <tr><td>Prev instance</td>      <td><span class='key'>Shift+Tab</span></td></tr>
  <tr><td>Set class 1–9</td>      <td><span class='key'>1</span>–<span class='key'>9</span></td></tr>
  <tr class='sec'><td colspan='2'>⚙️ Processing</td></tr>
  <tr><td>Process current batch</td><td><span class='key'>Enter</span></td></tr>
  <tr><td>Save progress</td>      <td><span class='key'>Ctrl+S</span></td></tr>
  <tr><td>Export YOLO dataset</td><td><span class='key'>Ctrl+E</span></td></tr>
  <tr class='sec'><td colspan='2'>👁 View</td></tr>
  <tr><td>Toggle mask overlay</td><td><span class='key'>M</span></td></tr>
  <tr><td>Toggle crosshair</td>   <td><span class='key'>C</span></td></tr>
  <tr><td>Toggle grid</td>        <td><span class='key'>G</span></td></tr>
  <tr><td>Toggle corner zoom</td> <td><span class='key'>Z</span></td></tr>
  <tr><td>Zoom in / out</td>      <td><span class='key'>+</span> / <span class='key'>-</span></td></tr>
  <tr><td>Pan canvas</td>         <td><span class='key'>LClick</span> drag</td></tr>
  <tr><td>Show shortcuts</td>     <td><span class='key'>H</span></td></tr>
  <tr><td>Exit without saving</td><td><span class='key'>Ctrl+Q</span></td></tr>
</table>
"""
        cheat = QTextEdit()
        cheat.setReadOnly(True)
        cheat.setHtml(SHORTCUTS)
        cheat.setStyleSheet(f"background:{Colors.BG_DARK}; border:none;")
        layout.addWidget(cheat, 1)

        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(dlg.accept)
        layout.addWidget(bb)
        dlg.exec_()

    def process_current_batch(self):
        """Intelligently process or reprocess the current batch."""
        if self.is_processing:
            return
            
        self.handler.save_current_annotation()
        batch = self.handler.current_frame_idx // self.config.batch_size
        current_frame = self.handler.current_frame_idx
        
        # If we are NOT on the first frame of the batch, it's a refinement (reprocess)
        if current_frame % self.config.batch_size != 0:
            logger.info(f"Refining batch {batch} from anchor frame {current_frame}...")
            self.start_processing_thread(batch, query_frame_idx=current_frame)
        else:
            logger.info(f"Processing full batch {batch}...")
            self.start_processing_thread(batch)

    def start_processing_thread(self, batch, query_frame_idx=None):
        self.is_processing = True
        self.processing_batch = batch
        self.btn_process_batch.setEnabled(False)
        self.btn_process_batch.setText("Processing...")
        
        # Show loader only if we are still viewing this batch
        if self.handler.current_frame_idx // self.config.batch_size == batch:
            self.loader_label.show()
            
        self.processor_thread = BatchProcessorThread(self.handler, batch, query_frame_idx=query_frame_idx)
        self.processor_thread.finished_batch.connect(self.on_processing_finished)
        self.processor_thread.start()
        
    def on_processing_finished(self, batch):
        self.is_processing = False
        self.btn_process_batch.setEnabled(True)
        self.btn_process_batch.setText("Process Batch")
        
        if self.handler.current_frame_idx // self.config.batch_size == batch:
            self.loader_label.hide()
            
        logger.info(f"Finished processing batch {batch}")
        
        # Stay on current frame after processing to allow manual inspection
        self.handler.load_frame_for_ui(self.handler.current_frame_idx)
