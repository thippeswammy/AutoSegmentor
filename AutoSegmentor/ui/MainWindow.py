"""
MainWindow.py - Main PyQt dialog window for the AutoSegmenter annotation UI.
"""

import traceback

import cv2
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QKeySequence, QFont
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QMenuBar, QMenu,
    QToolBar, QAction, QStatusBar, QLabel, QPushButton, QComboBox,
    QUndoStack, QShortcut, QWidget, QSizePolicy, QTextEdit, QDialogButtonBox,
    QMessageBox
)

from .UITheme import DARK_STYLESHEET, SIDEBAR_WIDTH, TOOLBAR_HEIGHT, STATUSBAR_HEIGHT, Colors
from .AnnotationCanvas import AnnotationCanvas
from .SidePanel import SidePanel
from .NavigationManager import AddPointCommand, ResetPointsCommand, DeletePointCommand, SkipPointCommand, DragPointCommand, NavigationState
from .ExportDialog import ExportDialog
from .logger_config import logger

class BatchProcessorThread(QThread):
    finished_batch = pyqtSignal(int)

    def __init__(self, handler, batch, query_frame_idx=None, backward_tracking=False):
        super().__init__()
        self.handler = handler
        self.batch = batch
        self.query_frame_idx = query_frame_idx
        self.backward_tracking = backward_tracking

    def run(self):
        """Run SAM2 mask generation + CoTracker in the background thread."""
        try:
            processor = self.handler.pipeline_processor

            batch_index = self.batch * processor.config.batch_size
            processor.frame_handler.move_and_copy_frames(batch_index, processor.frame_paths, processor.config.batch_size)

            if processor.config.sam_enabled:
                logger.debug(f"[BatchThread] Batch {self.batch}: Running SAM2 mask generation...")
                processor.mask_processor.generate_mask(
                    batch_number=self.batch,
                    sam2_predictor=processor.sam2_predictor,
                    temp_directory=processor.config.temp_directory,
                    prompt_encoding=processor.prompt_encoding,
                    auto_prompt_encoding=processor.auto_prompt_encoding,
                    predictor_lock=processor._predictor_lock,
                    starting_frame_idx=batch_index
                )
                logger.debug(f"[BatchThread] Batch {self.batch}: SAM2 done.")

            if (processor.config.pose_config and processor.config.pose_config.get('enabled')
                    and processor.config.pose_config.get('tracker', 'lk').lower() == 'cotracker'):
                logger.debug(f"[BatchThread] Batch {self.batch}: Running CoTracker...")
                processor._track_batch_inline(self.batch, query_frame_idx=self.query_frame_idx, backward_tracking=self.backward_tracking)
                logger.debug(f"[BatchThread] Batch {self.batch}: CoTracker done.")
                
                # Persistence: Save tracked keypoints to JSON continuously
                tracked_data = processor.per_batch_tracked_data[self.batch]
                if tracked_data:
                    self.handler.annotation_manager.save_tracked_batch(tracked_data, self.batch)

            self.finished_batch.emit(self.batch)
        except Exception:
            logger.error(
                f"[BatchThread] Batch {self.batch} CRASHED:\n{traceback.format_exc()}"
            )
            # Still emit so the UI unlocks
            self.finished_batch.emit(self.batch)


class PreviewThread(QThread):
    """Runs SAM2 single-frame mask preview off the main thread.

    Emitting preview_ready triggers the UI to update the image overlay
    without blocking the Qt event loop.
    """
    preview_ready = pyqtSignal()  # Fires when current_frame is updated

    def __init__(self, handler):
        super().__init__()
        self.handler = handler

    def run(self):
        import time
        start_t = time.perf_counter()
        try:
            logger.debug("[Preview] SAM2 preview started")
            self.handler.user_prompt_adder_pyqt()
            elapsed = time.perf_counter() - start_t
            logger.debug(f"[Preview] SAM2 preview done in {elapsed:.3f}s")
        except Exception:
            logger.error(f"[Preview] SAM2 preview failed:\n{traceback.format_exc()}")
        finally:
            self.preview_ready.emit()

class AnnotationWindow(QDialog):
    """Main AutoSegmenter annotation window. Runs modally, handles bg processing."""

    def __init__(self, handler, config, parent=None):
        super().__init__(parent)
        self.handler = handler
        self.config = config
        self.is_processing = False
        self._preview_thread = None  # Background SAM2 preview thread
        self._preview_pending = False # Queue flag for high-responsiveness

        # Debounce timer: SAM preview fires 500ms after user STOPS navigating.
        # This prevents SAM inference from running on every single frame while
        # the user holds A/D to scroll — only runs when they pause.
        self._preview_debounce_timer = QTimer(self)
        self._preview_debounce_timer.setSingleShot(True)
        self._preview_debounce_timer.setInterval(500)  # ms after last nav key
        self._preview_debounce_timer.timeout.connect(self._start_preview_thread)

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
        # Fires every 50 ms while A/D/←/→ is held down (20 FPS)
        self._nav_timer = QTimer(self)
        self._nav_timer.setInterval(50) 
        self._nav_timer.timeout.connect(self._on_nav_timer)
        self._nav_direction = 0   # -1 = prev, +1 = next, 0 = idle
        self._nav_is_turbo = False 

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
        self.loader_label = QLabel("Processing...", self.canvas)
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

        finish_action = QAction("&Save and Finish", self)
        finish_action.setShortcut("Ctrl+Return")
        finish_action.setStatusTip("Save current progress and finish the manual annotation phase")
        finish_action.triggered.connect(self._on_finish_pipeline)
        file_menu.addAction(finish_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit Without Saving", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.setStatusTip("Close the tool without saving the current frame")
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

    def closeEvent(self, event):
        """Intercept window close to prevent accidental data loss and orphan threads."""
        if self.is_processing:
            reply = QMessageBox.question(
                self, 'Background Task Running',
                "A background process (SAM2/CoTracker) is currently active.\n\n"
                "Closing the window now may interrupt this process and cause data loss.\n"
                "Force exit anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            else:
                # Forcefully terminate processor thread if it exists
                if hasattr(self, 'processor_thread') and self.processor_thread.isRunning():
                    logger.warning("Terminating background processor thread due to forced exit.")
                    self.processor_thread.terminate()
                    self.processor_thread.wait()

        # Check for unsaved changes on the current frame
        is_dirty = self.undo_stack.canUndo() or self.handler.selected_points
        if is_dirty:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Unsaved Progress")
            msg_box.setText("You have unsaved changes on the current frame.")
            msg_box.setInformativeText("Would you like to save your progress before closing?")
            msg_box.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            msg_box.setDefaultButton(QMessageBox.Save)
            
            # Make the dialog style match the theme
            msg_box.setStyleSheet(DARK_STYLESHEET)
            
            reply = msg_box.exec_()

            if reply == QMessageBox.Save:
                self.handler.save_current_annotation()
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

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
            QShortcut(QKeySequence(str(i)), self, lambda idx=i: self.set_class(idx))

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

        # Save / Export / Help / Quit
        QShortcut(QKeySequence("Ctrl+S"), self, self._on_save_progress)
        QShortcut(QKeySequence("Ctrl+E"), self, self._on_export_yolo)
        QShortcut(QKeySequence("Ctrl+W"), self, self.close)  # Standard Window Close
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
        self.sidebar.live_config.backward_tracking_toggled.connect(self._on_backward_tracking_toggled)
        self.sidebar.export_requested.connect(self._on_export_yolo)
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
        modifiers = event.modifiers()
        if key in self._NAV_KEYS and not event.isAutoRepeat() and not self._is_text_widget_focused():
            direction = -1 if key in (Qt.Key_Left, Qt.Key_A) else 1
            self._nav_is_turbo = bool(modifiers & Qt.ShiftModifier)
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
        """Called while a navigation key is held."""
        # Update turbo state dynamically if modifiers change while holding
        self._nav_is_turbo = bool(QtCore.QCoreApplication.instance().keyboardModifiers() & Qt.ShiftModifier)
        
        step = 5 if self._nav_is_turbo else 1
        if self._nav_direction == -1:
            for _ in range(step): self.prev_image()
        elif self._nav_direction == 1:
            for _ in range(step): self.next_image()
        else:
            self._nav_timer.stop()

    # ─── Navigation Handlers ─────────────────────────────────────────────────
    def prev_image(self):
        idx = max(0, self.handler.current_frame_idx - 1)
        logger.debug(f"[Nav] prev_image: {self.handler.current_frame_idx} -> {idx}")
        self.handler.load_frame_for_ui(idx)

    def next_image(self):
        idx = min(len(self.handler.frame_paths) - 1, self.handler.current_frame_idx + 1)
        logger.debug(f"[Nav] next_image: {self.handler.current_frame_idx} -> {idx}")
        self.handler.load_frame_for_ui(idx)

    def prev_batch(self):
        idx = max(0, self.handler.current_frame_idx - self.config.batch_size)
        idx = (idx // self.config.batch_size) * self.config.batch_size
        logger.debug(f"[Nav] prev_batch: {self.handler.current_frame_idx} -> {idx}")
        self.handler.load_frame_for_ui(idx)

    def next_batch(self):
        idx = min(len(self.handler.frame_paths) - 1, self.handler.current_frame_idx + self.config.batch_size)
        idx = (idx // self.config.batch_size) * self.config.batch_size
        logger.debug(f"[Nav] next_batch: {self.handler.current_frame_idx} -> {idx}")
        self.handler.load_frame_for_ui(idx)

    # ─── Event Handlers ──────────────────────────────────────────────────────
    def handle_canvas_click(self, x, y, button):
        full_label = self.handler.encode_label(self.handler.current_class_label, self.handler.current_instance_id)
        btn_name = "RightButton" if button == Qt.RightButton else "LeftButton"
        logger.debug(f"[UI] handle_canvas_click: ({x},{y})  button={btn_name}  label_before_sign={full_label}")
        if button == Qt.RightButton:
            full_label *= -1

        pose_click = None
        if self.handler.pose_mode:
            num_kps = len(self.handler.pose_keypoints)
            if num_kps > 0:
                if full_label < 0:
                    kp_name = "Negative_Point"
                    p_id = len(self.handler.pose_click_coords)
                else:
                    instance_count = self.handler.get_instance_keypoint_count()
                    # Auto-advance instance when current one is complete
                    if instance_count > 0 and instance_count % num_kps == 0:
                        self.handler.current_instance_id += 1
                        self.handler._recalc_keypoint_index()
                        full_label = self.handler.encode_label(
                            self.handler.current_class_label, self.handler.current_instance_id
                        )
                        instance_count = 0
                        self._update_sidebar()
                    kp_name = self.handler.pose_keypoints[instance_count % num_kps]
                    p_id = len(self.handler.pose_click_coords)
                pose_click = {
                    "name": kp_name,
                    "point_id": p_id,
                    "x": int(x),
                    "y": int(y),
                    "visible": True,
                    "label": full_label
                }
                logger.debug(f"[UI] handle_canvas_click: pose_click={pose_click}")

        logger.debug(f"[UI] Pushing AddPointCommand: point=[{x},{y}]  label={full_label}")
        cmd = AddPointCommand(self.handler, [x, y], full_label, pose_click)
        self.undo_stack.push(cmd)

    def handle_point_deleted(self, index):
        if index < len(self.handler.selected_points):
            logger.debug(f"[UI] handle_point_deleted: index={index}  point={self.handler.selected_points[index]}")
            cmd = DeletePointCommand(self.handler, index)
            self.undo_stack.push(cmd)

    def skip_point(self):
        if self.handler.pose_mode and self.handler.current_keypoint_index < len(self.handler.pose_keypoints):
            logger.debug(f"[UI] skip_point: kp_index={self.handler.current_keypoint_index}")
            cmd = SkipPointCommand(self.handler)
            self.undo_stack.push(cmd)

    def handle_mouse_move(self, x, y):
        self.nav_state.update_mouse(int(x), int(y))
        self.coord_label.setText(f" 📍 ({int(x)}, {int(y)})")

    def handle_point_moved(self, index, old_x, old_y, new_x, new_y):
        if index < len(self.handler.selected_points):
            logger.debug(f"[UI] handle_point_moved: index={index}  ({old_x},{old_y}) -> ({new_x},{new_y})")
            cmd = DragPointCommand(self.handler, index, [old_x, old_y], [new_x, new_y])
            self.undo_stack.push(cmd)
            logger.debug(f"[UI] handle_point_moved: DragPointCommand pushed, triggering SAM preview")
            self._trigger_prompt_update()

    def handle_point_dragging(self, index, x, y):
        if index < len(self.handler.selected_points):
            logger.debug(f"[UI] handle_point_dragging: index={index}  pos=({x},{y})")
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
                logger.debug(f"[UI] handle_visibility_toggled: index={index}  visible={is_visible}")
                self.handler.pose_click_coords[index]['visible'] = is_visible
                self._trigger_prompt_update()

    # ─── Frame Updates ───────────────────────────────────────────────────────
    def refresh_display(self):
        """Called on every frame navigation. Instantly updates image + skeleton/points.

        SAM preview debounce rules:
        - PROCESSED frame (mask file exists on disk):
            Preview ONLY fires on point click/change (_trigger_prompt_update).
            Navigation never auto-triggers SAM — the mask is already rendered.
        - UNPROCESSED frame (no mask yet):
            Preview auto-fires 500ms after user stops navigating, so they can
            see a live SAM preview without needing to click first.
        """
        if self.btn_toggle_mask.isChecked():
            self.canvas.set_image(self.handler.current_frame_only_with_points)
        else:
            self.canvas.set_image(cv2.imread(self.handler.frame_paths[self.handler.current_frame_idx]))
        # Draw skeleton + points immediately (no GPU needed)
        self._redraw_annotations()

        # Only auto-debounce SAM preview on UNPROCESSED frames with points
        if self.handler.selected_points and not self._frame_is_processed():
            self._preview_debounce_timer.start()
        else:
            # Cancel any pending debounce from a previous unprocessed frame
            self._preview_debounce_timer.stop()

    def _frame_is_processed(self) -> bool:
        """Return True if the current frame already has data (mask, tracked, or manual prompt).

        Delegates to handler.has_data_for_frame which correctly checks:
        - Rendered mask file on disk
        - CoTracker per_batch_tracked_data
        - Manual annotation prompts
        """
        return self.handler.has_data_for_frame(self.handler.current_frame_idx)


    def _on_undo_stack_changed(self, idx):
        logger.debug(f"[UI] undo_stack index changed to {idx}  — triggering preview update")
        self._trigger_prompt_update()

    def _trigger_prompt_update(self):
        """Immediately redraw skeleton/points, then start async SAM2 mask preview.

        The skeleton and annotation points are drawn synchronously (instant).
        The SAM2 mask overlay runs in a background PreviewThread so the UI
        never freezes while waiting for GPU inference.
        """
        logger.debug(
            f"[UI] _trigger_prompt_update: frame={self.handler.current_frame_idx}  "
            f"pts={len(self.handler.selected_points)}"
        )
        # 1. Immediately draw annotations and skeleton — no GPU needed, instant
        self._redraw_annotations()
        self._update_sidebar()

        # 2. Start async SAM2 preview in background thread
        logger.debug(f"[UI] Triggering SAM2 preview for frame {self.handler.current_frame_idx}")
        self._start_preview_thread()

    def _start_preview_thread(self):
        """Kick off a SAM2 single-frame preview in a background thread."""
        if self._preview_thread is not None and self._preview_thread.isRunning():
            self._preview_pending = True
            logger.debug("[Preview] Busy — queuing update for latest state")
            return

        logger.debug(f"[Preview] Starting PreviewThread for frame {self.handler.current_frame_idx}")
        self._preview_pending = False
        self.status_bar.showMessage(" ⏳ Updating Mask...", 5000)
        self._preview_thread = PreviewThread(self.handler)
        self._preview_thread.preview_ready.connect(self._on_preview_ready)
        self._preview_thread.start()

    def _on_preview_ready(self):
        """Called on the main thread when SAM2 preview is complete."""
        logger.debug(f"[Preview] _on_preview_ready: frame={self.handler.current_frame_idx}  pending={self._preview_pending}")
        self.status_bar.clearMessage()

        # High-Responsiveness: if a preview request arrived while we were busy,
        # run another one now with the very latest positions.
        if self._preview_pending:
            logger.debug("[Preview] Running pending update...")
            self._start_preview_thread()
        else:
            # Show final result: mask overlay if toggle is ON, raw frame if OFF
            if self.btn_toggle_mask.isChecked():
                logger.debug("[Preview] Showing SAM mask overlay on canvas")
                self.canvas.update_image(self.handler.current_frame)
            else:
                logger.debug("[Preview] Mask toggle OFF — refreshing canvas from disk frame")
                self.canvas.update_image(
                    cv2.imread(self.handler.frame_paths[self.handler.current_frame_idx])
                )
        # Re-draw annotations on top of the updated image
        self._redraw_annotations()



    def _redraw_annotations(self):
        self.canvas.draw_annotations(
            self.handler.selected_points,
            self.handler.selected_labels,
            pose_keypoints=self.handler.pose_keypoints if self.handler.pose_mode else None,
            pose_coords=self.handler.pose_click_coords if self.handler.pose_mode else None,
            pose_config=self.config.pose_config if self.handler.pose_mode else None
        )
        
    def _update_sidebar(self):
        pose_coords = self.handler.pose_click_coords if self.handler.pose_mode else None
        self.sidebar.annotation_list.update_annotations(
            self.handler.selected_points, 
            self.handler.selected_labels,
            pose_coords=pose_coords
        )
        
        mode_str = "Pose" if self.handler.pose_mode else "Segment"
        self.sidebar.tool_info.update_info(
            mode_str, 
            self.handler.current_class_label, 
            self.handler.current_instance_id
        )
        self.inst_lbl.setText(str(self.handler.current_instance_id))
        
        if self.handler.pose_mode:
            num_kps = len(self.handler.pose_keypoints)
            # Count keypoints for the CURRENT class+instance only
            instance_kp_count = self.handler.get_instance_keypoint_count()
            # Clamp to num_kps so a full set shows N/N, not 0/N
            current_mod = min(instance_kp_count, num_kps) if num_kps > 0 else 0
            
            # Find pose_coords belonging to current instance for visibility toggles
            target_label = self.handler.encode_label(
                self.handler.current_class_label, self.handler.current_instance_id
            )
            current_coords = [
                pc for pc in self.handler.pose_click_coords
                if pc.get('name') != 'Negative_Point'
                and abs(pc.get('label', 0)) == target_label
            ]
            
            self.sidebar.keypoint_progress.update_progress(
                current_mod,
                current_coords
            )
            self.sidebar.keypoint_progress.set_current(current_mod)

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
        logger.debug(f"[UI] toggle_mask: checked={self.btn_toggle_mask.isChecked()}")
        self.refresh_display()

    def set_class(self, class_id):
        logger.debug(f"[UI] set_class: {class_id}")
        self.class_combo.setCurrentIndex(class_id - 1)

    def change_class(self, index):
        logger.debug(f"[UI] change_class: combo_index={index}  -> class_id={index + 1}")
        self.handler.change_class_label_pyqt(index + 1)
        # Rebuild keypoint panel rows for the new class's keypoint list
        if self.handler.pose_mode:
            self.sidebar.keypoint_progress.setup_keypoints(self.handler.pose_keypoints)
        self._update_sidebar()

    def next_instance(self):
        self.handler.current_instance_id += 1
        self.handler._recalc_keypoint_index()
        logger.debug(f"[UI] next_instance: instance_id={self.handler.current_instance_id}  kp_index={self.handler.current_keypoint_index}")
        self._update_sidebar()

    def prev_instance(self):
        if self.handler.current_instance_id > 1:
            self.handler.current_instance_id -= 1
            self.handler._recalc_keypoint_index()
            logger.debug(f"[UI] prev_instance: instance_id={self.handler.current_instance_id}  kp_index={self.handler.current_keypoint_index}")
            self._update_sidebar()

    def reset_points(self):
        """Clear all annotation state for the current frame and show a clean image.

        Clears:
        - selected_points / selected_labels
        - pose_click_coords / current_keypoint_index
        - inference_state_temp (force fresh SAM init on next click)
        - Canvas image → shows _raw_frame (no disk mask overlay, no SAM mask)
        - Skeleton / point overlays on canvas

        The user can then re-annotate from scratch or navigate away.
        """
        h = self.handler

        # Clear all annotation state
        h.selected_points = []
        h.selected_labels = []
        h.pose_click_coords = []
        h.current_keypoint_index = 0
        h.inference_state_temp = None  # Force fresh init on next click

        # Show the clean raw frame (no disk mask, no SAM mask)
        raw = getattr(h, '_raw_frame', None)
        clean = raw.copy() if raw is not None else __import__('cv2').imread(
            h.frame_paths[h.current_frame_idx])
        h.current_frame = clean
        h.current_frame_only_with_points = clean.copy()

        # Update canvas: raw image + clear all overlays
        self.canvas.update_image(clean)
        self.canvas.draw_annotations([], [], pose_keypoints=None, pose_coords=None, pose_config=None)
        self._update_sidebar()
        logger.debug(f"[Reset] Frame {h.current_frame_idx} cleared")


    def toggle_crosshair(self):
        self.nav_state.show_crosshair = self.canvas.toggle_crosshair()

    def toggle_grid(self):
        self.nav_state.show_grid = self.canvas.toggle_grid()
        
    def toggle_corner_zoom(self):
        self.canvas.zoom_view.setVisible(self.zoom_view_action.isChecked())

    def jump_to_frame(self):
        text = self.frame_jump_input.text()
        logger.debug(f"[UI] jump_to_frame: input='{text}'")
        try:
            val = int(text)
            if 0 <= val < len(self.handler.frame_paths):
                logger.debug(f"[UI] jump_to_frame: jumping to frame {val}")
                self.handler.load_frame_for_ui(val)
                self.frame_jump_input.clear()
            else:
                logger.warning(f"[UI] jump_to_frame: invalid frame index {val} (total={len(self.handler.frame_paths)})")
        except ValueError:
            logger.debug(f"[UI] jump_to_frame: non-integer input '{text}', ignoring")
            
    def _on_backward_tracking_toggled(self, checked):
        self.nav_state.propagate_backward = checked
        logger.debug(f"[UI] Propagate backward toggled: {checked}")

    # ─── New Action Handlers ──────────────────────────────────────────────────

    def _on_save_progress(self):
        """Ctrl+S — save current annotation and flash status bar."""
        self.handler.save_current_annotation()
        self._save_status_label.setText("  ✓  Saved")
        QTimer.singleShot(2000, lambda: self._save_status_label.setText(""))
        logger.info("Progress saved by user (Ctrl+S).")

    def _on_finish_pipeline(self):
        """Save current frame and exit pipeline phase with success."""
        self.handler.save_current_annotation()
        logger.info("Save and Finish triggered. Pipeline phase accepted.")
        self.accept()

    def _on_export_yolo(self):
        """Ctrl+E — show YOLO dataset export dialog."""
        dlg = ExportDialog(self, self.handler)
        dlg.exec_()

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
            logger.debug("[UI] process_current_batch: already processing, ignoring")
            return

        logger.debug(f"[UI] process_current_batch: frame={self.handler.current_frame_idx}")
        self.handler.save_current_annotation()
        batch = self.handler.current_frame_idx // self.config.batch_size
        current_frame = self.handler.current_frame_idx
        logger.debug(f"[UI] process_current_batch: batch={batch}  current_frame={current_frame}")

        # Check if the batch has any prompts (manual or tracked carry-forward)
        batch_prompts = self.handler.annotation_manager.get_batch_prompts(batch, self.config.batch_size)
        has_prev_tracked = (
            batch > 0
            and hasattr(self.handler.pipeline_processor, 'per_batch_tracked_data')
            and batch - 1 < len(self.handler.pipeline_processor.per_batch_tracked_data)
            and bool(self.handler.pipeline_processor.per_batch_tracked_data[batch - 1])
        )

        if not batch_prompts and not has_prev_tracked:
            msg = QMessageBox(self)
            msg.setWindowTitle("No Prompts Found")
            msg.setIcon(QMessageBox.Warning)
            msg.setText(
                f"<b>Batch {batch + 1}</b> has no annotation prompts and no tracked data to carry forward.\n\n"
                "Please annotate at least one frame in this batch before processing."
            )
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Ignore)
            msg.button(QMessageBox.Ignore).setText("Process Anyway")
            if msg.exec_() == QMessageBox.Ok:
                logger.info(f"[UI] Batch {batch} processing cancelled — no prompts.")
                return
            else:
                logger.warning(f"[UI] Batch {batch} being processed with no prompts (user override).")

        # If we are NOT on the first frame of the batch, it's a refinement (reprocess)
        if current_frame % self.config.batch_size != 0:
            logger.info(f"Refining batch {batch} from anchor frame {current_frame}...")
            self.start_processing_thread(batch, query_frame_idx=current_frame, backward_tracking=self.nav_state.propagate_backward)
        else:
            logger.info(f"Processing full batch {batch}...")
            self.start_processing_thread(batch, backward_tracking=self.nav_state.propagate_backward)

    def start_processing_thread(self, batch, query_frame_idx=None, backward_tracking=False):
        logger.debug(f"[UI] start_processing_thread: batch={batch}  query_frame_idx={query_frame_idx} backward_tracking={backward_tracking}")
        self.is_processing = True
        self.processing_batch = batch
        self.btn_process_batch.setEnabled(False)
        self.btn_process_batch.setText("Processing...")

        # Show loader only if we are still viewing this batch
        if self.handler.current_frame_idx // self.config.batch_size == batch:
            self.loader_label.show()

        self.processor_thread = BatchProcessorThread(self.handler, batch, query_frame_idx=query_frame_idx, backward_tracking=backward_tracking)
        self.processor_thread.finished_batch.connect(self.on_processing_finished)
        self.processor_thread.start()
        logger.debug(f"[UI] BatchProcessorThread started for batch {batch}")
        
    def on_processing_finished(self, batch):
        logger.debug(f"[UI] on_processing_finished: batch={batch}  current_frame={self.handler.current_frame_idx}")
        self.is_processing = False
        self.btn_process_batch.setEnabled(True)
        self.btn_process_batch.setText("Process Batch")

        if self.handler.current_frame_idx // self.config.batch_size == batch:
            self.loader_label.hide()

        logger.info(f"Finished processing batch {batch}")

        # Explicitly reload current frame to show new masks
        logger.debug(f"[UI] on_processing_finished: reloading frame {self.handler.current_frame_idx}")
        self.handler.load_frame_for_ui(self.handler.current_frame_idx)
        self.canvas.update_image(self.handler.current_frame)
        self.refresh_display()
        self._update_sidebar()
        logger.debug(f"[UI] on_processing_finished: display fully refreshed for batch {batch}")
