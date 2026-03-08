"""
MainWindow.py - Main PyQt dialog window for the AutoSegmenter annotation UI.
"""

import cv2
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QMenuBar, QMenu,
    QToolBar, QAction, QStatusBar, QLabel, QPushButton, QComboBox,
    QUndoStack, QShortcut, QWidget, QSizePolicy
)

from .UITheme import DARK_STYLESHEET, SIDEBAR_WIDTH, TOOLBAR_HEIGHT, STATUSBAR_HEIGHT
from .AnnotationCanvas import AnnotationCanvas
from .SidePanel import SidePanel
from .NavigationManager import AddPointCommand, ResetPointsCommand, DeletePointCommand, SkipPointCommand, DragPointCommand, NavigationState


class AnnotationWindow(QDialog):
    """Main AutoSegmenter annotation window. Runs modally to block pipeline."""

    def __init__(self, handler, config, parent=None):
        super().__init__(parent)
        self.handler = handler
        self.config = config
        
        # UI State
        self.nav_state = NavigationState()
        self.nav_state.show_crosshair = self.config.ui_show_crosshair if hasattr(self.config, 'ui_show_crosshair') else True
        self.nav_state.show_grid = self.config.ui_show_grid if hasattr(self.config, 'ui_show_grid') else False
        self.undo_stack = QUndoStack(self)
        self.undo_stack.setUndoLimit(30)
        
        self._init_ui()
        self._setup_shortcuts()
        self._connect_signals()
        
        # Load initial image and state
        self.refresh_display()
        self._update_sidebar()

    def _init_ui(self):
        """Initialize main layout and widgets."""
        self.setWindowTitle("AutoSegmenter Annotation Tool")
        self.setMinimumSize(1024, 768)
        # Apply dark theme
        self.setStyleSheet(DARK_STYLESHEET)
        
        # Force modal behavior
        self.setWindowModality(Qt.ApplicationModal)
        # Prevent escaping with 'Esc' without confirmation
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Menu Bar
        self.menu_bar = QMenuBar(self)
        self._build_menu()
        main_layout.addWidget(self.menu_bar)
        
        # Tool Bar
        self.tool_bar = QToolBar("Main Tools", self)
        self.tool_bar.setFixedHeight(TOOLBAR_HEIGHT)
        self.tool_bar.setMovable(False)
        self._build_toolbar()
        main_layout.addWidget(self.tool_bar)
        
        # Splitter (Canvas | Sidebar)
        self.splitter = QSplitter(Qt.Horizontal, self)
        
        # Canvas
        self.canvas = AnnotationCanvas(self)
        self.canvas.set_show_crosshair(self.nav_state.show_crosshair)
        self.canvas.set_show_grid(self.nav_state.show_grid)
        self.splitter.addWidget(self.canvas)
        
        # Sidebar
        self.sidebar = SidePanel(self)
        # Enable pose mode if config says so
        is_pose = self.handler.pose_mode
        if is_pose:
            self.sidebar.set_pose_mode(True, self.handler.pose_keypoints)
        else:
            self.sidebar.set_pose_mode(False)
        self.splitter.addWidget(self.sidebar)
        
        # Set splitter sizes (give most space to canvas)
        self.splitter.setSizes([800, SIDEBAR_WIDTH])
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        main_layout.addWidget(self.splitter, 1) # stretch factor 1
        
        # Status Bar
        self.status_bar = QStatusBar(self)
        self.status_bar.setFixedHeight(STATUSBAR_HEIGHT)
        self._build_statusbar()
        main_layout.addWidget(self.status_bar)

    def _build_menu(self):
        # File Menu
        file_menu = self.menu_bar.addMenu("&File")
        
        accept_action = QAction("Accept & Finish", self)
        accept_action.setShortcut("Return")
        accept_action.triggered.connect(self.accept_annotation)
        file_menu.addAction(accept_action)
        
        file_menu.addSeparator()
        
        quit_action = QAction("E&xit Without Saving", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.reject)
        file_menu.addAction(quit_action)
        
        # Edit Menu
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
        
        # View Menu
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

    def _build_toolbar(self):
        # Quick actions
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
        
        # Class Selection
        self.tool_bar.addWidget(QLabel("  Class: "))
        self.class_combo = QComboBox()
        for i in range(1, 11):
            self.class_combo.addItem(f"Class {i}", i)
        self.class_combo.setCurrentIndex(self.handler.current_class_label - 1)
        self.class_combo.currentIndexChanged.connect(self.change_class)
        self.tool_bar.addWidget(self.class_combo)
        if self.handler.pose_mode:
            self.class_combo.setEnabled(False) # Lock class in pose mode
            
        # Instance Selection
        self.tool_bar.addWidget(QLabel("  Inst: "))
        self.inst_lbl = QLabel(str(self.handler.current_instance_id))
        self.inst_lbl.setFixedWidth(20)
        self.tool_bar.addWidget(self.inst_lbl)
        
        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.tool_bar.addWidget(spacer)
        
        # Accept Action
        self.btn_accept = QPushButton("Accept (Enter)")
        self.btn_accept.setObjectName("acceptButton")
        self.btn_accept.clicked.connect(self.accept_annotation)
        self.tool_bar.addWidget(self.btn_accept)
        
    def _build_statusbar(self):
        self.coord_label = QLabel(" 📍 (0, 0)")
        self.coord_label.setObjectName("statusCoord")
        self.status_bar.addWidget(self.coord_label)
        
        # Spacer
        self.status_bar.addPermanentWidget(QLabel(" | Ctrl+LClick: Add (+) | Ctrl+RClick: Add (-) | RClick (on Pt): Del | LClick: Pan | Shift+LClick: Move | Space: Skip Pt | Enter: Accept | +/-: Zoom | M: Mask"))

    def _setup_shortcuts(self):
        # Class switching 1-9
        for i in range(1, 10):
            QShortcut(QKeySequence(str(i)), self, lambda checked, idx=i: self.set_class(idx))
            
        # Instance switching (Tab / Shift+Tab)
        QShortcut(QKeySequence("Tab"), self, self.next_instance)
        QShortcut(QKeySequence("Shift+Tab"), self, self.prev_instance)
        
        # Skip point
        QShortcut(QKeySequence("Space"), self, self.skip_point)
        
        # Zoom (+ / -)
        QShortcut(QKeySequence("+"), self, self.canvas.zoom_in)
        QShortcut(QKeySequence("="), self, self.canvas.zoom_in)
        QShortcut(QKeySequence("-"), self, self.canvas.zoom_out)
        
        # Toggle Mask Shortcut
        QShortcut(QKeySequence("M"), self, self.btn_toggle_mask.animateClick)

        # Undo fallback (U) to match old behavior
        QShortcut(QKeySequence("U"), self, self.undo_stack.undo)
        
        # Accept Shortcuts
        QShortcut(QKeySequence("Return"), self, self.btn_accept.animateClick)
        QShortcut(QKeySequence("Enter"), self, self.btn_accept.animateClick)

    def _connect_signals(self):
        self.canvas.point_clicked.connect(self.handle_canvas_click)
        self.canvas.mouse_moved.connect(self.handle_mouse_move)
        self.canvas.point_moved.connect(self.handle_point_moved)
        self.canvas.point_dragging.connect(self.handle_point_dragging)
        self.canvas.point_deleted.connect(self.handle_point_deleted)
        self.sidebar.keypoint_progress.visibility_toggled.connect(self.handle_visibility_toggled)
        self.undo_stack.indexChanged.connect(self._on_undo_stack_changed)

    # ─── Event Handlers ──────────────────────────────────────────────────────
    
    def handle_canvas_click(self, x, y, button):
        full_label = self.handler.encode_label(self.handler.current_class_label, self.handler.current_instance_id)
        if button == Qt.RightButton:
            full_label *= -1 # Negative prompt
            
        pose_click = None
        if self.handler.pose_mode:
            # Only record if we haven't reached max keypoints
            if self.handler.current_keypoint_index < len(self.handler.pose_keypoints):
                kp_name = self.handler.pose_keypoints[self.handler.current_keypoint_index]
                pose_click = {
                    "name": kp_name,
                    "point_id": self.handler.current_keypoint_index,
                    "x": int(x),
                    "y": int(y),
                    "visible": True
                }

        # Issue command to stack
        cmd = AddPointCommand(self.handler, [x, y], full_label, pose_click)
        self.undo_stack.push(cmd)  # This will also trigger _trigger_prompt_update via indexChanged

    def handle_point_deleted(self, index):
        """Handle deletion of a point from the canvas."""
        if index < len(self.handler.selected_points):
            cmd = DeletePointCommand(self.handler, index)
            self.undo_stack.push(cmd)

    def skip_point(self):
        """Skip the current point (mostly for pose mode out-of-frame keypoints)."""
        if self.handler.pose_mode and self.handler.current_keypoint_index < len(self.handler.pose_keypoints):
            cmd = SkipPointCommand(self.handler)
            self.undo_stack.push(cmd)

    def handle_mouse_move(self, x, y):
        self.nav_state.update_mouse(int(x), int(y))
        self.coord_label.setText(f" 📍 ({int(x)}, {int(y)})")

    def handle_point_moved(self, index, old_x, old_y, new_x, new_y):
        """Update point coordinates when dragged on canvas using undo stack."""
        if index < len(self.handler.selected_points):
            cmd = DragPointCommand(
                self.handler, 
                index, 
                [old_x, old_y], 
                [new_x, new_y]
            )
            self.undo_stack.push(cmd)  # Triggers _trigger_prompt_update via indexChanged

    def handle_point_dragging(self, index, x, y):
        """Live update point coordinates and skeleton during drag."""
        if index < len(self.handler.selected_points):
            self.handler.selected_points[index] = [x, y]
            if self.handler.pose_mode and self.handler.pose_click_coords:
                if index < len(self.handler.pose_click_coords):
                    self.handler.pose_click_coords[index]['x'] = int(x)
                    self.handler.pose_click_coords[index]['y'] = int(y)
            # Update visuals only (don't trigger SAM2 during drag for performance)
            self.canvas.update_skeleton(
                self.handler.selected_points,
                labels=self.handler.selected_labels,
                pose_coords=self.handler.pose_click_coords if self.handler.pose_mode else None
            )
            self.coord_label.setText(f" 📍 ({int(x)}, {int(y)})")

    def handle_visibility_toggled(self, index, is_visible):
        """Update the visibility flag for a pose keypoint."""
        if self.handler.pose_mode and self.handler.pose_click_coords:
            if index < len(self.handler.pose_click_coords):
                self.handler.pose_click_coords[index]['visible'] = is_visible
                # Update display to show current point properties
                self._trigger_prompt_update()

    # ─── Frame Updates ───────────────────────────────────────────────────────

    def refresh_display(self):
        """Update canvas with the current frame and annotations directly without invoking SAM2."""
        if self.btn_toggle_mask.isChecked():
            self.canvas.set_image(self.handler.current_frame_only_with_points)
        else:
            self.canvas.set_image(self.handler.current_frame)
        self._redraw_annotations()

    def _on_undo_stack_changed(self, idx):
        """Handle undo stack updates."""
        self._trigger_prompt_update()

    def _trigger_prompt_update(self):
        """Invoke SAM2 to update the mask based on current points."""
        # Using the handler's prompt adder. We pass None for parameter to tell handler we are in PyQt mode
        # UserInteraction logic will need to be adjusted slightly to not use cv2 bounds for UI
        self.handler.user_prompt_adder_pyqt()
        
        # Update canvas image with new mask
        if self.btn_toggle_mask.isChecked():
            self.canvas.update_image(self.handler.current_frame_only_with_points)
        else:
            self.canvas.update_image(self.handler.current_frame)
            
        self._redraw_annotations()
        self._update_sidebar()

    def _redraw_annotations(self):
        """Draw points and updates keypoints on canvas."""
        self.canvas.draw_annotations(
            self.handler.selected_points,
            self.handler.selected_labels,
            pose_keypoints=self.handler.pose_keypoints if self.handler.pose_mode else None,
            pose_coords=self.handler.pose_click_coords if self.handler.pose_mode else None,
            pose_config=self.config.pose_config if self.handler.pose_mode else None
        )
        
    def _update_sidebar(self):
        """Update side panel data."""
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
        """Called by pipeline to initialize frame info before showing."""
        self.nav_state.set_batch_info(batch, total_batches, frame_idx, total_frames)
        self.sidebar.batch_info.update_info(batch, total_batches, frame_idx, total_frames, 1.0)

    # ─── UI Actions ──────────────────────────────────────────────────────────

    def toggle_mask(self):
        """Toggle the visibility of SAM2 masks on the canvas."""
        if self.btn_toggle_mask.isChecked():
            # Show drawing with points/masks
            self.canvas.set_image(self.handler.current_frame_only_with_points)
        else:
            # Show original frame
            self.canvas.set_image(self.handler.current_frame)
        self._redraw_annotations()

    def set_class(self, class_id):
        """Set class index (1-9)."""
        if self.handler.pose_mode:
            return # Block class changes in pose mode
        self.class_combo.setCurrentIndex(class_id - 1)

    def change_class(self, index):
        """Handle combobox specific change."""
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
            self.undo_stack.push(cmd)  # Triggers update via indexChanged

    def toggle_crosshair(self):
        self.nav_state.show_crosshair = self.canvas.toggle_crosshair()

    def toggle_grid(self):
        self.nav_state.show_grid = self.canvas.toggle_grid()
        
    def toggle_corner_zoom(self):
        self.canvas.zoom_view.setVisible(self.zoom_view_action.isChecked())

    def accept_annotation(self):
        """Triggers the close flow, signifying user is done."""
        # Instead of managing the appending logic here, we let the closing dialog return control
        # to UserInteraction which handles the append logic.
        self.accept()
