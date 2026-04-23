"""
SetupDialog.py — Launch dialog for AutoSegmentor.

Shows before any processing begins. Provides:
  Tab 1 — Videos  : input paths, indices, output/working dir, cleanup options
  Tab 2 — Settings: processing params + tracker/pose config (merged)

All fields are pre-loaded from session_state.json (or default_config.yaml on
first run) and saved back on every "Start" or "Save as Defaults" click.
"""

import json
import os
import sys
import yaml
from .logger_config import logger
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QGroupBox, QFormLayout, QLabel, QLineEdit, QSpinBox,
    QDoubleSpinBox, QCheckBox, QComboBox, QRadioButton,
    QButtonGroup, QPushButton, QFileDialog, QListWidget,
    QListWidgetItem, QAbstractItemView, QSizePolicy,
    QDialogButtonBox, QMessageBox, QScrollArea, QFrame,
    QSplitter, QApplication, QToolButton
)

from .UITheme import DARK_STYLESHEET, Colors, Fonts

# ─── Paths ────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_WORKSPACE = os.path.join(_ROOT, "workspace")
_DEFAULT_CONFIG = os.path.join(_WORKSPACE, "inputs", "config", "default_config.yaml")
_SESSION_STATE  = os.path.join(_WORKSPACE, "inputs", "config", "session_state.json")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_yaml(path):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _load_session():
    """Load session_state.json; fall back to default YAML on first run."""
    if os.path.exists(_SESSION_STATE):
        try:
            with open(_SESSION_STATE, "r") as f:
                data = json.load(f)
                logger.debug(f"[Setup] Loaded session from {_SESSION_STATE}")
                return data
        except Exception as e:
            logger.error(f"[Setup] Error loading session: {e}")
    logger.debug(f"[Setup] Session file not found or failed, loading defaults from {_DEFAULT_CONFIG}")
    return _load_yaml(_DEFAULT_CONFIG)


def _save_session(data: dict):
    os.makedirs(os.path.dirname(_SESSION_STATE), exist_ok=True)
    
    # Construct nested structure
    nested = {
        "external_libs": data.get("external_libs", []),
        "video_inputs": {
            "template":   data.get("video_path_template", ""),
            "start":      data.get("video_start", 1),
            "end":        data.get("video_end", 1),
            "max_frames": data.get("images_ending_count", 0)
        },
        "video_outputs": {
            "final_path":  data.get("final_video_path", "./workspace/outputs"),
            "working_dir": data.get("working_dir_name", "./workspace/working_dir"),
            "prefix":      data.get("prefix", "Img"),
            "delete_after": data.get("delete", False)
        },
        "pipeline": {
            "run_mode":          data.get("run_mode", "all"),
            "batch_size":        data.get("batch_size", 30),
            "fps":               data.get("fps", 30),
            "review_from_start": data.get("review_from_start", False),
            "auto_prompt":       data.get("auto_prompt_encoding", True)
        },
        "models": {
            "sam": {
                "enabled":      data.get("sam_enabled", True),
                "checkpoint":   data.get("sam_checkpoint", ""),
                "model_config": data.get("sam_model_config", "sam2_hiera_l.yaml")
            },
            "pose": {
                "enabled":   data.get("pose_enabled", False),
                "tracker":   data.get("tracker", "cotracker"),
                "radius":    data.get("keypoint_radius", 5),
                "classes": [{
                    "class_id":  data.get("pose_class_id", 1),
                    "object_id": data.get("pose_object_id", 1),
                    "keypoints": data.get("keypoints", [])
                }],
                "cotracker": {
                    "checkpoint": data.get("cotracker_checkpoint", ""),
                    "window_len": data.get("cotracker_window_len", 60)
                }
            }
        }
    }

    with open(_SESSION_STATE, "w") as f:
        json.dump(nested, f, indent=2)
    logger.debug(f"[Setup] Saved session to {_SESSION_STATE}")


def _save_defaults(data: dict):
    """Write changed values back to default_config.yaml in nested format."""
    cfg = _load_yaml(_DEFAULT_CONFIG)
    
    # Update nested structures
    cfg["video_inputs"] = {
        "template":   data.get("video_path_template", ""),
        "start":      data.get("video_start", 1),
        "end":        data.get("video_end", 1),
        "max_frames": data.get("images_ending_count", 0)
    }
    cfg["video_outputs"] = {
        "final_path":  data.get("final_video_path", "./workspace/outputs"),
        "working_dir": data.get("working_dir_name", "./workspace/working_dir"),
        "prefix":      data.get("prefix", "Img"),
        "delete_after": data.get("delete", False)
    }
    cfg["pipeline"] = {
        "run_mode":          data.get("run_mode", "all"),
        "batch_size":        data.get("batch_size", 30),
        "fps":               data.get("fps", 30),
        "review_from_start": data.get("review_from_start", False),
        "auto_prompt":       data.get("auto_prompt_encoding", True)
    }
    
    mods = cfg.get("models", {})
    sam = mods.get("sam", {})
    sam["enabled"] = data.get("sam_enabled", sam.get("enabled", True))
    
    pose = mods.get("pose", {})
    pose["enabled"] = data.get("pose_enabled", False)
    pose["tracker"] = data.get("tracker", "cotracker")
    pose["radius"]  = data.get("keypoint_radius", 5)
    
    classes = pose.get("classes", [{}])
    classes[0]["class_id"] = data.get("pose_class_id", 1)
    classes[0]["object_id"] = data.get("pose_object_id", 1)
    classes[0]["keypoints"] = data.get("keypoints", [])
    pose["classes"] = classes
    
    ct = pose.get("cotracker", {})
    ct["checkpoint"] = data.get("cotracker_checkpoint", "")
    ct["window_len"] = data.get("cotracker_window_len", 60)
    pose["cotracker"] = ct
    
    cfg["models"] = {"sam": sam, "pose": pose}

    os.makedirs(os.path.dirname(_DEFAULT_CONFIG), exist_ok=True)
    _write_config_with_sections(cfg, _DEFAULT_CONFIG)


def _write_config_with_sections(cfg, filepath):
    """Write configuration with logical sections based on nested structure."""
    order = [
        ("External Libraries", ["external_libs"]),
        ("Video Inputs", ["video_inputs"]),
        ("Video Outputs & Storage", ["video_outputs"]),
        ("Pipeline Settings", ["pipeline"]),
        ("Models", ["models"])
    ]
    
    lines = []
    processed_keys = set()
    
    for section_name, keys in order:
        section_dict = {k: cfg[k] for k in keys if k in cfg}
        if not section_dict: continue
        processed_keys.update(keys)
        
        lines.append(f"# {'=' * 40}")
        lines.append(f"# {section_name}")
        lines.append(f"# {'=' * 40}")
        lines.append(yaml.dump(section_dict, default_flow_style=False, allow_unicode=True).strip())
        lines.append("")

    # Any leftover keys
    other = {k: v for k, v in cfg.items() if k not in processed_keys}
    if other:
        lines.append(f"# {'=' * 40}")
        lines.append(f"# Other Configs")
        lines.append(f"# {'=' * 40}")
        lines.append(yaml.dump(other, default_flow_style=False, allow_unicode=True).strip())
        lines.append("")
        
    with open(filepath, "w") as f:
        f.write("\n".join(lines))


# ─── Styled sub-widgets ───────────────────────────────────────────────────────

def _section(title):
    box = QGroupBox(title)
    box.setFont(Fonts.header())
    return box


def _form_row(label_text, widget, tooltip=None):
    lbl = QLabel(label_text)
    lbl.setFont(Fonts.body())
    if tooltip:
        widget.setToolTip(tooltip)
        lbl.setToolTip(tooltip)
    return lbl, widget


def _browse_btn(callback):
    btn = QToolButton()
    btn.setText("…")
    btn.setFixedWidth(28)
    btn.clicked.connect(callback)
    return btn


def _path_row(placeholder="", callback=None):
    """Return (QWidget row, QLineEdit ref)."""
    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    edit.setFont(Fonts.mono())
    row = QWidget()
    hl = QHBoxLayout(row)
    hl.setContentsMargins(0, 0, 0, 0)
    hl.setSpacing(4)
    hl.addWidget(edit)
    if callback:
        hl.addWidget(_browse_btn(callback))
    return row, edit


# ─── Tab 1: Videos ────────────────────────────────────────────────────────────

class VideosTab(QScrollArea):
    def __init__(self, session: dict, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # ── Video Path Template ──
        vpath_box = _section("📹  Input Video")
        vpath_form = QFormLayout(vpath_box)
        vpath_form.setSpacing(8)

        self._video_template_row, self.video_template = _path_row(
            "./workspace/VideoInputs/Video{}.mp4",
            self._browse_video_dir
        )
        vpath_form.addRow("Path Template:", self._video_template_row)

        idx_row = QWidget()
        idx_hl = QHBoxLayout(idx_row)
        idx_hl.setContentsMargins(0, 0, 0, 0)
        idx_hl.setSpacing(10)
        self.video_start = QSpinBox(); self.video_start.setRange(1, 9999); self.video_start.setFixedWidth(70)
        self.video_end   = QSpinBox(); self.video_end.setRange(1, 9999);   self.video_end.setFixedWidth(70)
        idx_hl.addWidget(QLabel("Start:")); idx_hl.addWidget(self.video_start)
        idx_hl.addWidget(QLabel("End (count):")); idx_hl.addWidget(self.video_end)
        idx_hl.addStretch()
        vpath_form.addRow("Video Range:", idx_row)
        
        self.max_frames = QSpinBox(); self.max_frames.setRange(0, 99999); self.max_frames.setFixedWidth(80)
        self.max_frames.setToolTip("Max number of frames to extract per video (0 = all)")
        vpath_form.addRow("Max Extract Frames:", self.max_frames)
        
        layout.addWidget(vpath_box)

        # ── Output ──
        out_box = _section("📂  Output & Working Directories")
        out_form = QFormLayout(out_box)
        out_form.setSpacing(8)

        self._output_row, self.output_dir = _path_row("./workspace/outputs", self._browse_output_dir)
        out_form.addRow("Output Folder:", self._output_row)

        self._working_row, self.working_dir = _path_row("./workspace/working_dir", self._browse_working_dir)
        out_form.addRow("Working Dir:", self._working_row)

        self.auto_delete = QCheckBox("Auto-delete working dir after processing")
        self.auto_delete.setFont(Fonts.body())
        out_form.addRow("", self.auto_delete)

        layout.addWidget(out_box)
        layout.addStretch()
        self.setWidget(container)

        self._populate(session)

    # ── browse callbacks ──────────────────────────────────────────────────────

    def _browse_video_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Video Input Folder")
        if d:
            self.video_template.setText(os.path.join(d, "Video{}.mp4").replace("\\", "/"))

    def _browse_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if d:
            self.output_dir.setText(d)

    def _browse_working_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Working Directory")
        if d:
            self.working_dir.setText(d)

    # ── populate / collect ────────────────────────────────────────────────────

    def _populate(self, s: dict):
        # Video Inputs
        vi = s.get("video_inputs", {})
        self.video_template.setText(str(vi.get("template") or s.get("video_path_template", "./workspace/VideoInputs/Video{}.mp4")))
        self.video_start.setValue(int(vi.get("start") or s.get("video_start", 1)))
        self.video_end.setValue(int(vi.get("end") or s.get("video_end", 1)))
        self.max_frames.setValue(int(vi.get("max_frames") or s.get("images_ending_count", 0)))
        
        # Video Outputs
        vo = s.get("video_outputs", {})
        self.output_dir.setText(str(vo.get("final_path") or s.get("final_video_path", "./workspace/outputs")))
        self.working_dir.setText(str(vo.get("working_dir") or s.get("working_dir_name", "./workspace/working_dir")))
        
        delete_val = vo.get("delete_after") if "delete_after" in vo else s.get("delete", False)
        self.auto_delete.setChecked(str(delete_val).lower() == "yes" if not isinstance(delete_val, bool) else delete_val)

    def collect(self) -> dict:
        return {
            "video_path_template": self.video_template.text().strip(),
            "video_start":         self.video_start.value(),
            "video_end":           self.video_end.value(),
            "final_video_path":    self.output_dir.text().strip(),
            "working_dir_name":    self.working_dir.text().strip(),
            "images_ending_count": self.max_frames.value(),
            "delete":              self.auto_delete.isChecked(),
        }


# ─── Tab 2: Settings (Processing + Tracker merged) ────────────────────────────

class SettingsTab(QScrollArea):
    def __init__(self, session: dict, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # ── Processing ───────────────────────────────────────────────────────
        proc_box = _section("⚙️  Processing")
        proc_form = QFormLayout(proc_box)
        proc_form.setSpacing(8)

        self.prefix = QLineEdit(); self.prefix.setMaximumWidth(120)
        proc_form.addRow("Prefix:", self.prefix)

        self.batch_size = QSpinBox(); self.batch_size.setRange(1, 500); self.batch_size.setFixedWidth(80)
        proc_form.addRow("Batch Size:", self.batch_size)

        self.fps = QSpinBox(); self.fps.setRange(1, 240); self.fps.setFixedWidth(80)
        proc_form.addRow("FPS:", self.fps)

        self.run_mode = QComboBox()
        self.run_mode.addItems(["all", "mask_only", "pose_only"])
        self.run_mode.setFixedWidth(130)
        proc_form.addRow("Pipeline Mode:", self.run_mode)

        self.sam_enabled = QCheckBox("Enable SAM masking")
        self.sam_enabled.setFont(Fonts.body())
        proc_form.addRow("SAM:", self.sam_enabled)

        self.auto_prompt = QCheckBox("Auto-prompt encoding (use prev mask to seed next batch)")
        self.auto_prompt.setFont(Fonts.body())
        proc_form.addRow("", self.auto_prompt)

        self.review_from_start = QCheckBox("Review Mode (Ignore existing labels, start at frame 0)")
        self.review_from_start.setFont(Fonts.body())
        self.review_from_start.setToolTip("Start UI at Frame 0 even if the batch is fully annotated.")
        proc_form.addRow("", self.review_from_start)

        self.mask_engine = QComboBox()
        self.mask_engine.addItem("SAM2 (default)", "sam2")
        self.mask_engine.addItem("SAM3 (future)", "sam3")
        self.mask_engine.setFixedWidth(160)
        proc_form.addRow("Mask Engine:", self.mask_engine)

        layout.addWidget(proc_box)

        # ── Pose / Tracker ───────────────────────────────────────────────────
        pose_box = _section("🦴  Pose & Tracker")
        pose_layout = QVBoxLayout(pose_box)
        pose_layout.setSpacing(8)

        self.pose_enabled = QCheckBox("Enable Pose Estimation")
        self.pose_enabled.setFont(Fonts.body())
        self.pose_enabled.toggled.connect(self._on_pose_toggled)
        pose_layout.addWidget(self.pose_enabled)

        self._pose_inner = QWidget()
        inner_layout = QVBoxLayout(self._pose_inner)
        inner_layout.setContentsMargins(12, 0, 0, 0)
        inner_layout.setSpacing(8)

        # tracker selection
        tracker_row = QWidget()
        tracker_hl = QHBoxLayout(tracker_row)
        tracker_hl.setContentsMargins(0, 0, 0, 0)
        self.tracker_group = QButtonGroup(self)
        self.rb_cotracker = QRadioButton("CoTracker (default)")
        self.rb_lk        = QRadioButton("Lucas-Kanade (LK)")
        self.rb_cotracker.setFont(Fonts.body())
        self.rb_lk.setFont(Fonts.body())
        self.tracker_group.addButton(self.rb_cotracker, 0)
        self.tracker_group.addButton(self.rb_lk, 1)
        self.rb_cotracker.setChecked(True)
        self.rb_cotracker.toggled.connect(self._on_tracker_changed)
        tracker_hl.addWidget(self.rb_cotracker)
        tracker_hl.addWidget(self.rb_lk)
        tracker_hl.addStretch()
        inner_layout.addWidget(tracker_row)

        # cotracker config
        self._ct_widget = QWidget()
        ct_form = QFormLayout(self._ct_widget)
        ct_form.setSpacing(6)
        ct_form.setContentsMargins(0, 0, 0, 0)
        self._ct_ckpt_row, self.ct_checkpoint = _path_row(
            "external/co-tracker/checkpoints/scaled_offline.pth",
            self._browse_ct_checkpoint
        )
        ct_form.addRow("Checkpoint:", self._ct_ckpt_row)
        self.ct_window_len = QSpinBox(); self.ct_window_len.setRange(1, 512); self.ct_window_len.setFixedWidth(80)
        ct_form.addRow("Window Length:", self.ct_window_len)
        inner_layout.addWidget(self._ct_widget)

        # class/object ids
        ids_row = QWidget()
        ids_hl = QHBoxLayout(ids_row)
        ids_hl.setContentsMargins(0, 0, 0, 0)
        ids_hl.setSpacing(10)
        self.pose_class_id  = QSpinBox(); self.pose_class_id.setRange(1, 99); self.pose_class_id.setFixedWidth(65)
        self.pose_object_id = QSpinBox(); self.pose_object_id.setRange(1, 99); self.pose_object_id.setFixedWidth(65)
        ids_hl.addWidget(QLabel("Class ID:")); ids_hl.addWidget(self.pose_class_id)
        ids_hl.addWidget(QLabel("Object ID:")); ids_hl.addWidget(self.pose_object_id)
        ids_hl.addStretch()
        inner_layout.addWidget(ids_row)

        self.kp_radius = QSpinBox(); self.kp_radius.setRange(1, 50); self.kp_radius.setFixedWidth(65)
        kp_radius_row = QWidget(); kp_hl = QHBoxLayout(kp_radius_row)
        kp_hl.setContentsMargins(0, 0, 0, 0)
        kp_hl.addWidget(QLabel("Keypoint Radius:")); kp_hl.addWidget(self.kp_radius); kp_hl.addStretch()
        inner_layout.addWidget(kp_radius_row)

        # keypoints list
        kp_header = QWidget(); kp_header_hl = QHBoxLayout(kp_header)
        kp_header_hl.setContentsMargins(0, 0, 0, 0)
        kp_header_hl.addWidget(QLabel("Keypoints:"))
        kp_header_hl.addStretch()
        btn_add = QPushButton("+ Add"); btn_add.setFixedWidth(60)
        btn_del = QPushButton("− Del"); btn_del.setFixedWidth(60)
        btn_add.clicked.connect(self._add_keypoint)
        btn_del.clicked.connect(self._del_keypoint)
        kp_header_hl.addWidget(btn_add); kp_header_hl.addWidget(btn_del)
        inner_layout.addWidget(kp_header)

        self.kp_list = QListWidget()
        self.kp_list.setMaximumHeight(140)
        self.kp_list.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.kp_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.kp_list.setToolTip("Double-click a keypoint to rename it. Drag to reorder.")
        inner_layout.addWidget(self.kp_list)

        pose_layout.addWidget(self._pose_inner)
        layout.addWidget(pose_box)
        layout.addStretch()
        self.setWidget(container)

        self._populate(session)

    # ── Browse callbacks ──────────────────────────────────────────────────────

    def _browse_ct_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CoTracker Checkpoint", "", "Checkpoint (*.pth *.pt)")
        if path:
            self.ct_checkpoint.setText(path)

    # ── Keypoint helpers ──────────────────────────────────────────────────────

    def _add_keypoint(self):
        count = self.kp_list.count()
        item = QListWidgetItem(f"p{count + 1}")
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.kp_list.addItem(item)
        self.kp_list.editItem(item)

    def _del_keypoint(self):
        row = self.kp_list.currentRow()
        if row >= 0:
            self.kp_list.takeItem(row)

    # ── Toggle helpers ────────────────────────────────────────────────────────

    def _on_pose_toggled(self, checked: bool):
        self._pose_inner.setEnabled(checked)

    def _on_tracker_changed(self, checked: bool):
        self._ct_widget.setVisible(self.rb_cotracker.isChecked())

    # ── populate / collect ────────────────────────────────────────────────────

    def _populate(self, s: dict):
        """Populate fields from session dict (handles both flat and nested formats)."""
        vo = s.get("video_outputs", {})
        pl = s.get("pipeline", {})
        
        self.prefix.setText(str(vo.get("prefix") or s.get("prefix", "Img")))
        self.batch_size.setValue(int(pl.get("batch_size") or s.get("batch_size", 30)))
        self.fps.setValue(int(pl.get("fps") or s.get("fps", 30)))

        run_mode = str(pl.get("run_mode") or s.get("run_mode", "all"))
        idx = self.run_mode.findText(run_mode)
        self.run_mode.setCurrentIndex(max(0, idx))

        self.auto_prompt.setChecked(bool(pl.get("auto_prompt") if "auto_prompt" in pl else s.get("auto_prompt_encoding", True)))
        self.review_from_start.setChecked(bool(pl.get("review_from_start") if "review_from_start" in pl else s.get("review_from_start", False)))

        # SAM & Pose — nested under 'models'
        mods = s.get("models", {})
        sam = mods.get("sam") or s.get("sam", {})
        pose = mods.get("pose") or s.get("pose_estimation", {})
        
        # SAM
        if isinstance(sam, dict):
            self.sam_enabled.setChecked(bool(sam.get("enabled", True)))
        else:
            self.sam_enabled.setChecked(bool(sam))

        # Pose
        if isinstance(pose, dict):
            self.pose_enabled.setChecked(bool(pose.get("enabled", False)))
            tracker = str(pose.get("tracker", "cotracker")).lower()
            self.rb_cotracker.setChecked(tracker == "cotracker")
            self.rb_lk.setChecked(tracker == "lk")
            self._ct_widget.setVisible(tracker == "cotracker")
            
            ct = pose.get("cotracker", {})
            if isinstance(ct, dict):
                self.ct_checkpoint.setText(str(ct.get("checkpoint", "")))
                self.ct_window_len.setValue(int(ct.get("window_len", 60)))
            
            # Handle classes array
            classes = pose.get("classes", [])
            if classes and isinstance(classes[0], dict):
                primary_class = classes[0]
                self.pose_class_id.setValue(int(primary_class.get("class_id", 1)))
                self.pose_object_id.setValue(int(primary_class.get("object_id", 1)))
                keypoints = primary_class.get("keypoints", [])
            else:
                self.pose_class_id.setValue(int(pose.get("class_id", 1)))
                self.pose_object_id.setValue(int(pose.get("object_id", 1)))
                keypoints = pose.get("keypoints", [])
            self.kp_radius.setValue(int(pose.get("radius", 5)))
        else:
            self.pose_enabled.setChecked(bool(s.get("pose_enabled", False)))
            keypoints = s.get("keypoints", [])

        self.kp_list.clear()
        for kp in keypoints:
            item = QListWidgetItem(str(kp))
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.kp_list.addItem(item)

        self._pose_inner.setEnabled(self.pose_enabled.isChecked())

    def collect(self) -> dict:
        keypoints = [self.kp_list.item(i).text() for i in range(self.kp_list.count())]
        return {
            "video_path_template": self.video_template.text().strip(),
            "video_start":         self.video_start.value(),
            "video_end":           self.video_end.value(),
            "final_video_path":    self.output_dir.text().strip(),
            "working_dir_name":    self.working_dir.text().strip(),
            "images_ending_count": self.max_frames.value(),
            "delete":              self.auto_delete.isChecked(),
            "prefix":               self.prefix.text().strip(),
            "batch_size":           self.batch_size.value(),
            "fps":                  self.fps.value(),
            "run_mode":             self.run_mode.currentText(),
            "review_from_start":    self.review_from_start.isChecked(),
            "sam_enabled":          self.sam_enabled.isChecked(),
            "auto_prompt_encoding": self.auto_prompt.isChecked(),
            "mask_engine":          self.mask_engine.currentData(),
            "pose_enabled":         self.pose_enabled.isChecked(),
            "tracker":              "cotracker" if self.rb_cotracker.isChecked() else "lk",
            "cotracker_checkpoint": self.ct_checkpoint.text().strip(),
            "cotracker_window_len": self.ct_window_len.value(),
            "pose_class_id":        self.pose_class_id.value(),
            "pose_object_id":       self.pose_object_id.value(),
            "keypoint_radius":      self.kp_radius.value(),
            "keypoints":            keypoints,
        }


# ─── Main Dialog ──────────────────────────────────────────────────────────────

class SetupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AutoSegmentor — Session Setup")
        self.setMinimumSize(620, 560)
        self.setStyleSheet(DARK_STYLESHEET)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowMinimizeButtonHint
        )

        self._session = _load_session()
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QLabel("  🚀  AutoSegmentor — Session Setup")
        header.setFont(QFont("Segoe UI", 14, QFont.Bold))
        header.setFixedHeight(48)
        header.setStyleSheet(
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            f"stop:0 #1a237e, stop:1 #0d47a1); "
            f"color: #e3f2fd; padding-left: 12px;"
        )
        root.addWidget(header)

        self.tabs = QTabWidget()
        self.tabs.setFont(Fonts.body())
        self.tab_videos   = VideosTab(self._session)
        self.tab_settings = SettingsTab(self._session)
        self.tabs.addTab(self.tab_videos,   "📹  Videos")
        self.tabs.addTab(self.tab_settings, "⚙️  Settings")
        root.addWidget(self.tabs, 1)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background:{Colors.BORDER}; max-height:1px;")
        root.addWidget(sep)

        btn_row = QWidget()
        btn_hl = QHBoxLayout(btn_row)
        btn_hl.setContentsMargins(12, 8, 12, 8)
        btn_hl.setSpacing(8)

        self.btn_save_defaults = QPushButton("💾  Save as Defaults")
        self.btn_save_defaults.setToolTip("Write current settings back to default_config.yaml")
        self.btn_save_defaults.clicked.connect(self._on_save_defaults)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_start = QPushButton("▶  Start Session")
        self.btn_start.setObjectName("acceptButton")
        self.btn_start.setDefault(True)
        self.btn_start.clicked.connect(self._on_start)

        btn_hl.addWidget(self.btn_save_defaults)
        btn_hl.addStretch()
        btn_hl.addWidget(self.btn_cancel)
        btn_hl.addWidget(self.btn_start)
        root.addWidget(btn_row)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet(f"color: {Colors.ACCENT_GREEN}; font-size: 8pt; padding-bottom: 4px;")
        root.addWidget(self._status)

    def _collect(self) -> dict:
        data = {}
        data.update(self.tab_videos.collect())
        data.update(self.tab_settings.collect())
        return data

    def _on_start(self):
        data = self._collect()
        _save_session(data)
        self.accept()

    def _on_save_defaults(self):
        data = self._collect()
        _save_session(data)
        _save_defaults(data)
        self._status.setText("✓  Saved to default_config.yaml")

    def get_config(self) -> dict:
        """Return the fully nested config dict for the pipeline."""
        data = self._collect()
        wdir = data["working_dir_name"]
        
        return {
            "external_libs": self._session.get("external_libs", []),
            "video_inputs": {
                "template":   data["video_path_template"],
                "start":      data["video_start"],
                "end":        data["video_end"],
                "max_frames": data["images_ending_count"]
            },
            "video_outputs": {
                "final_path":  data["final_video_path"],
                "working_dir": wdir,
                "prefix":      data["prefix"],
                "delete_after": data["delete"],
                "images_extract_dir":   os.path.join(wdir, "images"),
                "temp_processing_dir":  os.path.join(wdir, "temp"),
                "rendered_dir":         os.path.join(wdir, "render"),
                "overlap_dir":          os.path.join(wdir, "overlap"),
                "verified_img_dir":     os.path.join(wdir, "verified", "images"),
                "verified_mask_dir":    os.path.join(wdir, "verified", "mask"),
            },
            "pipeline": {
                "run_mode":          data["run_mode"],
                "batch_size":        data["batch_size"],
                "fps":               data["fps"],
                "review_from_start": data.get("review_from_start", False),
                "auto_prompt":       data["auto_prompt_encoding"]
            },
            "models": {
                "sam": {
                    "enabled":      data["sam_enabled"],
                    "checkpoint":   self._session.get("models", {}).get("sam", {}).get("checkpoint", ""),
                    "model_config": self._session.get("models", {}).get("sam", {}).get("model_config", "sam2_hiera_l.yaml")
                },
                "pose": {
                    "enabled":  data["pose_enabled"],
                    "tracker":  data["tracker"],
                    "radius":   data["keypoint_radius"],
                    "classes": [{
                        "class_id":  data["pose_class_id"],
                        "object_id": data["pose_object_id"],
                        "keypoints": data["keypoints"],
                    }],
                    "cotracker": {
                        "checkpoint": data["cotracker_checkpoint"],
                        "window_len": data["cotracker_window_len"],
                    },
                }
            }
        }

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dlg = SetupDialog()
    if dlg.exec_() == QDialog.Accepted:
        import pprint
        pprint.pprint(dlg.get_config())
