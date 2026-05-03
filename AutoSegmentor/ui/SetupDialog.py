"""
SetupDialog.py — Launch dialog for AutoSegmentor.

Provides a premium sidebar-based navigation for configuring:
  - Video Inputs & Ranges
  - Model Parameters (SAM2, CoTracker)
  - Output & Storage settings
"""

import json
import os
import sys
import yaml
from .logger_config import logger
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon, QLinearGradient, QPalette, QBrush, QColor
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QStackedWidget, QWidget,
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
    if os.path.exists(_SESSION_STATE):
        try:
            with open(_SESSION_STATE, "r") as f:
                return json.load(f)
        except Exception: pass
    return _load_yaml(_DEFAULT_CONFIG)


def _save_session(data: dict):
    os.makedirs(os.path.dirname(_SESSION_STATE), exist_ok=True)
    
    classes = data.get("pose_classes", [])

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
                "checkpoint":   data.get("sam_checkpoint") or "external/segment_anything_2/checkpoints/sam2_hiera_large.pt",
                "model_config": data.get("sam_model_config") or "sam2_hiera_l.yaml"
            },
            "pose": {
                "enabled":   data.get("pose_enabled", False),
                "tracker":   data.get("tracker", "cotracker"),
                "radius":    data.get("keypoint_radius", 5),
                "classes": classes,
                "cotracker": {
                    "checkpoint": data.get("cotracker_checkpoint", ""),
                    "window_len": data.get("cotracker_window_len", 60)
                }
            }
        },
        "interaction": {
            "active_target_models": data.get("active_target_models", ["sam", "pose"]),
            "auto_shift_enabled": data.get("auto_shift_enabled", True)
        }
    }
    with open(_SESSION_STATE, "w") as f:
        json.dump(nested, f, indent=2)


def _save_defaults(data: dict):
    cfg = _load_yaml(_DEFAULT_CONFIG)
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
    sam["enabled"] = data.get("sam_enabled", True)
    pose = mods.get("pose", {})
    pose["enabled"] = data.get("pose_enabled", False)
    pose["tracker"] = data.get("tracker", "cotracker")
    pose["radius"]  = data.get("keypoint_radius", 5)
    pose["classes"] = data.get("pose_classes", [])
    ct = pose.get("cotracker", {})
    ct["checkpoint"] = data.get("cotracker_checkpoint", "")
    ct["window_len"] = data.get("cotracker_window_len", 60)
    pose["cotracker"] = ct
    cfg["models"] = {"sam": sam, "pose": pose}
    
    cfg["interaction"] = {
        "active_target_models": data.get("active_target_models", ["sam", "pose"]),
        "auto_shift_enabled": data.get("auto_shift_enabled", True)
    }
    _write_config_with_sections(cfg, _DEFAULT_CONFIG)


def _write_config_with_sections(cfg, filepath):
    order = [
        ("External Libraries", ["external_libs"]),
        ("Video Inputs", ["video_inputs"]),
        ("Video Outputs & Storage", ["video_outputs"]),
        ("Pipeline Settings", ["pipeline"]),
        ("Models", ["models"]),
        ("User Interaction & Routing", ["interaction"])
    ]
    lines = []
    processed_keys = set()
    for section_name, keys in order:
        section_dict = {k: cfg[k] for k in keys if k in cfg}
        if not section_dict: continue
        processed_keys.update(keys)
        lines.append(f"# {'=' * 40}\n# {section_name}\n# {'=' * 40}")
        lines.append(yaml.dump(section_dict, default_flow_style=False, allow_unicode=True).strip())
        lines.append("")
    other = {k: v for k, v in cfg.items() if k not in processed_keys}
    if other:
        lines.append(f"# {'=' * 40}\n# Other Configs\n# {'=' * 40}")
        lines.append(yaml.dump(other, default_flow_style=False, allow_unicode=True).strip())
        lines.append("")
    with open(filepath, "w") as f:
        f.write("\n".join(lines))


# ─── Styled sub-widgets ───────────────────────────────────────────────────────

def _section_card(title, icon=None):
    box = QGroupBox(f"{icon + '  ' if icon else ''}{title}")
    box.setFont(Fonts.header())
    box.setStyleSheet(f"QGroupBox {{ background-color: {Colors.BG_MID}; border: 1px solid {Colors.BORDER}; border-radius: 12px; margin-top: 20px; padding-top: 24px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 15px; color: {Colors.ACCENT_BLUE}; }}")
    return box


def _path_row(placeholder="", callback=None):
    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    edit.setFont(Fonts.mono())
    row = QWidget()
    hl = QHBoxLayout(row)
    hl.setContentsMargins(0, 0, 0, 0)
    hl.setSpacing(6)
    hl.addWidget(edit)
    if callback:
        btn = QToolButton()
        btn.setText("…")
        btn.setFixedSize(30, 30)
        btn.clicked.connect(callback)
        hl.addWidget(btn)
    return row, edit


# ─── Pages ────────────────────────────────────────────────────────────────────

class VideoPage(QScrollArea):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # Video Inputs
        in_box = _section_card("Input Video", "📹")
        in_form = QFormLayout(in_box)
        in_form.setSpacing(12)
        in_form.setLabelAlignment(Qt.AlignRight)

        self._vrow, self.video_template = _path_row("./workspace/VideoInputs/Video{}.mp4", self._browse_video_dir)
        in_form.addRow("Path Template:", self._vrow)

        idx_row = QWidget(); idx_hl = QHBoxLayout(idx_row); idx_hl.setContentsMargins(0,0,0,0)
        self.video_start = QSpinBox(); self.video_start.setRange(1,9999); self.video_start.setFixedWidth(80)
        self.video_end   = QSpinBox(); self.video_end.setRange(1,9999);   self.video_end.setFixedWidth(80)
        idx_hl.addWidget(QLabel("Start ID:")); idx_hl.addWidget(self.video_start)
        idx_hl.addSpacing(20)
        idx_hl.addWidget(QLabel("Count:")); idx_hl.addWidget(self.video_end)
        idx_hl.addStretch()
        in_form.addRow("Video Range:", idx_row)

        self.max_frames = QSpinBox(); self.max_frames.setRange(0,99999); self.max_frames.setFixedWidth(80)
        in_form.addRow("Max Frames:", self.max_frames)
        layout.addWidget(in_box)

        # Output & Storage
        out_box = _section_card("Storage & Cleanup", "📂")
        out_form = QFormLayout(out_box)
        out_form.setSpacing(12)

        self._orow, self.output_dir = _path_row("./workspace/outputs", self._browse_output_dir)
        out_form.addRow("Output Root:", self._orow)

        self._wrow, self.working_dir = _path_row("./workspace/working_dir", self._browse_working_dir)
        out_form.addRow("Working Dir:", self._wrow)

        self.auto_delete = QCheckBox("Auto-delete working directory after processing")
        out_form.addRow("", self.auto_delete)
        layout.addWidget(out_box)

        layout.addStretch()
        self.setWidget(container)
        self._populate(session)

    def _browse_video_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if d: self.video_template.setText(os.path.join(d, "Video{}.mp4").replace("\\","/"))
    def _browse_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if d: self.output_dir.setText(d)
    def _browse_working_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Working Directory")
        if d: self.working_dir.setText(d)

    def _populate(self, s: dict):
        vi = s.get("video_inputs", {})
        self.video_template.setText(str(vi.get("template") or s.get("video_path_template", "")))
        self.video_start.setValue(int(vi.get("start") or s.get("video_start", 1)))
        self.video_end.setValue(int(vi.get("end") or s.get("video_end", 1)))
        self.max_frames.setValue(int(vi.get("max_frames") or s.get("images_ending_count", 0)))
        vo = s.get("video_outputs", {})
        self.output_dir.setText(str(vo.get("final_path") or s.get("final_video_path", "")))
        self.working_dir.setText(str(vo.get("working_dir") or s.get("working_dir_name", "")))
        dv = vo.get("delete_after") if "delete_after" in vo else s.get("delete", False)
        self.auto_delete.setChecked(str(dv).lower() == "yes" if not isinstance(dv, bool) else dv)

    def collect(self) -> dict:
        return {
            "video_path_template": self.video_template.text(),
            "video_start": self.video_start.value(),
            "video_end": self.video_end.value(),
            "images_ending_count": self.max_frames.value(),
            "final_video_path": self.output_dir.text(),
            "working_dir_name": self.working_dir.text(),
            "delete": self.auto_delete.isChecked(),
        }


class ModelsPage(QScrollArea):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # SAM
        sam_box = _section_card("Segmentation (SAM2)", "🎨")
        sam_form = QFormLayout(sam_box)
        self.sam_enabled = QCheckBox("Enable Mask Generation")
        sam_form.addRow("Mode:", self.sam_enabled)
        self.auto_prompt = QCheckBox("Auto-prompt Encoding")
        self.auto_prompt.setToolTip("Uses mask from frame N to seed frame N+1")
        sam_form.addRow("", self.auto_prompt)
        layout.addWidget(sam_box)

        # Pose / Tracker
        pose_box = _section_card("Pose & Tracking", "🦴")
        pose_form = QFormLayout(pose_box)
        self.pose_enabled = QCheckBox("Enable Keypoint Tracking")
        pose_form.addRow("Mode:", self.pose_enabled)

        self._pose_inner = QWidget(); pi_layout = QVBoxLayout(self._pose_inner)
        pi_layout.setContentsMargins(0,10,0,0)
        
        trow = QWidget(); thl = QHBoxLayout(trow); thl.setContentsMargins(0,0,0,0)
        self.rb_cotracker = QRadioButton("CoTracker (Learned)"); self.rb_lk = QRadioButton("Lucas-Kanade (Flow)")
        self.rb_cotracker.setChecked(True)
        thl.addWidget(self.rb_cotracker); thl.addWidget(self.rb_lk); thl.addStretch()
        pose_form.addRow("Backend:", trow)

        self._ct_row, self.ct_checkpoint = _path_row("external/co-tracker/checkpoints/scaled_offline.pth", self._browse_ct)
        pose_form.addRow("Checkpoint:", self._ct_row)

        idx_row = QWidget(); idx_hl = QHBoxLayout(idx_row); idx_hl.setContentsMargins(0,0,0,0)
        self.pose_class_id = QSpinBox(); self.pose_class_id.setRange(1,99); self.pose_class_id.setFixedWidth(60)
        self.pose_num_kp   = QSpinBox(); self.pose_num_kp.setRange(0,99);   self.pose_num_kp.setFixedWidth(60)
        idx_hl.addWidget(QLabel("Class:")); idx_hl.addWidget(self.pose_class_id)
        idx_hl.addSpacing(20)
        idx_hl.addWidget(QLabel("Auto Keypoints:")); idx_hl.addWidget(self.pose_num_kp)
        idx_hl.addStretch()
        pose_form.addRow("Target IDs:", idx_row)

        self.kp_list = QListWidget(); self.kp_list.setMaximumHeight(120)
        self.kp_list.setEditTriggers(QAbstractItemView.DoubleClicked)
        
        kp_btns = QWidget(); kph = QHBoxLayout(kp_btns); kph.setContentsMargins(0,0,0,0)
        self.btn_add = QPushButton("+ Add"); self.btn_del = QPushButton("- Del")
        self.btn_add.setFixedWidth(70); self.btn_del.setFixedWidth(70)
        self.btn_add.clicked.connect(self._add_kp); self.btn_del.clicked.connect(self._del_kp)
        kph.addWidget(QLabel("Keypoints:")); kph.addStretch(); kph.addWidget(self.btn_add); kph.addWidget(self.btn_del)
        
        pi_layout.addWidget(kp_btns); pi_layout.addWidget(self.kp_list)
        layout.addWidget(pose_box)
        pose_form.addRow("", self._pose_inner)

        layout.addStretch()
        self.setWidget(container)
        self._populate(session)
        self.pose_enabled.toggled.connect(self._pose_inner.setVisible)
        self._pose_inner.setVisible(self.pose_enabled.isChecked())

    def _browse_ct(self):
        p, _ = QFileDialog.getOpenFileName(self, "CoTracker Checkpoint", "", "pth (*.pth)")
        if p: self.ct_checkpoint.setText(p)
    def _add_kp(self):
        it = QListWidgetItem(f"p{self.kp_list.count()+1}")
        it.setFlags(it.flags() | Qt.ItemIsEditable); self.kp_list.addItem(it)
    def _del_kp(self):
        if self.kp_list.currentRow() >= 0: self.kp_list.takeItem(self.kp_list.currentRow())

    def _populate(self, s: dict):
        m = s.get("models", {})
        sam = m.get("sam", {}); pose = m.get("pose", {})
        self.sam_enabled.setChecked(bool(sam.get("enabled", True)))
        self.auto_prompt.setChecked(bool(s.get("pipeline", {}).get("auto_prompt", True)))
        self.pose_enabled.setChecked(bool(pose.get("enabled", False)))
        self.rb_cotracker.setChecked(str(pose.get("tracker", "cotracker")).lower() == "cotracker")
        self.ct_checkpoint.setText(str(pose.get("cotracker", {}).get("checkpoint", "")))
        
        classes = pose.get("classes", [])
        self.classes_data = {}
        for cls in classes:
            c_id = cls.get("class_id", 1)
            num_kp = cls.get("num_keypoints", 0)
            kp_list = cls.get("keypoints", [])
            # Auto-expand: if num_keypoints given but keypoints list empty, generate p1..pN
            if num_kp > 0 and not kp_list:
                kp_list = [f"p{i+1}" for i in range(num_kp)]
            elif kp_list and num_kp == 0:
                num_kp = len(kp_list)
            self.classes_data[c_id] = {
                "class_id": c_id,
                "num_keypoints": num_kp,
                "keypoints": kp_list
            }
        
        first_id = classes[0].get("class_id", 1) if classes else 1
        self.current_class_id = first_id
        self.pose_class_id.blockSignals(True)
        self.pose_class_id.setValue(first_id)
        self.pose_class_id.blockSignals(False)
        self._load_current_class()
        
        self.pose_class_id.valueChanged.connect(self._on_class_changed)
        self.pose_num_kp.valueChanged.connect(self._on_num_kp_changed)
        
    def _on_class_changed(self, new_id):
        self._save_current_class()
        self.current_class_id = new_id
        self._load_current_class()

    def _on_num_kp_changed(self, n):
        """When user sets Auto Keypoints spinner, auto-fill the list with p1..pN."""
        existing = [self.kp_list.item(i).text() for i in range(self.kp_list.count())]
        if n == 0:
            return
        # Rebuild list: keep existing names where possible, append new ones
        self.kp_list.clear()
        for i in range(n):
            name = existing[i] if i < len(existing) else f"p{i+1}"
            it = QListWidgetItem(name)
            it.setFlags(it.flags() | Qt.ItemIsEditable)
            self.kp_list.addItem(it)

    def _save_current_class(self):
        c_id = self.current_class_id
        if c_id not in self.classes_data:
            self.classes_data[c_id] = {"class_id": c_id}
        kps = [self.kp_list.item(i).text() for i in range(self.kp_list.count())]
        self.classes_data[c_id]["num_keypoints"] = len(kps) if kps else self.pose_num_kp.value()
        self.classes_data[c_id]["keypoints"] = kps

    def _load_current_class(self):
        data = self.classes_data.get(self.current_class_id, {"num_keypoints": 0, "keypoints": []})
        kps = data.get("keypoints", [])
        num_kp = data.get("num_keypoints", 0) or len(kps)
        self.pose_num_kp.blockSignals(True)
        self.pose_num_kp.setValue(num_kp)
        self.pose_num_kp.blockSignals(False)
        self.kp_list.clear()
        for k in kps:
            it = QListWidgetItem(str(k)); it.setFlags(it.flags() | Qt.ItemIsEditable); self.kp_list.addItem(it)

    def collect(self) -> dict:
        self._save_current_class()
        classes_list = []
        for c_id, data in sorted(self.classes_data.items()):
            num_kp = data.get("num_keypoints", 0)
            kps = data.get("keypoints", [])
            # Auto-expand num_keypoints -> keypoints if list still empty
            if num_kp > 0 and not kps:
                kps = [f"p{i+1}" for i in range(num_kp)]
                data["keypoints"] = kps
            elif kps and not num_kp:
                data["num_keypoints"] = len(kps)
            if kps or num_kp > 0:
                classes_list.append(data)
        if not classes_list:
            classes_list.append(self.classes_data.get(self.current_class_id, {"class_id": self.current_class_id, "num_keypoints": 0, "keypoints": []}))
            
        return {
            "sam_enabled": self.sam_enabled.isChecked(),
            "auto_prompt_encoding": self.auto_prompt.isChecked(),
            "pose_enabled": self.pose_enabled.isChecked(),
            "tracker": "cotracker" if self.rb_cotracker.isChecked() else "lk",
            "cotracker_checkpoint": self.ct_checkpoint.text(),
            "pose_classes": classes_list,
            "active_target_models": self.session.get("interaction", {}).get("active_target_models", ["sam", "pose"]),
            "auto_shift_enabled": self.session.get("interaction", {}).get("auto_shift_enabled", True)
        }


class SettingsPage(QScrollArea):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 24, 24, 24)

        card = _section_card("Pipeline Parameters", "⚙️")
        form = QFormLayout(card)
        form.setSpacing(12)

        self.prefix = QLineEdit(); self.prefix.setFixedWidth(120)
        form.addRow("Filename Prefix:", self.prefix)
        self.batch_size = QSpinBox(); self.batch_size.setRange(1, 500); self.batch_size.setFixedWidth(80)
        form.addRow("Batch Size:", self.batch_size)
        self.fps = QSpinBox(); self.fps.setRange(1, 240); self.fps.setFixedWidth(80)
        form.addRow("Output FPS:", self.fps)
        self.run_mode = QComboBox(); self.run_mode.addItems(["all", "mask_only", "pose_only"])
        form.addRow("Pipeline Mode:", self.run_mode)
        self.review_mode = QCheckBox("Review Mode (start at frame 0)")
        form.addRow("", self.review_mode)
        
        layout.addWidget(card)
        layout.addStretch()
        self.setWidget(container)
        self._populate(session)

    def _populate(self, s: dict):
        pl = s.get("pipeline", {}); vo = s.get("video_outputs", {})
        self.prefix.setText(str(vo.get("prefix") or "Img"))
        self.batch_size.setValue(int(pl.get("batch_size", 30)))
        self.fps.setValue(int(pl.get("fps", 30)))
        self.run_mode.setCurrentText(str(pl.get("run_mode", "all")))
        self.review_mode.setChecked(bool(pl.get("review_from_start", False)))

    def collect(self) -> dict:
        return {
            "prefix": self.prefix.text(),
            "batch_size": self.batch_size.value(),
            "fps": self.fps.value(),
            "run_mode": self.run_mode.currentText(),
            "review_from_start": self.review_mode.isChecked(),
        }


# ─── Main Dialog ──────────────────────────────────────────────────────────────

class SetupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AutoSegmentor Launchpad")
        self.setFixedSize(850, 620)
        self.setStyleSheet(DARK_STYLESHEET)
        self._session = _load_session()
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header
        header = QFrame(); header.setFixedHeight(70)
        header.setStyleSheet(f"background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1a237e, stop:1 #311b92); border-bottom: 1px solid {Colors.BORDER};")
        hl = QHBoxLayout(header)
        title = QLabel("  🚀  AUTOSEGMENTOR"); title.setFont(QFont("Segoe UI", 16, QFont.Bold)); title.setStyleSheet("color: white;")
        hl.addWidget(title); hl.addStretch()
        main_layout.addWidget(header)

        # Content area with Sidebar
        content = QWidget(); chl = QHBoxLayout(content); chl.setContentsMargins(0,0,0,0); chl.setSpacing(0)
        
        self.sidebar = QListWidget(); self.sidebar.setObjectName("sidebar"); self.sidebar.setFixedWidth(200)
        self.sidebar.addItems(["📹  Videos", "🧠  Models", "⚙️  Settings"])
        self.sidebar.setCurrentRow(0)
        
        self.stack = QStackedWidget()
        self.page_v = VideoPage(self._session); self.page_m = ModelsPage(self._session); self.page_s = SettingsPage(self._session)
        self.stack.addWidget(self.page_v); self.stack.addWidget(self.page_m); self.stack.addWidget(self.page_s)
        
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)
        
        chl.addWidget(self.sidebar); chl.addWidget(self.stack)
        main_layout.addWidget(content)

        # Footer
        footer = QFrame(); footer.setFixedHeight(60); footer.setStyleSheet(f"background: {Colors.BG_DARKEST}; border-top: 1px solid {Colors.BORDER};")
        fl = QHBoxLayout(footer); fl.setContentsMargins(20,0,20,0)
        self._status = QLabel(""); self._status.setStyleSheet(f"color: {Colors.ACCENT_GREEN};")
        fl.addWidget(self._status); fl.addStretch()
        
        btn_def = QPushButton("Save Defaults"); btn_def.clicked.connect(self._on_save_defaults)
        btn_can = QPushButton("Cancel"); btn_can.clicked.connect(self.reject)
        btn_go  = QPushButton("▶  Launch Pipeline"); btn_go.setObjectName("acceptButton"); btn_go.clicked.connect(self._on_start)
        fl.addWidget(btn_def); fl.addWidget(btn_can); fl.addWidget(btn_go)
        main_layout.addWidget(footer)

    def _collect(self) -> dict:
        d = {}
        d.update(self.page_v.collect()); d.update(self.page_m.collect()); d.update(self.page_s.collect())
        
        sam_state = self._session.get("models", {}).get("sam", {})
        d["sam_checkpoint"] = sam_state.get("checkpoint") or "external/segment_anything_2/checkpoints/sam2_hiera_large.pt"
        d["sam_model_config"] = sam_state.get("model_config") or "sam2_hiera_l.yaml"
        
        return d

    def _on_start(self):
        _save_session(self._collect()); self.accept()

    def _on_save_defaults(self):
        data = self._collect(); _save_session(data); _save_defaults(data)
        self._status.setText("✓ Settings saved to defaults")

    def get_config(self) -> dict:
        data = self._collect()
        base_wdir = data["working_dir_name"]
        start_id = data["video_start"]
        
        # Calculate video-specific working dir for the first video (for Setup purposes)
        # Note: main_app.py will handle the actual loop-level directory switching.
        wdir = os.path.join(base_wdir, f"video{start_id}").replace("\\", "/")
        
        return {
            "external_libs": self._session.get("external_libs", []),
            "video_inputs": {
                "template": data["video_path_template"], "start": data["video_start"], "end": data["video_end"], "max_frames": data["images_ending_count"]
            },
            "video_outputs": {
                "final_path": data["final_video_path"], "working_dir": base_wdir, "prefix": data["prefix"], "delete_after": data["delete"],
                "images_extract_dir": os.path.join(wdir, "images").replace("\\", "/"), 
                "temp_processing_dir": os.path.join(wdir, "temp").replace("\\", "/"),
                "rendered_dir": os.path.join(wdir, "render").replace("\\", "/"), 
                "overlap_dir": os.path.join(wdir, "overlap").replace("\\", "/"),
                "verified_img_dir": os.path.join(wdir, "verified", "images").replace("\\", "/"), 
                "verified_mask_dir": os.path.join(wdir, "verified", "mask").replace("\\", "/"),
            },
            "pipeline": {
                "run_mode": data["run_mode"], "batch_size": data["batch_size"], "fps": data["fps"],
                "review_from_start": data["review_from_start"], "auto_prompt": data["auto_prompt_encoding"]
            },
            "models": {
                "sam": { "enabled": data["sam_enabled"], "checkpoint": data["sam_checkpoint"], "model_config": data["sam_model_config"] },
                "pose": {
                    "enabled": data["pose_enabled"], "tracker": data["tracker"], "radius": 5,
                    "classes": data["pose_classes"],
                    "cotracker": {"checkpoint": data["cotracker_checkpoint"], "window_len": 60}
                }
            },
            "interaction": {
                "active_target_models": data["active_target_models"],
                "auto_shift_enabled": data["auto_shift_enabled"]
            }
        }

if __name__ == "__main__":
    app = QApplication(sys.argv); dlg = SetupDialog()
    if dlg.exec_() == QDialog.Accepted: print(dlg.get_config())
