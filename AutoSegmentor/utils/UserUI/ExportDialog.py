import os
import threading
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QCheckBox, QFileDialog, QSpinBox, 
                             QProgressBar, QFormLayout, QGroupBox, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal, QThread

from .UITheme import DARK_STYLESHEET, Fonts, Colors
from .logger_config import logger

class ExportThread(QThread):
    progress_update = pyqtSignal(int)
    finished = pyqtSignal(bool, str)

    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def run(self):
        try:
            # We wrap the tqdm-based distribute_files_with_threads to emit signals
            # Since the original script uses tqdm, we could monkeypatch or just run it.
            # For simplicity, we'll run it and emit finished.
            self.processor.distribute_files_with_threads()
            self.finished.emit(True, "Export completed successfully!")
        except Exception as e:
            self.finished.emit(False, str(e))

class ExportDialog(QDialog):
    def __init__(self, parent, handler):
        super().__init__(parent)
        self.handler = handler
        self.config = handler.config
        self.setWindowTitle("Export YOLO Dataset")
        self.setMinimumWidth(500)
        self.setStyleSheet(DARK_STYLESHEET)
        
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        header = QLabel("Dataset Export Configuration")
        header.setFont(Fonts.header())
        layout.addWidget(header)

        # Path Selection
        path_group = QGroupBox("Destination")
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        # Default to a subfolder in DatasetManager
        default_base = r"F:\RunningProjects\AutoSegmentor\DatasetManager"
        default_path = os.path.join(default_base, f"Video{self.config.video_number}_Export")
        self.path_edit.setText(default_path)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_path)
        
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(browse_btn)
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)

        # Export Modes
        mode_group = QGroupBox("Export Types")
        mode_layout = QVBoxLayout()
        self.cb_detect = QCheckBox("YOLO Detection (Bounding Boxes)")
        self.cb_segment = QCheckBox("YOLO Segmentation (Polygons)")
        self.cb_pose = QCheckBox("YOLO Pose (Keypoints)")
        
        self.cb_detect.setChecked(True)
        self.cb_segment.setChecked(True)
        self.cb_pose.setChecked(True)
        
        mode_layout.addWidget(self.cb_detect)
        mode_layout.addWidget(self.cb_segment)
        mode_layout.addWidget(self.cb_pose)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Parameters
        param_group = QGroupBox("Parameters")
        param_layout = QFormLayout()
        
        self.aug_spin = QSpinBox()
        self.aug_spin.setRange(1, 50)
        self.aug_spin.setValue(10)
        param_layout.addRow("Augmentation Times:", self.aug_spin)
        
        self.val_split = QSpinBox()
        self.val_split.setRange(0, 100)
        self.val_split.setValue(10)
        self.val_split.setSuffix("%")
        param_layout.addRow("Validation Split:", self.val_split)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Export")
        self.start_btn.setObjectName("acceptButton")
        self.start_btn.clicked.connect(self.start_export)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(self.start_btn)
        layout.addLayout(btn_layout)

    def browse_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Export Directory", self.path_edit.text())
        if path:
            self.path_edit.setText(path)

    def start_export(self):
        export_path = self.path_edit.text()
        if not export_path:
            return

        # Prepare types
        export_types = []
        if self.cb_detect.isChecked(): export_types.append('box')
        if self.cb_segment.isChecked(): export_types.append('mask')
        if self.cb_pose.isChecked(): export_types.append('pose')

        if not export_types:
            logger.warning("No export types selected.")
            return

        # Build dynamic config for DatasetCreatere
        # Note: We need to import YoloProcessor here to avoid circular imports if any
        try:
            import sys
            # Append YolovDatasetManager to path
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            manager_path = os.path.join(repo_root, 'DatasetManager', 'YolovDatasetManager')
            if manager_path not in sys.path:
                sys.path.insert(0, manager_path)
            
            from DatasetCreatere import YoloProcessor
        except ImportError as e:
            logger.error(f"Could not import YoloProcessor: {e}")
            return

        # Class Mapping: Get from UI classes
        # AnnotationColors/UITheme usually has a 1-based mapping.
        # We'll build a simplified map for YOLO (0-based)
        class_names = []
        class_to_id = {}
        color_to_label = {}
        
        # Get from MainWindow's theme or similar
        # For now, let's use the standard 10 classes
        from .UITheme import ANNOTATION_COLORS_BGR
        for i in range(1, 11):
            name = f"class_{i}"
            class_names.append(name)
            class_to_id[name] = i - 1 # YOLO index
            # Map BGR to label
            color_to_label[ANNOTATION_COLORS_BGR[i]] = i - 1

        v_pct = self.val_split.value() / 100.0
        
        export_config = {
            "dataset_path": self.config.working_dir,
            "SOURCE_mask_folder_name": "render",
            "SOURCE_original_folder_name": "images",
            "SOURCE_mask_type_ext": ".png",
            "SOURCE_img_type_ext": ".jpeg",
            "augment_times": self.aug_spin.value(),
            "test_split": 0.0,
            "val_split": v_pct,
            "train_split": 1.0 - v_pct,
            "Keep_val_dataset_original": True,
            "num_threads": 4,
            "class_to_id": class_to_id,
            "color_to_label": color_to_label,
            "dataset_saving_working_dir": os.path.dirname(export_path),
            "folder_name": os.path.basename(export_path),
            "class_names": class_names,
            "DESTINATION_img_type_ext": ".jpg",
            "DESTINATION_label_type_ext": ".txt",
            "FromDataType": "",
            "ToDataTypeFormate": "",
            "annotation_manager": self.handler.annotation_manager,
            "export_types": export_types
        }

        self.start_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate while starting

        try:
            processor = YoloProcessor(config=export_config)
            self.thread = ExportThread(processor)
            self.thread.finished.connect(self.on_export_finished)
            self.thread.start()
        except Exception as e:
            self.on_export_finished(False, str(e))

    def on_export_finished(self, success, message):
        self.start_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        if success:
            logger.info(message)
            self.accept()
        else:
            logger.error(f"Export failed: {message}")
