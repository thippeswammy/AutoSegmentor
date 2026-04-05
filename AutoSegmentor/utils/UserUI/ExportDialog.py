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
        path_form = QFormLayout(path_group)
        
        # Base Directory (The parent folder)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        self.default_base = os.path.join(repo_root, "DatasetManager")
        
        base_layout = QHBoxLayout()
        self.base_dir_edit = QLineEdit(self.default_base)
        self.base_dir_edit.setReadOnly(True)
        self.base_dir_edit.setStyleSheet("background-color: #2b2b2b; color: #888;")
        
        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self.browse_base_path)
        
        base_layout.addWidget(self.base_dir_edit)
        base_layout.addWidget(browse_btn)
        path_form.addRow("Base Directory:", base_layout)
        
        # Folder Name (The target dataset name)
        self.folder_name_edit = QLineEdit(f"Video{self.config.video_number}_Export")
        self.folder_name_edit.setPlaceholderText("e.g. Video6_Export")
        path_form.addRow("Folder Name:", self.folder_name_edit)
        
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
        param_group = QGroupBox("Splits & Count")
        param_layout = QFormLayout()
        
        self.aug_spin = QSpinBox()
        self.aug_spin.setRange(1, 50)
        self.aug_spin.setValue(1) # Default to 1 as requested
        param_layout.addRow("Augmentation Times:", self.aug_spin)
        
        self.train_split = QSpinBox()
        self.train_split.setRange(0, 100)
        self.train_split.setValue(80)
        self.train_split.setSuffix("%")
        
        self.val_split = QSpinBox()
        self.val_split.setRange(0, 100)
        self.val_split.setValue(10)
        self.val_split.setSuffix("%")
        
        self.test_split = QSpinBox()
        self.test_split.setRange(0, 100)
        self.test_split.setValue(10)
        self.test_split.setSuffix("%")
        
        self.train_split.valueChanged.connect(lambda: self._balance_splits(self.train_split))
        self.val_split.valueChanged.connect(lambda: self._balance_splits(self.val_split))
        self.test_split.valueChanged.connect(lambda: self._balance_splits(self.test_split))
        
        param_layout.addRow("Train Split:", self.train_split)
        param_layout.addRow("Validation Split:", self.val_split)
        param_layout.addRow("Test Split:", self.test_split)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # Augmentations
        aug_group = QGroupBox("Augmentations")
        aug_layout = QVBoxLayout()
        
        self.cb_keep_val_original = QCheckBox("Keep Validation/Test Datasets Original")
        self.cb_keep_val_original.setChecked(True)
        self.cb_keep_val_original.setToolTip("If checked, Validation and Test sets will not receive augmentations.")
        
        self.cb_aug_color = QCheckBox("Color Jitter (Brightness/Contrast)")
        self.cb_aug_color.setChecked(True)
        
        self.cb_aug_gauss_blur = QCheckBox("Gaussian Blur")
        self.cb_aug_gauss_blur.setChecked(True)
        
        self.cb_aug_avg_blur = QCheckBox("Average Blur")
        self.cb_aug_avg_blur.setChecked(True)
        
        self.cb_aug_gauss_noise = QCheckBox("Gaussian Noise")
        self.cb_aug_gauss_noise.setChecked(True)
        
        self.cb_aug_sp_noise = QCheckBox("Salt & Pepper Noise")
        self.cb_aug_sp_noise.setChecked(True)
        
        aug_layout.addWidget(self.cb_keep_val_original)
        aug_layout.addSpacing(10)
        aug_layout.addWidget(self.cb_aug_color)
        aug_layout.addWidget(self.cb_aug_gauss_blur)
        aug_layout.addWidget(self.cb_aug_avg_blur)
        aug_layout.addWidget(self.cb_aug_gauss_noise)
        aug_layout.addWidget(self.cb_aug_sp_noise)
        
        aug_group.setLayout(aug_layout)
        layout.addWidget(aug_group)

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

    def browse_base_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Base Directory", self.base_dir_edit.text())
        if path:
            self.base_dir_edit.setText(path)

    def _balance_splits(self, sender):
        """Keep the sum of Train, Val, and Test to 100%."""
        self.train_split.blockSignals(True)
        self.val_split.blockSignals(True)
        self.test_split.blockSignals(True)
        
        t = self.train_split.value()
        v = self.val_split.value()
        test = self.test_split.value()
        
        diff = 100 - (t + v + test)
        
        if diff != 0:
            if sender == self.train_split:
                if self.val_split.value() + diff >= 0:
                    self.val_split.setValue(self.val_split.value() + diff)
                else: 
                    self.test_split.setValue(max(0, self.test_split.value() + diff))
            elif sender == self.val_split:
                if self.train_split.value() + diff >= 0:
                    self.train_split.setValue(self.train_split.value() + diff)
                else: 
                    self.test_split.setValue(max(0, self.test_split.value() + diff))
            else:
                if self.val_split.value() + diff >= 0:
                    self.val_split.setValue(self.val_split.value() + diff)
                else:
                    self.train_split.setValue(max(0, self.train_split.value() + diff))
                    
        # Fallback correction to ensure exactly 100
        t, v, test = self.train_split.value(), self.val_split.value(), self.test_split.value()
        if t + v + test != 100:
            if sender != self.train_split: self.train_split.setValue(100 - v - test)
            else: self.val_split.setValue(100 - t - test)
            
        self.train_split.blockSignals(False)
        self.val_split.blockSignals(False)
        self.test_split.blockSignals(False)

    def start_export(self):
        base_path = self.base_dir_edit.text()
        folder_name = self.folder_name_edit.text().strip()
        if not folder_name:
            return
            
        logger.debug(f"[Export] start_export: base={base_path}  folder={folder_name}")
        export_path = os.path.join(base_path, folder_name)

        # Prepare types
        export_types = []
        if self.cb_detect.isChecked(): export_types.append('box')
        if self.cb_segment.isChecked(): export_types.append('mask')
        if self.cb_pose.isChecked(): export_types.append('pose')

        logger.debug(f"[Export] Selected types: {export_types}")
        if not export_types:
            logger.warning("No export types selected.")
            return

        # Build dynamic config for DatasetCreator
        # Note: We need to import YoloProcessor here to avoid circular imports if any
        try:
            import sys
            # Append YolovDatasetManager to path
            # Up 3 levels: UserUI -> utils -> AutoSegmentor -> Root
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            manager_path = os.path.join(repo_root, 'DatasetManager', 'YolovDatasetManager')
            if manager_path not in sys.path:
                sys.path.insert(0, manager_path)
            
            from DatasetCreator import YoloProcessor
        except ImportError as e:
            logger.error(f"[Export] Could not import YoloProcessor: {e}")
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

        # Collect enabled augmentations
        enabled_augs = []
        if self.cb_aug_color.isChecked(): enabled_augs.append('color')
        if self.cb_aug_gauss_blur.isChecked(): enabled_augs.append('gauss_blur')
        if self.cb_aug_avg_blur.isChecked(): enabled_augs.append('avg_blur')
        if self.cb_aug_gauss_noise.isChecked(): enabled_augs.append('gauss_noise')
        if self.cb_aug_sp_noise.isChecked(): enabled_augs.append('sp_noise')

        export_config = {
            "dataset_path": self.config.working_dir,
            "SOURCE_mask_folder_name": "render",
            "SOURCE_original_folder_name": "images",
            "SOURCE_mask_type_ext": ".png",
            "SOURCE_img_type_ext": ".jpeg",
            "augment_times": self.aug_spin.value(),
            "test_split": self.test_split.value() / 100.0,
            "val_split": self.val_split.value() / 100.0,
            "train_split": self.train_split.value() / 100.0,
            "Keep_val_dataset_original": self.cb_keep_val_original.isChecked(),
            "enabled_augmentations": enabled_augs,
            "num_threads": 4,
            "class_to_id": class_to_id,
            "color_to_label": color_to_label,
            "dataset_saving_working_dir": base_path,
            "folder_name": folder_name,
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
            logger.info(f"[Export] Success: {message}")
            self.accept()
        else:
            logger.error(f"[Export] Failed: {message}")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Export Error", f"Failed to export dataset:\n{message}")
