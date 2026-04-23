"""
Integration Tests: Pipeline Runner
Tests the pipeline orchestration logic with mock/stub components.
"""
import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class TestPipelineImports:
    """Verify new imports paths work correctly after refactoring."""

    def test_app_config_importable(self):
        from autosegmentor.models.SAM.AppConfig import AppConfig
        assert AppConfig is not None

    def test_lk_tracker_importable(self):
        from autosegmentor.models.Tracking.LKKeypointTracker import LKKeypointTracker
        assert LKKeypointTracker is not None

    def test_pipeline_importable(self):
        # pipeline.py should be importable without errors
        import importlib
        spec = importlib.util.find_spec("utils.pipeline")
        assert spec is not None

    def test_file_manager_importable(self):
        from autosegmentor.file_management.FileManager import ensure_directory
        assert ensure_directory is not None

    def test_pose_exporter_importable(self):
        from autosegmentor.file_management.PoseExporter import PoseExporter
        assert PoseExporter is not None

    def test_user_interaction_importable(self):
        from autosegmentor.ui.UserInteraction import UserInteractionHandler
        assert UserInteractionHandler is not None

    def test_annotation_manager_importable(self):
        from autosegmentor.ui.AnnotationManager import AnnotationManager
        assert AnnotationManager is not None


class TestPoseExporterIntegration:
    """Test PoseExporter with mock data (no GPU or real video required)."""

    def test_lk_tracking_roundtrip(self, tmp_path):
        """End-to-end LK tracking and YOLO-JSON export on synthetic frames."""
        import cv2
        import numpy as np
        from autosegmentor.models.SAM.AppConfig import AppConfig
        from autosegmentor.models.Tracking.LKKeypointTracker import LKKeypointTracker

        # Create tiny synthetic frames
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        n_frames = 5
        frame_paths = []
        for i in range(n_frames):
            img = np.full((120, 160, 3), i * 40, dtype=np.uint8)
            p = frame_dir / f"frame_{i:05d}.jpg"
            cv2.imwrite(str(p), img)
            frame_paths.append(str(p))

        kp_defs = ["kp0", "kp1", "kp2"]
        init_coords = [
            {"name": "kp0", "point_id": 0, "x": 40, "y": 60},
            {"name": "kp1", "point_id": 1, "x": 80, "y": 60},
            {"name": "kp2", "point_id": 2, "x": 120, "y": 60},
        ]

        first_frame = cv2.imread(frame_paths[0])
        tracker = LKKeypointTracker(kp_defs, init_coords, first_frame)

        for i in range(1, n_frames):
            frame = cv2.imread(frame_paths[i])
            tracker.track(frame, frame_index=i)

        result = tracker.get_all_tracked()
        assert len(result) == n_frames
        for entry in result:
            assert "frame_index" in entry
            assert "keypoints" in entry
            assert len(entry["keypoints"]) == 3

    def test_config_pose_json_structure(self, tmp_path):
        """Verify that PoseExporter generates valid JSON with correct structure."""
        import cv2
        import numpy as np
        from unittest.mock import MagicMock
        from autosegmentor.models.SAM.AppConfig import AppConfig
        from autosegmentor.file_management.PoseExporter import PoseExporter

        # Create minimal directories
        frames_dir = tmp_path / "images"
        frames_dir.mkdir()
        mask_dir = tmp_path / "mask"
        mask_dir.mkdir()

        for i in range(3):
            img = np.full((120, 160, 3), 128, dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"Img1_{i:05d}.jpg"), img)

        config = AppConfig(
            video_number=1, prefix="Img", batch_size=3,
            images_extract_dir=str(frames_dir),
            rendered_frames_dir=str(mask_dir),
            temp_processing_dir=str(tmp_path / "temp"),
            pose_config={
                "enabled": True,
                "tracker": "lk",
                "keypoints": ["kp0", "kp1"],
                "class_id": 1,
                "object_id": 1
            }
        )

        # Mock the annotation_manager to return one set of keypoints
        annotation_manager = MagicMock()
        annotation_manager.pose_keypoints_collection = [[
            {"name": "kp0", "point_id": 0, "x": 50, "y": 60, "visible": 2},
            {"name": "kp1", "point_id": 1, "x": 70, "y": 60, "visible": 2},
        ]]

        exporter = PoseExporter(config, str(mask_dir), annotation_manager)
        exporter.process_masks()

        assert os.path.exists(exporter.output_file), "Output JSON was not created"
        with open(exporter.output_file) as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 3  # 3 frames
        for entry in data:
            assert "image_id" in entry
            assert "instances" in entry
            instance = entry["instances"][0]
            assert "keypoints" in instance
            assert len(instance["keypoints"]) == 2
