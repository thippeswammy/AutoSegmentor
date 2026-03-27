"""
Unit Tests: FileManagement Utilities
Tests for FileManager, FrameHandler, and AnnotationManager I/O logic.
"""
import os
import sys
import json
import shutil
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.FileManagement.FileManager import ensure_directory


class TestEnsureDirectory:
    """Test directory creation utility."""

    def test_creates_new_directory(self, tmp_path):
        target = tmp_path / "new_dir"
        assert not target.exists()
        ensure_directory(str(target))
        assert target.exists()

    def test_does_not_fail_if_exists(self, tmp_path):
        target = tmp_path / "existing"
        target.mkdir()
        ensure_directory(str(target))  # Should not raise
        assert target.exists()

    def test_creates_nested_directories(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        ensure_directory(str(target))
        assert target.exists()


class TestAnnotationManagerIO:
    """Test annotation save/load roundtrip."""

    def test_save_and_load_points(self, tmp_path):
        """Test that saved annotation data can be reloaded correctly."""
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        from utils.Models.AppConfig import AppConfig
        from utils.UserUI.AnnotationManager import AnnotationManager

        # Minimal config pointing at tmp directories
        config = AppConfig(
            video_number=1,
            prefix="Img",
            images_extract_dir=str(tmp_path / "images"),
            rendered_frames_dir=str(tmp_path / "render"),
            temp_processing_dir=str(tmp_path / "temp")
        )

        # Create a dummy image file so AnnotationManager finds frame paths
        (tmp_path / "images").mkdir(exist_ok=True)
        dummy_frame = tmp_path / "images" / "Img1_00000.jpg"
        dummy_frame.write_bytes(b"\xff\xd8\xff")  # Minimal JPEG header

        frame_paths = [str(dummy_frame)]
        manager = AnnotationManager(config, frame_paths)

        # Save a prompt
        manager.save_points_and_labels(
            frame_idx=0,
            points=[[100, 200], [300, 400]],
            labels=[1001, 1001],
            pose_keypoints=[{"name": "p1", "point_id": 0, "x": 100, "y": 200, "visible": 2}]
        )

        # Reload and check data
        manager2 = AnnotationManager(config, frame_paths)
        manager2.load_points_and_labels()

        assert len(manager2.points_collection) > 0

    def test_get_prompt_for_frame(self, tmp_path):
        """Test retrieval of saved prompts by frame index."""
        from utils.Models.AppConfig import AppConfig
        from utils.UserUI.AnnotationManager import AnnotationManager

        config = AppConfig(
            video_number=2,
            prefix="Img",
            images_extract_dir=str(tmp_path / "images"),
            rendered_frames_dir=str(tmp_path / "render"),
            temp_processing_dir=str(tmp_path / "temp")
        )

        (tmp_path / "images").mkdir(exist_ok=True)
        dummy_frame = tmp_path / "images" / "Img2_00000.jpg"
        dummy_frame.write_bytes(b"\xff\xd8\xff")

        frame_paths = [str(dummy_frame)]
        manager = AnnotationManager(config, frame_paths)

        manager.save_points_and_labels(
            frame_idx=0,
            points=[[50, 75]],
            labels=[1001],
            pose_keypoints=None
        )

        prompt = manager.get_prompt_for_frame(0)
        assert prompt is not None
        assert prompt["points"] == [[50, 75]]
        assert prompt["labels"] == [1001]
