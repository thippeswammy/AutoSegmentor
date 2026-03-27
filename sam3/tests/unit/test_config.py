"""
Unit Tests: AppConfig
Tests loading, defaults, and path construction for AppConfig.
"""
import os
import sys
import pytest
import tempfile

# Ensure sam3 is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.Models.AppConfig import AppConfig


class TestAppConfigDefaults:
    """Test default values in AppConfig."""

    def test_defaults_are_set(self, tmp_path):
        """AppConfig should fill in sensible defaults when optional args are None."""
        config = AppConfig(video_number=1, images_extract_dir=str(tmp_path / "images"),
                           rendered_frames_dir=str(tmp_path / "render"),
                           temp_processing_dir=str(tmp_path / "temp"))
        assert config.batch_size == 120
        assert config.prefix == "file"
        assert config.sam_enabled is True
        assert config.auto_prompt_encoding is True
        assert config.memory_bank_size == 5
        assert config.prompt_memory_size == 5

    def test_video_number_stored(self, tmp_path):
        config = AppConfig(video_number=42, images_extract_dir=str(tmp_path / "images"),
                           rendered_frames_dir=str(tmp_path / "render"),
                           temp_processing_dir=str(tmp_path / "temp"))
        assert config.video_number == 42

    def test_sam_disabled(self, tmp_path):
        config = AppConfig(video_number=1, sam_enabled=False,
                           images_extract_dir=str(tmp_path / "images"),
                           rendered_frames_dir=str(tmp_path / "render"),
                           temp_processing_dir=str(tmp_path / "temp"))
        assert config.sam_enabled is False

    def test_pose_config_stored(self, tmp_path):
        pose = {"enabled": True, "tracker": "cotracker", "keypoints": ["p1", "p2"]}
        config = AppConfig(video_number=1, pose_config=pose,
                           images_extract_dir=str(tmp_path / "images"),
                           rendered_frames_dir=str(tmp_path / "render"),
                           temp_processing_dir=str(tmp_path / "temp"))
        assert config.pose_config == pose
        assert config.pose_config["tracker"] == "cotracker"

    def test_directories_created(self, tmp_path):
        """AppConfig should auto-create the output directories."""
        frames_dir = tmp_path / "images"
        render_dir = tmp_path / "render"
        temp_dir = tmp_path / "temp"

        # None of these exist yet
        assert not frames_dir.exists()
        config = AppConfig(video_number=1, images_extract_dir=str(frames_dir),
                           rendered_frames_dir=str(render_dir),
                           temp_processing_dir=str(temp_dir))
        assert frames_dir.exists()
        assert render_dir.exists()
        assert temp_dir.exists()

    def test_label_colors_default(self, tmp_path):
        config = AppConfig(video_number=1, images_extract_dir=str(tmp_path / "images"),
                           rendered_frames_dir=str(tmp_path / "render"),
                           temp_processing_dir=str(tmp_path / "temp"))
        assert 1 in config.label_colors
        assert isinstance(config.label_colors[1], tuple)
        assert len(config.label_colors[1]) == 3  # BGR

    def test_custom_label_colors(self, tmp_path):
        custom_colors = {1: (1, 2, 3)}
        config = AppConfig(video_number=1, label_colors=custom_colors,
                           images_extract_dir=str(tmp_path / "images"),
                           rendered_frames_dir=str(tmp_path / "render"),
                           temp_processing_dir=str(tmp_path / "temp"))
        assert config.label_colors == custom_colors
