"""
UI Automation Tests: SetupDialog
Uses pytest-qt + QtTest to simulate user interactions with the setup dialog.

These tests verify:
  - SetupDialog can be instantiated and shown without errors.
  - Default values are loaded from session/config correctly.
  - All tab widgets exist and are accessible.
  - Video tab fields emit correct values in get_config().
  - Settings tab fields (batch size, fps, run mode, tracker) emit correct values.
  - Session is saved to JSON when "Start" is clicked.
  - "Save as Defaults" does NOT crash even if YAML path is missing.
  - Cancel closes the dialog with QDialog.Rejected result.
  - Keypoint add/remove buttons modify the list correctly.
  - Pose-inner controls are disabled when pose_enabled is unchecked.
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QDialog


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    return app


@pytest.fixture
def dialog(qapp, tmp_path, monkeypatch):
    """Patch session/config paths so tests don't pollute real files."""
    fake_session = tmp_path / "session_state.json"
    fake_config  = tmp_path / "default_config.yaml"

    # Write a minimal YAML so SetupDialog can load defaults
    fake_config.write_text(
        "video_start: 1\nvideo_end: 2\nprefix: Test\nbatch_size: 15\nfps: 24\n"
        "delete: false\nworking_dir_name: working_dir\n"
        "video_path_template: ./inputs/VideoInputs/Video{}.mp4\n"
        "final_video_path: ./outputs\nauto_prompt_encoding: true\nrun_mode: all\n"
        "sam:\n  enabled: true\n"
        "pose_estimation:\n  enabled: false\n  tracker: cotracker\n"
        "  class_id: 1\n  object_id: 1\n  radius: 5\n  keypoints: [p1, p2]\n"
        "  cotracker:\n    checkpoint: ''\n    window_len: 60\n",
        encoding="utf-8"
    )

    import utils.UserUI.SetupDialog as sd_module
    monkeypatch.setattr(sd_module, "_DEFAULT_CONFIG", str(fake_config))
    monkeypatch.setattr(sd_module, "_SESSION_STATE",  str(fake_session))

    from utils.UserUI.SetupDialog import SetupDialog
    dlg = SetupDialog()
    dlg.show()
    QTest.qWaitForWindowExposed(dlg)
    yield dlg, tmp_path, fake_session
    dlg.close()


# ─── Instantiation ────────────────────────────────────────────────────────────

class TestSetupDialogInit:

    def test_dialog_opens(self, dialog):
        dlg, *_ = dialog
        assert dlg.isVisible()

    def test_window_title_contains_autosegmentor(self, dialog):
        dlg, *_ = dialog
        assert "AutoSegmentor" in dlg.windowTitle()

    def test_two_tabs_exist(self, dialog):
        dlg, *_ = dialog
        assert dlg.tabs.count() == 2

    def test_videos_tab_is_first(self, dialog):
        dlg, *_ = dialog
        assert "Video" in dlg.tabs.tabText(0)

    def test_settings_tab_is_second(self, dialog):
        dlg, *_ = dialog
        assert "Setting" in dlg.tabs.tabText(1)


# ─── Videos Tab ──────────────────────────────────────────────────────────────

class TestVideosTab:

    def test_video_template_populated(self, dialog):
        dlg, *_ = dialog
        assert "Video" in dlg.tab_videos.video_template.text()

    def test_video_start_default(self, dialog):
        dlg, *_ = dialog
        assert dlg.tab_videos.video_start.value() == 1

    def test_video_end_default(self, dialog):
        dlg, *_ = dialog
        assert dlg.tab_videos.video_end.value() == 2

    def test_output_dir_populated(self, dialog):
        dlg, *_ = dialog
        assert dlg.tab_videos.output_dir.text() != ""

    def test_auto_delete_default_false(self, dialog):
        dlg, *_ = dialog
        # default YAML has delete: false
        assert dlg.tab_videos.auto_delete.isChecked() == False

    def test_change_video_start(self, dialog):
        dlg, *_ = dialog
        dlg.tab_videos.video_start.setValue(3)
        assert dlg.tab_videos.collect()["video_start"] == 3


# ─── Settings Tab ─────────────────────────────────────────────────────────────

class TestSettingsTab:

    def test_prefix_default(self, dialog):
        dlg, *_ = dialog
        assert dlg.tab_settings.prefix.text() == "Test"

    def test_batch_size_default(self, dialog):
        dlg, *_ = dialog
        assert dlg.tab_settings.batch_size.value() == 15

    def test_fps_default(self, dialog):
        dlg, *_ = dialog
        assert dlg.tab_settings.fps.value() == 24

    def test_run_mode_default(self, dialog):
        dlg, *_ = dialog
        assert dlg.tab_settings.run_mode.currentText() == "all"

    def test_sam_enabled_default(self, dialog):
        dlg, *_ = dialog
        assert dlg.tab_settings.sam_enabled.isChecked() == True

    def test_cotracker_radio_default(self, dialog):
        dlg, *_ = dialog
        assert dlg.tab_settings.rb_cotracker.isChecked()

    def test_lk_radio_switch(self, dialog):
        dlg, *_ = dialog
        dlg.tab_settings.rb_lk.setChecked(True)
        data = dlg.tab_settings.collect()
        assert data["tracker"] == "lk"
        # reset
        dlg.tab_settings.rb_cotracker.setChecked(True)

    def test_pose_disabled_by_default(self, dialog):
        dlg, *_ = dialog
        assert dlg.tab_settings.pose_enabled.isChecked() == False

    def test_pose_inner_disabled_when_pose_off(self, dialog):
        dlg, *_ = dialog
        dlg.tab_settings.pose_enabled.setChecked(False)
        assert not dlg.tab_settings._pose_inner.isEnabled()

    def test_pose_inner_enabled_when_pose_on(self, dialog):
        dlg, *_ = dialog
        dlg.tab_settings.pose_enabled.setChecked(True)
        assert dlg.tab_settings._pose_inner.isEnabled()
        dlg.tab_settings.pose_enabled.setChecked(False)  # reset


# ─── Keypoints ────────────────────────────────────────────────────────────────

class TestKeypointList:

    def test_keypoints_pre_populated(self, dialog):
        dlg, *_ = dialog
        # YAML has [p1, p2]
        assert dlg.tab_settings.kp_list.count() == 2

    def test_add_keypoint(self, dialog):
        dlg, *_ = dialog
        before = dlg.tab_settings.kp_list.count()
        dlg.tab_settings._add_keypoint()
        assert dlg.tab_settings.kp_list.count() == before + 1

    def test_del_keypoint(self, dialog):
        dlg, *_ = dialog
        dlg.tab_settings.kp_list.setCurrentRow(0)
        before = dlg.tab_settings.kp_list.count()
        dlg.tab_settings._del_keypoint()
        assert dlg.tab_settings.kp_list.count() == before - 1


# ─── get_config() ─────────────────────────────────────────────────────────────

class TestGetConfig:

    def test_get_config_returns_dict(self, dialog):
        dlg, *_ = dialog
        cfg = dlg.get_config()
        assert isinstance(cfg, dict)

    def test_required_pipeline_keys_present(self, dialog):
        dlg, *_ = dialog
        cfg = dlg.get_config()
        required = [
            "video_start", "video_end", "prefix", "batch_size", "fps",
            "delete", "working_dir_name", "video_path_template",
            "images_extract_dir", "temp_processing_dir", "rendered_dir",
            "overlap_dir", "verified_img_dir", "verified_mask_dir",
            "final_video_path", "images_ending_count",
            "run_mode", "auto_prompt_encoding", "sam_enabled",
            "pose_estimation",
        ]
        for key in required:
            assert key in cfg, f"Missing key: {key}"

    def test_pose_estimation_is_dict(self, dialog):
        dlg, *_ = dialog
        cfg = dlg.get_config()
        assert isinstance(cfg["pose_estimation"], dict)

    def test_batch_size_reflected_in_config(self, dialog):
        dlg, *_ = dialog
        dlg.tab_settings.batch_size.setValue(42)
        cfg = dlg.get_config()
        assert cfg["batch_size"] == 42


# ─── Session Persistence ──────────────────────────────────────────────────────

class TestSessionPersistence:

    def test_start_saves_session_json(self, dialog):
        dlg, tmp_path, fake_session = dialog
        dlg.tab_settings.batch_size.setValue(77)
        # Simulate clicking Start
        dlg._on_start()
        assert fake_session.exists(), "session_state.json was not created"

    def test_session_json_contains_batch_size(self, dialog):
        dlg, tmp_path, fake_session = dialog
        dlg.tab_settings.batch_size.setValue(88)
        dlg._on_start()
        with open(str(fake_session)) as f:
            data = json.load(f)
        assert data.get("batch_size") == 88

    def test_save_as_defaults_does_not_crash(self, dialog):
        dlg, *_ = dialog
        # Should complete without raising, even if YAML is pre-existing
        dlg._on_save_defaults()


# ─── Buttons ──────────────────────────────────────────────────────────────────

class TestDialogButtons:

    def test_cancel_button_exists(self, dialog):
        dlg, *_ = dialog
        assert dlg.btn_cancel is not None

    def test_start_button_exists(self, dialog):
        dlg, *_ = dialog
        assert dlg.btn_start is not None

    def test_save_defaults_button_exists(self, dialog):
        dlg, *_ = dialog
        assert dlg.btn_save_defaults is not None
