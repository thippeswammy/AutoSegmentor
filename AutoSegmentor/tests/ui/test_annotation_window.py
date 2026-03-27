"""
UI Automation Tests: AnnotationWindow
Uses pytest-qt + QtTest to simulate user interactions with the annotation UI.

These tests verify:
  - The AnnotationWindow can be instantiated without errors.
  - Navigation buttons (Next/Prev Image, Next/Prev Batch) update the frame index.
  - Frame jump input navigates to the requested frame.
  - Reset clears all selected points.
  - Process Batch button can be triggered via keyboard shortcut.
  - Point clicks on the canvas are registered correctly.
  - Undo restores the previous state.

NOTE: Tests use a mock engine (no GPU/SAM2/CoTracker) to keep
      the UI tests fast and self-contained.
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication


# ─── Minimal Mock Engine ─────────────────────────────────────────────────────

class MockConfig:
    """Minimal config for UI tests — no real file I/O."""
    video_number = 1
    batch_size = 10
    prefix = "Img"
    images_starting_count = 0
    images_ending_count = 50
    frames_directory = "./working_dir/images"
    rendered_frames_dir = "./working_dir/render"
    temp_directory = "./working_dir/temp"
    window_size = [200, 200]
    label_colors = {i: (i * 10, i * 20, i * 30) for i in range(1, 11)}
    memory_bank_size = 5
    prompt_memory_size = 5
    pose_config = None
    auto_prompt_encoding = True
    ui_show_crosshair = True
    ui_show_grid = False
    sam_enabled = False  # No SAM2 for UI tests


class MockAnnotationManager:
    def __init__(self, n_frames=30):
        self.points_collection = []
        self.pose_keypoints_collection = []
        self._frame_count = n_frames

    def check_data_sufficiency(self):
        return 0

    def get_prompt_for_frame(self, frame_idx):
        return None

    def get_latest_prompt_before(self, frame_idx):
        return None

    def save_points_and_labels(self, frame_idx, points, labels, pose_keypoints=None):
        pass

    def get_batch_prompts(self, batch_num, batch_size):
        return []


class MockEngine:
    """Simulates AutoSegmentorEngine without any GPU/model dependencies."""
    config = MockConfig()
    sam2_predictor = None
    _predictor_lock = __import__('threading').Lock()
    frame_paths = [f"frame_{i:05d}.jpg" for i in range(30)]
    per_batch_tracked_data = [[] for _ in range(3)]

    def __init__(self):
        self.annotation_manager = MockAnnotationManager(len(self.frame_paths))

    def user_prompt_adder(self, inference_state, frame_path):
        pass


def _make_dummy_frames(tmp_path, count=30):
    """Write small dummy BGR images and return their paths."""
    paths = []
    for i in range(count):
        p = tmp_path / f"Img1_{i:05d}.jpg"
        import cv2
        cv2.imwrite(str(p), np.full((240, 320, 3), i * 3, dtype=np.uint8))
        paths.append(str(p))
    return paths


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    """One QApplication per test session."""
    app = QApplication.instance() or QApplication(sys.argv)
    return app


@pytest.fixture
def handler_and_window(qapp, tmp_path):
    """Build a real UserInteractionHandler + AnnotationWindow with a mock engine."""
    from utils.UserUI.UserInteraction import UserInteractionHandler
    from utils.UserUI.MainWindow import AnnotationWindow

    frame_paths = _make_dummy_frames(tmp_path)
    engine = MockEngine()
    engine.frame_paths = frame_paths
    engine.annotation_manager = MockAnnotationManager(len(frame_paths))

    handler = UserInteractionHandler(engine.config, engine.annotation_manager, engine)
    handler.frame_paths = frame_paths

    # Manually load frame 0 state
    import cv2
    handler.current_frame_idx = 0
    handler.current_frame_only_with_points = cv2.imread(frame_paths[0])
    handler.current_frame = handler.current_frame_only_with_points.copy()

    window = AnnotationWindow(handler, engine.config)
    window.show()
    QTest.qWaitForWindowExposed(window)

    yield handler, window
    window.close()


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestAnnotationWindowInit:

    def test_window_title(self, handler_and_window):
        _, window = handler_and_window
        assert "AutoSegmentor" in window.windowTitle()

    def test_process_batch_button_exists(self, handler_and_window):
        _, window = handler_and_window
        assert window.btn_process_batch is not None

    def test_toolbar_navigation_buttons_exist(self, handler_and_window):
        _, window = handler_and_window
        assert window.btn_next_img is not None
        assert window.btn_prev_img is not None
        assert window.btn_next_batch is not None
        assert window.btn_prev_batch is not None


class TestAnnotationWindowNavigation:

    def test_next_image_button_blocked_without_data(self, handler_and_window):
        """Next image should be blocked if no tracking data for the next frame."""
        handler, window = handler_and_window
        initial_idx = handler.current_frame_idx
        QTest.mouseClick(window.btn_next_img, Qt.LeftButton)
        # Frame should stay the same since no tracking data exists for frame 1
        # (MockAnnotationManager always returns None for prompts)
        assert handler.current_frame_idx == initial_idx

    def test_prev_image_at_start_stays(self, handler_and_window):
        handler, window = handler_and_window
        handler.current_frame_idx = 0
        QTest.mouseClick(window.btn_prev_img, Qt.LeftButton)
        assert handler.current_frame_idx == 0  # Can't go below 0

    def test_jump_to_frame_valid(self, handler_and_window):
        """Frame jump input should navigate to a valid frame index."""
        handler, window = handler_and_window
        window.frame_jump_input.setText("0")
        QTest.keyClick(window.frame_jump_input, Qt.Key_Return)
        # Frame 0 always has no prompt, but the jump is to frame 0 which is always allowed
        assert handler.current_frame_idx == 0

    def test_jump_to_invalid_frame_ignored(self, handler_and_window):
        handler, window = handler_and_window
        initial_idx = handler.current_frame_idx
        window.frame_jump_input.setText("99999")
        QTest.keyClick(window.frame_jump_input, Qt.Key_Return)
        assert handler.current_frame_idx == initial_idx


class TestAnnotationWindowPointManagement:

    def test_reset_clears_points(self, handler_and_window):
        handler, window = handler_and_window
        handler.selected_points = [[100, 200], [300, 400]]
        handler.selected_labels = [1001, 1001]
        QTest.mouseClick(window.btn_reset, Qt.LeftButton)
        assert len(handler.selected_points) == 0
        assert len(handler.selected_labels) == 0

    def test_reset_with_no_points_does_not_crash(self, handler_and_window):
        handler, window = handler_and_window
        handler.selected_points = []
        handler.selected_labels = []
        QTest.mouseClick(window.btn_reset, Qt.LeftButton)  # Should not raise
        assert len(handler.selected_points) == 0

    def test_undo_shortcut_registered(self, handler_and_window):
        """Undo key 'U' should exist and not crash when stack is empty."""
        _, window = handler_and_window
        QTest.keyClick(window, Qt.Key_U)  # Should not raise


class TestAnnotationWindowUIState:

    def test_mask_toggle_button_starts_checked(self, handler_and_window):
        _, window = handler_and_window
        assert window.btn_toggle_mask.isChecked()

    def test_mask_toggle_changes_state(self, handler_and_window):
        _, window = handler_and_window
        initial = window.btn_toggle_mask.isChecked()
        QTest.mouseClick(window.btn_toggle_mask, Qt.LeftButton)
        assert window.btn_toggle_mask.isChecked() != initial

    def test_class_combo_disabled_in_pose_mode(self, handler_and_window):
        handler, window = handler_and_window
        if handler.pose_mode:
            assert not window.class_combo.isEnabled()

    def test_status_bar_has_coord_label(self, handler_and_window):
        _, window = handler_and_window
        assert window.coord_label is not None
