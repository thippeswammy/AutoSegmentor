import os
import sys
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Mock things needed for the test
from tests.ui.test_annotation_window import MockEngine, MockAnnotationManager, _make_dummy_frames

def test_drag_point_triggers_preview(tmp_path):
    """Verify that moving a point triggers the SAM2 preview update (no qtbot)."""
    # Create QApplication for UI widgets
    app = QApplication.instance() or QApplication(sys.argv)
    
    from utils.UserUI.UserInteraction import UserInteractionHandler
    from utils.UserUI.MainWindow import AnnotationWindow
    import cv2

    frame_paths = _make_dummy_frames(tmp_path, count=5)
    engine = MockEngine()
    engine.frame_paths = frame_paths
    engine.annotation_manager = MockAnnotationManager(len(frame_paths))

    handler = UserInteractionHandler(engine.config, engine.annotation_manager, engine)
    handler.frame_paths = frame_paths
    handler.current_frame_idx = 0
    handler.current_frame_only_with_points = cv2.imread(frame_paths[0])
    handler.current_frame = handler.current_frame_only_with_points.copy()
    handler._raw_frame = handler.current_frame.copy()

    # Pre-populate with some points
    handler.selected_points = [[50, 50], [100, 100]]
    handler.selected_labels = [1001, 1001]

    window = AnnotationWindow(handler, engine.config)
    
    # ── Test Logic ───────────────────────────────────────────────────────────
    assert len(handler.selected_points) == 2
    assert handler.selected_points[0] == [50, 50]
    
    spy = []
    original_start = window._start_preview_thread
    def mocked_start():
        spy.append(True)
        original_start()
    window._start_preview_thread = mocked_start

    # Move point 0 from [50, 50] to [60, 60]
    window.handle_point_moved(0, 50, 50, 60, 60)

    # 1. Verify data update
    assert handler.selected_points[0] == [60, 60]
    
    # 2. Verify command push
    assert window.undo_stack.count() == 1
    
    # 3. Verify interaction with preview thread (ASYNC TRIGGER)
    # The _on_undo_stack_changed should have caught the indexChanged signal.
    # QApp.processEvents() will process the signal-slot connection.
    app.processEvents()
    
    assert len(spy) > 0, "SAM2 preview was not triggered after move!"
    
    # 4. Verify Undo
    spy.clear()
    window.undo_stack.undo()
    assert handler.selected_points[0] == [50, 50]
    app.processEvents()
    assert len(spy) > 0, "SAM2 preview was not triggered after undo!"

    print("Test passed successfully!")
    window.close()

if __name__ == "__main__":
    import pathlib
    # Mock tmp_path for standalone run
    tmp = pathlib.Path("./tmp_test")
    tmp.mkdir(exist_ok=True)
    try:
        test_drag_point_triggers_preview(tmp)
    finally:
        import shutil
        shutil.rmtree(tmp)
