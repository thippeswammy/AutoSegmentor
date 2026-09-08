"""
NavigationManager.py — Undo/redo actions and navigation state for the annotation UI.

Provides QUndoCommand subclasses for point management and a NavigationState
data object tracking current frame, batch, zoom, and tool mode.
"""

from PyQt5.QtWidgets import QUndoCommand
from .logger_config import logger


class AddPointCommand(QUndoCommand):
    """Undoable command for adding an annotation point."""

    def __init__(self, handler, point, label, pose_click=None, target_models=None, description="Add Point"):
        super().__init__(description)
        self.handler = handler
        self.point = point
        self.label = label
        self.pose_click = pose_click  # Optional pose keypoint dict
        self.target_models = target_models or ["sam", "pose"]

    def redo(self):
        logger.debug(f"[CMD:AddPoint] redo  — point={self.point}  label={self.label}  pose={bool(self.pose_click)}")
        self.handler.selected_points.append(self.point)
        self.handler.selected_labels.append(self.label)
        self.handler.selected_targets.append(self.target_models)
        if self.pose_click:
            self.handler.pose_click_coords.append(self.pose_click)
            if self.pose_click.get("name") != "Negative_Point":
                self.handler.current_keypoint_index += 1
        logger.debug(f"[CMD:AddPoint] after redo — total_points={len(self.handler.selected_points)}")

    def undo(self):
        logger.debug(f"[CMD:AddPoint] undo  — removing point={self.point}  label={self.label}")
        if self.handler.selected_points:
            self.handler.selected_points.pop()
            self.handler.selected_labels.pop()
            self.handler.selected_targets.pop()
        if self.pose_click and self.handler.pose_click_coords:
            self.handler.pose_click_coords.pop()
            if self.pose_click.get("name") != "Negative_Point":
                self.handler.current_keypoint_index = max(0, self.handler.current_keypoint_index - 1)
        logger.debug(f"[CMD:AddPoint] after undo — total_points={len(self.handler.selected_points)}")


class ResetPointsCommand(QUndoCommand):
    """Undoable command for resetting all annotation points."""

    def __init__(self, handler, description="Reset All Points"):
        super().__init__(description)
        self.handler = handler
        # Snapshot current state for undo
        self.saved_points = list(handler.selected_points)
        self.saved_labels = list(handler.selected_labels)
        self.saved_targets = list(handler.selected_targets)
        self.saved_pose_clicks = list(handler.pose_click_coords)
        self.saved_keypoint_index = handler.current_keypoint_index

    def redo(self):
        logger.debug(f"[CMD:ResetPoints] redo  — clearing {len(self.saved_points)} points")
        self.handler.selected_points.clear()
        self.handler.selected_labels.clear()
        self.handler.selected_targets.clear()
        self.handler.pose_click_coords.clear()
        if self.handler.pose_mode:
            self.handler.current_keypoint_index = 0

    def undo(self):
        logger.debug(f"[CMD:ResetPoints] undo  — restoring {len(self.saved_points)} points")
        self.handler.selected_points = list(self.saved_points)
        self.handler.selected_labels = list(self.saved_labels)
        self.handler.selected_targets = list(self.saved_targets)
        self.handler.pose_click_coords = list(self.saved_pose_clicks)
        self.handler.current_keypoint_index = self.saved_keypoint_index


class DeletePointCommand(QUndoCommand):
    """Undoable command for deleting an existing annotation point."""

    def __init__(self, handler, index, description="Delete Point"):
        super().__init__(description)
        self.handler = handler
        self.index = index
        # Store state to restore on undo
        self.deleted_point = handler.selected_points[index]
        self.deleted_label = handler.selected_labels[index]
        self.deleted_targets = handler.selected_targets[index]
        self.deleted_pose_click = handler.pose_click_coords[index] if handler.pose_click_coords and index < len(handler.pose_click_coords) else None

    def redo(self):
        logger.debug(f"[CMD:DeletePoint] redo  — index={self.index}  point={self.deleted_point}  label={self.deleted_label}")
        self.handler.selected_points.pop(self.index)
        self.handler.selected_labels.pop(self.index)
        self.handler.selected_targets.pop(self.index)
        if self.deleted_pose_click:
            self.handler.pose_click_coords.pop(self.index)
            if self.deleted_pose_click.get("name") != "Negative_Point":
                self.handler.current_keypoint_index = max(0, self.handler.current_keypoint_index - 1)
        logger.debug(f"[CMD:DeletePoint] after redo — total_points={len(self.handler.selected_points)}")

    def undo(self):
        logger.debug(f"[CMD:DeletePoint] undo  — reinserting index={self.index}  point={self.deleted_point}")
        self.handler.selected_points.insert(self.index, self.deleted_point)
        self.handler.selected_labels.insert(self.index, self.deleted_label)
        self.handler.selected_targets.insert(self.index, self.deleted_targets)
        if self.deleted_pose_click:
            self.handler.pose_click_coords.insert(self.index, self.deleted_pose_click)
            if self.deleted_pose_click.get("name") != "Negative_Point":
                self.handler.current_keypoint_index += 1


class DragPointCommand(QUndoCommand):
    """Undoable command for moving an existing annotation point.

    pose_idx is the item's matching index into handler.pose_click_coords —
    it can differ from `index` (the selected_points index) whenever an
    occluded keypoint exists earlier in the frame's keypoint list, so it must
    be used (not `index`) for the pose_click_coords write.
    """

    def __init__(self, handler, index, old_pos, new_pos, pose_idx=None, description="Move Point"):
        super().__init__(description)
        self.handler = handler
        self.index = index
        self.old_pos = old_pos
        self.new_pos = new_pos
        self.pose_idx = pose_idx

    def _apply(self, pos):
        if self.index < len(self.handler.selected_points):
            self.handler.selected_points[self.index] = list(pos)
            if (self.handler.pose_mode and self.handler.pose_click_coords
                    and self.pose_idx is not None and self.pose_idx < len(self.handler.pose_click_coords)):
                self.handler.pose_click_coords[self.pose_idx]['x'] = int(pos[0])
                self.handler.pose_click_coords[self.pose_idx]['y'] = int(pos[1])

    def redo(self):
        logger.debug(f"[CMD:DragPoint] redo  — index={self.index}  {self.old_pos} -> {self.new_pos}")
        self._apply(self.new_pos)

    def undo(self):
        logger.debug(f"[CMD:DragPoint] undo  — index={self.index}  {self.new_pos} -> {self.old_pos}")
        self._apply(self.old_pos)


class CorrectPosePointCommand(QUndoCommand):
    """Undoable command for correcting an occluded/failed-tracking pose keypoint.

    Covers both drag-to-correct (new position, auto-marks visible) and the
    right-click "Mark Occluded" / "Mark Visible" toggle (position unchanged).
    Always re-syncs selected_points/labels/targets afterwards so the SAM
    prompt list reflects the corrected visibility.
    """

    def __init__(self, handler, pose_idx, old_pos, new_pos, old_visible, new_visible,
                 description="Correct Pose Point"):
        super().__init__(description)
        self.handler = handler
        self.pose_idx = pose_idx
        self.old_pos = old_pos
        self.new_pos = new_pos
        self.old_visible = old_visible
        self.new_visible = new_visible

    def _apply(self, pos, visible):
        if self.pose_idx < len(self.handler.pose_click_coords):
            kp = self.handler.pose_click_coords[self.pose_idx]
            kp['x'] = int(pos[0])
            kp['y'] = int(pos[1])
            kp['visible'] = visible
            self.handler._sync_selected_from_pose_coords()

    def redo(self):
        logger.debug(
            f"[CMD:CorrectPosePoint] redo — pose_idx={self.pose_idx}  "
            f"{self.old_pos}->{self.new_pos}  visible {self.old_visible}->{self.new_visible}"
        )
        self._apply(self.new_pos, self.new_visible)

    def undo(self):
        logger.debug(
            f"[CMD:CorrectPosePoint] undo — pose_idx={self.pose_idx}  "
            f"{self.new_pos}->{self.old_pos}  visible {self.new_visible}->{self.old_visible}"
        )
        self._apply(self.old_pos, self.old_visible)


class SkipPointCommand(QUndoCommand):
    """Undoable command for skipping an absent/occluded pose keypoint."""

    def __init__(self, handler, description="Skip Keypoint"):
        super().__init__(description)
        self.handler = handler
        self.point = [0, 0] # Dummy coordinates for skipped points
        self.label = handler.encode_label(handler.current_class_label, handler.current_instance_id) * -1 # Mark as negative
        self.pose_click = None
        if self.handler.pose_mode:
            num_kps = len(self.handler.pose_keypoints)
            if num_kps > 0:
                instance_count = self.handler.get_instance_keypoint_count()
                kp_name = self.handler.pose_keypoints[instance_count % num_kps]
                self.pose_click = {
                    "name": kp_name,
                    "point_id": len(self.handler.pose_click_coords),
                    "x": 0,
                    "y": 0,
                    "visible": False,
                    "label": self.label
                }

    def redo(self):
        logger.debug(f"[CMD:SkipPoint] redo  — keypoint={self.pose_click.get('name') if self.pose_click else 'N/A'}")
        self.handler.selected_points.append(self.point)
        self.handler.selected_labels.append(self.label)
        self.handler.selected_targets.append(["sam", "pose"])
        if self.pose_click:
            self.handler.pose_click_coords.append(self.pose_click)
            self.handler.current_keypoint_index += 1

    def undo(self):
        logger.debug(f"[CMD:SkipPoint] undo  — removing skipped keypoint={self.pose_click.get('name') if self.pose_click else 'N/A'}")
        if self.handler.selected_points:
            self.handler.selected_points.pop()
            self.handler.selected_labels.pop()
            self.handler.selected_targets.pop()
        if self.pose_click and self.handler.pose_click_coords:
            self.handler.pose_click_coords.pop()
            self.handler.current_keypoint_index = max(0, self.handler.current_keypoint_index - 1)


class NavigationState:
    """Tracks current navigation context for the annotation session."""

    def __init__(self):
        self.current_batch = 0
        self.total_batches = 1
        self.current_frame_idx = 0
        self.total_frames = 0
        self.zoom_level = 1.0
        self.tool_mode = "point"  # "point" or "navigate"
        self.show_crosshair = True
        self.show_grid = False
        self.show_zoom = True
        self.mouse_x = 0
        self.mouse_y = 0
        self.propagate_backward = True

    def set_batch_info(self, batch, total_batches, frame_idx, total_frames):
        self.current_batch = batch
        self.total_batches = total_batches
        self.current_frame_idx = frame_idx
        self.total_frames = total_frames

    def update_mouse(self, x, y):
        self.mouse_x = x
        self.mouse_y = y

    @property
    def batch_label(self):
        return f"Batch {self.current_batch + 1} / {self.total_batches}"

    @property
    def frame_label(self):
        return f"Frame {self.current_frame_idx}"

    @property
    def coord_label(self):
        return f"({self.mouse_x}, {self.mouse_y})"

    @property
    def zoom_label(self):
        return f"{self.zoom_level:.0%}"
