"""
NavigationManager.py — Undo/redo actions and navigation state for the annotation UI.

Provides QUndoCommand subclasses for point management and a NavigationState
data object tracking current frame, batch, zoom, and tool mode.
"""

from PyQt5.QtWidgets import QUndoCommand


class AddPointCommand(QUndoCommand):
    """Undoable command for adding an annotation point."""

    def __init__(self, handler, point, label, pose_click=None, description="Add Point"):
        super().__init__(description)
        self.handler = handler
        self.point = point
        self.label = label
        self.pose_click = pose_click  # Optional pose keypoint dict

    def redo(self):
        self.handler.selected_points.append(self.point)
        self.handler.selected_labels.append(self.label)
        if self.pose_click:
            self.handler.pose_click_coords.append(self.pose_click)
            self.handler.current_keypoint_index += 1

    def undo(self):
        if self.handler.selected_points:
            self.handler.selected_points.pop()
            self.handler.selected_labels.pop()
        if self.pose_click and self.handler.pose_click_coords:
            self.handler.pose_click_coords.pop()
            self.handler.current_keypoint_index = max(0, self.handler.current_keypoint_index - 1)


class ResetPointsCommand(QUndoCommand):
    """Undoable command for resetting all annotation points."""

    def __init__(self, handler, description="Reset All Points"):
        super().__init__(description)
        self.handler = handler
        # Snapshot current state for undo
        self.saved_points = list(handler.selected_points)
        self.saved_labels = list(handler.selected_labels)
        self.saved_pose_clicks = list(handler.pose_click_coords)
        self.saved_keypoint_index = handler.current_keypoint_index

    def redo(self):
        self.handler.selected_points.clear()
        self.handler.selected_labels.clear()
        self.handler.pose_click_coords.clear()
        if self.handler.pose_mode:
            self.handler.current_keypoint_index = 0

    def undo(self):
        self.handler.selected_points = list(self.saved_points)
        self.handler.selected_labels = list(self.saved_labels)
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
        self.deleted_pose_click = handler.pose_click_coords[index] if handler.pose_click_coords and index < len(handler.pose_click_coords) else None

    def redo(self):
        self.handler.selected_points.pop(self.index)
        self.handler.selected_labels.pop(self.index)
        if self.deleted_pose_click:
            self.handler.pose_click_coords.pop(self.index)
            self.handler.current_keypoint_index = max(0, self.handler.current_keypoint_index - 1)

    def undo(self):
        self.handler.selected_points.insert(self.index, self.deleted_point)
        self.handler.selected_labels.insert(self.index, self.deleted_label)
        if self.deleted_pose_click:
            self.handler.pose_click_coords.insert(self.index, self.deleted_pose_click)
            self.handler.current_keypoint_index += 1


class DragPointCommand(QUndoCommand):
    """Undoable command for moving an existing annotation point."""

    def __init__(self, handler, index, old_pos, new_pos, description="Move Point"):
        super().__init__(description)
        self.handler = handler
        self.index = index
        self.old_pos = old_pos
        self.new_pos = new_pos

    def redo(self):
        if self.index < len(self.handler.selected_points):
            self.handler.selected_points[self.index] = list(self.new_pos)
            if self.handler.pose_mode and self.handler.pose_click_coords and self.index < len(self.handler.pose_click_coords):
                self.handler.pose_click_coords[self.index]['x'] = int(self.new_pos[0])
                self.handler.pose_click_coords[self.index]['y'] = int(self.new_pos[1])
            if hasattr(self.handler, 'user_prompt_adder_pyqt'):
                self.handler.user_prompt_adder_pyqt()

    def undo(self):
        if self.index < len(self.handler.selected_points):
            self.handler.selected_points[self.index] = list(self.old_pos)
            if self.handler.pose_mode and self.handler.pose_click_coords and self.index < len(self.handler.pose_click_coords):
                self.handler.pose_click_coords[self.index]['x'] = int(self.old_pos[0])
                self.handler.pose_click_coords[self.index]['y'] = int(self.old_pos[1])
            if hasattr(self.handler, 'user_prompt_adder_pyqt'):
                self.handler.user_prompt_adder_pyqt()


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
                kp_name = self.handler.pose_keypoints[self.handler.current_keypoint_index % num_kps]
                self.pose_click = {
                    "name": kp_name,
                    "point_id": self.handler.current_keypoint_index,
                    "x": 0,
                    "y": 0,
                    "visible": False,
                    "label": self.label
                }

    def redo(self):
        self.handler.selected_points.append(self.point)
        self.handler.selected_labels.append(self.label)
        if self.pose_click:
            self.handler.pose_click_coords.append(self.pose_click)
            self.handler.current_keypoint_index += 1

    def undo(self):
        if self.handler.selected_points:
            self.handler.selected_points.pop()
            self.handler.selected_labels.pop()
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
