import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter


class CurveManager:
    def __init__(self, data_manager, plot_manager):
        self.data_manager = data_manager
        self.plot_manager = plot_manager
        self.draw_points = []
        self.is_curve = False
        self.current_line = None
        self.show_debug_plot = False  # Disabled by default

    def add_draw_point(self, x, y):
        try:
            self.draw_points.append([x, y])
            self.update_draw_line()
        except Exception as e:
            print(f"Error adding draw point: {e}")

    def update_draw_line(self):
        if self.current_line:
            self.current_line.remove()
            self.current_line = None
        if len(self.draw_points) < 2:
            return
        points = np.array(self.draw_points)
        x, y = points[:, 0], points[:, 1]
        try:
            if self.is_curve and len(points) >= 5:
                window_length = min(5, len(y) // 2 * 2 + 1)
                polyorder = 2
                y_smooth = savgol_filter(y, window_length, polyorder)
                y_smooth[0], y_smooth[-1] = y[0], y[-1]
                smooth_points = np.stack((x, y_smooth), axis=1)
                self.current_line = self.plot_manager.ax.plot(
                    smooth_points[:, 0], smooth_points[:, 1], 'k-', alpha=0.5)[0]
            else:
                self.current_line = self.plot_manager.ax.plot(x, y, 'k-', alpha=0.5)[0]
            self.plot_manager.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error updating draw line: {e}")

    def finalize_draw(self, file_id):
        if len(self.draw_points) < 2:
            self.draw_points = []
            if self.current_line:
                self.current_line.remove()
                self.current_line = None
            return
        points = np.array(self.draw_points)
        x, y = points[:, 0], points[:, 1]
        try:
            if self.is_curve and len(points) >= 5:
                window_length = min(5, len(y) // 2 * 2 + 1)
                polyorder = 2
                y_smooth = savgol_filter(y, window_length, polyorder)
                y_smooth[0], y_smooth[-1] = y[0], y[-1]
                points = np.stack((x, y_smooth), axis=1)
            for x, y in points:
                self.data_manager.add_point(x, y, file_id)
            self.draw_points = []
            if self.current_line:
                self.current_line.remove()
                self.current_line = None
            self.plot_manager.update_plot(self.data_manager.data)
        except Exception as e:
            print(f"Error finalizing draw: {e}")

    def preview_smooth(self, selected_indices, lane_id, start_idx, end_idx):
        try:
            new_points = self._smooth_segment(selected_indices, lane_id, start_idx, end_idx, preview=True)
            return new_points
        except Exception as e:
            print(f"Error previewing smooth: {e}")
            return None

    def straighten_segment(self, selected_indices, lane_id, start_idx, end_idx):
        try:
            new_points = self._smooth_segment(selected_indices, lane_id, start_idx, end_idx, preview=False)
            if new_points is None:
                return []

            selected_indices = sorted(selected_indices)
            start_pos = selected_indices.index(start_idx)
            end_pos = selected_indices.index(end_idx)
            if start_pos > end_pos:
                start_idx, end_idx = end_idx, start_idx
                start_pos, end_pos = end_pos, start_pos
            segment_indices = selected_indices[start_pos:end_pos + 1]

            if len(new_points) != len(segment_indices):
                print(f"Warning: Expected {len(segment_indices)} new points, got {len(new_points)}")
                return []

            self.data_manager.data[segment_indices, 0:2] = new_points

            for i, idx in enumerate(segment_indices):
                if i < len(new_points) - 1:
                    dx = new_points[i + 1, 0] - new_points[i, 0]
                    dy = new_points[i + 1, 1] - new_points[i, 1]
                    self.data_manager.data[idx, 2] = np.arctan2(dy, dx)
                else:
                    self.data_manager.data[idx, 2] = self.data_manager.data[segment_indices[-2], 2] if len(
                        segment_indices) > 1 else 0.0

            self.data_manager.data[segment_indices, -1] = lane_id

            self.data_manager.history.append(self.data_manager.data.copy())
            self.data_manager.redo_stack = []

            self.plot_manager.update_plot(self.data_manager.data)
            return segment_indices
        except Exception as e:
            print(f"Error straightening segment: {e}")
            return []

    def _smooth_segment(self, selected_indices, lane_id, start_idx, end_idx, preview=False):
        if len(selected_indices) < 2:
            print("Need at least 2 points to smooth")
            return None

        selected_indices = sorted(selected_indices)

        if start_idx not in selected_indices or end_idx not in selected_indices:
            print("Start or end index not in selected indices")
            return None

        start_pos = selected_indices.index(start_idx)
        end_pos = selected_indices.index(end_idx)
        if start_pos > end_pos:
            start_idx, end_idx = end_idx, start_idx
            start_pos, end_pos = end_pos, start_pos

        segment_indices = selected_indices[start_pos:end_pos + 1]
        points = self.data_manager.data[segment_indices, :2]
        start_point = self.data_manager.data[start_idx, :2]
        end_point = self.data_manager.data[end_idx, :2]

        all_indices = np.arange(len(self.data_manager.data))
        selected_set = set(segment_indices)
        prev_point, next_point = None, None

        for idx in reversed(all_indices):
            if idx < start_idx and idx not in selected_set:
                prev_point = self.data_manager.data[idx, :2]
                break

        for idx in all_indices:
            if idx > end_idx and idx not in selected_set:
                next_point = self.data_manager.data[idx, :2]
                break

        fitting_points = points.copy()
        weights = np.ones(len(fitting_points)) * 30
        if prev_point is not None:
            fitting_points = np.vstack([prev_point, fitting_points])
            weights = np.concatenate(([1], weights))
        if next_point is not None:
            fitting_points = np.vstack([fitting_points, next_point])
            weights = np.concatenate((weights, [1]))

        try:
            x, y = fitting_points[:, 0], fitting_points[:, 1]
            distances = np.sqrt(np.sum(np.diff(fitting_points, axis=0) ** 2, axis=1))
            u = np.zeros(len(fitting_points))
            u[1:] = np.cumsum(distances)
            u = u / u[-1] if u[-1] > 0 else np.linspace(0, 1, len(fitting_points))

            smoothing_factor = len(points) * self.plot_manager.slider_smooth.val
            tck, u_fitted = splprep([x, y], u=u, s=smoothing_factor, k=3, w=weights)

            start_idx_u = 1 if prev_point is not None else 0
            end_idx_u = len(fitting_points) - (2 if next_point is not None else 1)
            u_segment = u_fitted[start_idx_u:end_idx_u]
            u_segment_normalized = (u_segment - u_segment[0]) / (u_segment[-1] - u_segment[0]) \
                if u_segment[-1] != u_segment[0] else np.linspace(0, 1, len(u_segment))

            num_new_points = len(segment_indices)
            u_fine = np.linspace(0, 1, num_new_points)
            x_smooth, y_smooth = splev(u_fine, tck)

            new_points = np.stack((x_smooth, y_smooth), axis=1)
            new_points[0] = start_point
            new_points[-1] = end_point

            if self.show_debug_plot and not preview:
                plt.figure(figsize=(8, 6))
                plt.plot(points[:, 0], points[:, 1], 'ro-', label='Original')
                plt.plot(new_points[:, 0], new_points[:, 1], 'g.-', label='Smoothed')
                if prev_point is not None:
                    plt.plot([prev_point[0], points[0, 0]], [prev_point[1], points[0, 1]], 'b--', label='Prev Adjacent')
                if next_point is not None:
                    plt.plot([points[-1, 0], next_point[0]], [points[-1, 1], next_point[1]], 'b--',
                             label='Next Adjacent')
                plt.legend()
                plt.title("Smoothing Segment")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.grid(True)
                plt.show()

            return new_points
        except ValueError as e:
            print(f"Spline fitting failed: {e}")
            return None
