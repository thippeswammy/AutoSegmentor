import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.widgets import Button
from curve_manager import CurveManager
from data_loader import DataLoader

class EventHandler:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.plot_manager = None
        self.curve_manager = None
        self.selection_mode = True
        self.draw_mode = False
        self.selected_id = 0
        self.id_set = True
        self.smoothing_point_selection = False
        self.smoothing_start_idx = None
        self.smoothing_end_idx = None
        self.smoothing_selected_indices = None
        self.smoothing_lane_id = None
        self.smoothing_preview_line = None
        self.merge_mode = False
        self.merge_point_1 = None
        self.merge_point_2 = None
        self.merge_lane_1 = None
        self.merge_lane_2 = None
        self.merge_point_1_type = None
        self.merge_point_2_type = None
        self.buttons = {}
        self.status_timeout = 5  # seconds
        self.last_status_time = 0

    def set_plot_manager(self, plot_manager):
        self.plot_manager = plot_manager
        self.curve_manager = CurveManager(self.data_manager, self.plot_manager)
        self.fig = self.plot_manager.fig
        self.setup_event_handlers()
        self.setup_buttons()
        self.update_button_states()

    def setup_event_handlers(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.plot_manager.rs.onselect = self.on_select

    def setup_buttons(self):
        ax_toggle = plt.axes([0.01, 0.95, 0.1, 0.04])
        self.buttons['toggle'] = Button(ax_toggle, 'Select Mode')
        self.buttons['toggle'].on_clicked(self.on_toggle_mode)

        ax_draw = plt.axes([0.01, 0.90, 0.1, 0.04])
        self.buttons['draw'] = Button(ax_draw, 'Draw')
        self.buttons['draw'].on_clicked(self.on_toggle_draw_mode)

        ax_linecurve = plt.axes([0.01, 0.85, 0.1, 0.04])
        self.buttons['linecurve'] = Button(ax_linecurve, 'Line')
        self.buttons['linecurve'].on_clicked(self.on_toggle_linecurve)

        ax_straighten = plt.axes([0.01, 0.80, 0.1, 0.04])
        self.buttons['straighten'] = Button(ax_straighten, 'Smooth')
        self.buttons['straighten'].on_clicked(self.on_straighten)

        ax_confirm_start = plt.axes([0.01, 0.75, 0.1, 0.04])
        self.buttons['confirm_start'] = Button(ax_confirm_start, 'Confirm Start')
        self.buttons['confirm_start'].on_clicked(self.on_confirm_start)

        ax_confirm_end = plt.axes([0.01, 0.70, 0.1, 0.04])
        self.buttons['confirm_end'] = Button(ax_confirm_end, 'Confirm End')
        self.buttons['confirm_end'].on_clicked(self.on_confirm_end)

        ax_cancel = plt.axes([0.01, 0.65, 0.1, 0.04])
        self.buttons['cancel'] = Button(ax_cancel, 'Cancel Smooth')
        self.buttons['cancel'].on_clicked(self.on_cancel_smoothing)

        ax_clear = plt.axes([0.01, 0.60, 0.1, 0.04])
        self.buttons['clear'] = Button(ax_clear, 'Clear Selection')
        self.buttons['clear'].on_clicked(self.on_clear_selection)

        ax_save = plt.axes([0.01, 0.55, 0.1, 0.04])
        self.buttons['save'] = Button(ax_save, 'Save')
        self.buttons['save'].on_clicked(self.save_data)

        ax_merge = plt.axes([0.01, 0.50, 0.1, 0.04])
        self.buttons['merge'] = Button(ax_merge, 'Merge Lanes')
        self.buttons['merge'].on_clicked(self.merge_lanes)

        ax_export = plt.axes([0.01, 0.45, 0.1, 0.04])
        self.buttons['export'] = Button(ax_export, 'Export Selected')
        self.buttons['export'].on_clicked(self.export_selected)

        ax_grid = plt.axes([0.01, 0.40, 0.1, 0.04])
        self.buttons['grid'] = Button(ax_grid, 'Toggle Grid')
        self.buttons['grid'].on_clicked(self.toggle_grid)

        self.fig.canvas.draw()

    def update_button_states(self):
        self.buttons['linecurve'].eventson = self.draw_mode
        self.buttons['linecurve'].ax.set_facecolor('white' if self.draw_mode else 'lightgray')
        self.buttons['linecurve'].label.set_color('black' if self.draw_mode else 'gray')
        self.buttons['straighten'].eventson = self.selection_mode
        self.buttons['straighten'].ax.set_facecolor('white' if self.selection_mode else 'lightgray')
        self.buttons['straighten'].label.set_color('black' if self.selection_mode else 'gray')
        self.buttons['confirm_start'].eventson = self.smoothing_point_selection
        self.buttons['confirm_start'].ax.set_facecolor('white' if self.smoothing_point_selection else 'lightgray')
        self.buttons['confirm_start'].label.set_color('black' if self.smoothing_point_selection else 'gray')
        self.buttons['confirm_end'].eventson = self.smoothing_point_selection
        self.buttons['confirm_end'].ax.set_facecolor('white' if self.smoothing_point_selection else 'lightgray')
        self.buttons['confirm_end'].label.set_color('black' if self.smoothing_point_selection else 'gray')
        self.buttons['cancel'].eventson = self.smoothing_point_selection
        self.buttons['cancel'].ax.set_facecolor('white' if self.smoothing_point_selection else 'lightgray')
        self.buttons['cancel'].label.set_color('black' if self.smoothing_point_selection else 'gray')
        self.buttons['export'].eventson = bool(self.plot_manager.selected_indices)
        self.buttons['export'].ax.set_facecolor('white' if self.plot_manager.selected_indices else 'lightgray')
        self.buttons['export'].label.set_color('black' if self.plot_manager.selected_indices else 'gray')
        self.fig.canvas.draw_idle()

    def toggle_grid(self, event):
        self.plot_manager.grid_visible = not self.plot_manager.grid_visible
        self.plot_manager.ax.grid(self.plot_manager.grid_visible)
        self.update_status(f"Grid {'enabled' if self.plot_manager.grid_visible else 'disabled'}")
        self.fig.canvas.draw()

    def on_toggle_mode(self, event):
        self.draw_mode = False
        self.selection_mode = not self.selection_mode
        self.plot_manager.rs.set_active(self.selection_mode)
        self.buttons['toggle'].label.set_text('Select Mode' if self.selection_mode else 'Add/Delete Mode')
        self.buttons['toggle'].color = 'lightcoral' if self.selection_mode else 'lightgreen'
        if not self.selection_mode:
            self.id_set = True
            self.clear_smoothing_state()
            if self.plot_manager.selected_indices:
                self.plot_manager.selected_indices = []
                self.update_point_sizes()
                print("Cleared selection")
        print(f"Entered {'selection' if self.selection_mode else 'add/delete'} mode")
        self.update_button_states()
        self.update_status()

    def on_toggle_draw_mode(self, event):
        self.selection_mode = False
        self.draw_mode = not self.draw_mode
        self.plot_manager.rs.set_active(False)
        self.buttons['toggle'].label.set_text('Select Mode')
        self.buttons['toggle'].color = 'lightcoral'
        if not self.draw_mode:
            self.id_set = True
            self.clear_smoothing_state()
            self.curve_manager.draw_points = []
            if self.curve_manager.current_line:
                self.curve_manager.current_line.remove()
                self.curve_manager.current_line = None
                self.plot_manager.fig.canvas.draw_idle()
            if self.plot_manager.selected_indices:
                self.plot_manager.selected_indices = []
                self.update_point_sizes()
                print("Cleared selection")
        print(f"Entered {'draw' if self.draw_mode else 'add/delete'} mode")
        self.update_button_states()
        self.update_status()

    def on_toggle_linecurve(self, event):
        if not self.draw_mode:
            print("Must be in Draw Mode to toggle line/curve")
            self.update_status("Enter Draw Mode first")
            return
        self.curve_manager.is_curve = not self.curve_manager.is_curve
        self.buttons['linecurve'].label.set_text('Curve' if self.curve_manager.is_curve else 'Line')
        self.curve_manager.update_draw_line()
        print(f"Drawing {'curve' if self.curve_manager.is_curve else 'line'}")
        self.update_status()

    def on_straighten(self, event):
        if not self.selection_mode or not self.plot_manager.selected_indices:
            print("Must be in Selection Mode with points selected to smooth")
            self.update_status("Select points in Selection Mode")
            return
        self.smoothing_selected_indices = self.plot_manager.selected_indices
        self.smoothing_lane_id = self.selected_id
        self.smoothing_point_selection = True
        self.smoothing_start_idx = None
        self.smoothing_end_idx = None
        self.update_point_sizes()
        self.update_button_states()
        print("Please click on the starting point for smoothing")
        self.update_status("Click to select smoothing start point")

    def on_confirm_start(self, event):
        if not self.smoothing_point_selection or self.smoothing_start_idx is None:
            print("Please select the starting point first")
            self.update_status("Select start point first")
            return
        print(f"Confirmed start point (index {self.smoothing_start_idx})")
        self.update_point_sizes()
        self.update_status("Click to select smoothing end point")

    def on_confirm_end(self, event):
        if not self.smoothing_point_selection or self.smoothing_start_idx is None or self.smoothing_end_idx is None:
            print("Please select both start and end points")
            self.update_status("Select both start and end points")
            return
        print(f"End point confirmed (index {self.smoothing_end_idx})")
        new_indices = self.curve_manager.straighten_segment(
            self.smoothing_selected_indices,
            self.smoothing_lane_id,
            self.smoothing_start_idx,
            self.smoothing_end_idx
        )
        self.clear_smoothing_state()
        if new_indices:
            self.plot_manager.selected_indices = new_indices
            self.plot_manager.update_plot(self.data_manager.data)
            print(f"Smoothed {len(new_indices)} points")
            self.update_status(f"Smoothed {len(new_indices)} points")
        else:
            self.plot_manager.selected_indices = []
            self.update_point_sizes()
            print("Smoothing failed, selection cleared")
            self.update_status("Smoothing failed")
        self.update_button_states()

    def on_cancel_smoothing(self, event):
        print("Smoothing canceled")
        self.clear_smoothing_state()
        self.update_point_sizes()
        self.update_button_states()
        self.update_status("Smoothing canceled")

    def on_clear_selection(self, event):
        if self.plot_manager.selected_indices:
            self.plot_manager.selected_indices = []
            print("Cleared selection")
        self.clear_smoothing_state()
        self.clear_merge_state()
        self.update_point_sizes()
        self.update_button_states()
        self.update_status("Selection cleared")

    def clear_smoothing_state(self):
        self.smoothing_point_selection = False
        self.smoothing_start_idx = None
        self.smoothing_end_idx = None
        self.smoothing_selected_indices = None
        self.smoothing_lane_id = None
        if self.smoothing_preview_line:
            self.smoothing_preview_line.remove()
            self.smoothing_preview_line = None
            self.plot_manager.fig.canvas.draw_idle()

    def clear_merge_state(self):
        self.merge_mode = False
        self.merge_point_1 = None
        self.merge_point_2 = None
        self.merge_lane_1 = None
        self.merge_lane_2 = None
        self.merge_point_1_type = None
        self.merge_point_2_type = None
        self.update_status()

    def merge_lanes(self, event):
        unique_lanes = np.unique(self.data_manager.data[:, -1])
        if len(unique_lanes) <= 1:
            print("Only one lane present, no merging needed")
            self.update_status("Only one lane present")
            self.clear_merge_state()
            return
        self.merge_mode = True
        self.merge_point_1 = None
        self.merge_point_2 = None
        self.merge_lane_1 = None
        self.merge_lane_2 = None
        self.merge_point_1_type = None
        self.merge_point_2_type = None
        print("Please select first point (start or end)")
        self.update_status("Select first point (start or end)")
        self.update_point_sizes()

    def finalize_merge(self):
        if self.merge_point_1 is None or self.merge_point_2 is None or self.merge_lane_1 == self.merge_lane_2:
            print("Two different lanes must be selected for merging")
            self.update_status("Select two different lanes")
            self.clear_merge_state()
            return
        self.data_manager.merge_lanes(
            self.merge_lane_1, self.merge_lane_2,
            self.merge_point_1, self.merge_point_2,
            self.merge_point_1_type, self.merge_point_2_type
        )
        self.data_manager.save_all_lanes()
        self.data_manager.clear_data()
        temp_loader = DataLoader("workspace-Temp")
        data, file_names = temp_loader.load_data()
        self.data_manager.__init__(data, file_names)
        self.plot_manager.selected_indices = []
        self.plot_manager.file_names = file_names
        self.plot_manager.update_plot(self.data_manager.data)
        print(f"Merged lane {self.merge_lane_2} into lane {self.merge_lane_1}")
        self.clear_merge_state()
        self.update_point_sizes()
        self.update_status(f"Merged lane {self.merge_lane_2} into lane {self.merge_lane_1}")

    def save_data(self, event):
        filename = self.data_manager.save()
        if filename:
            print(f"Saved to {filename}")
            self.update_status(f"Saved to {filename}")
        else:
            self.update_status("Save failed")

    def export_selected(self, event):
        if not self.plot_manager.selected_indices:
            print("No points selected to export")
            self.update_status("Select points to export")
            return
        try:
            selected_points = self.data_manager.data[np.array(self.plot_manager.selected_indices, dtype=int)]
            filename = f"selected_points_{int(time.time())}.npy"
            np.save(filename, selected_points[:, :3])
            print(f"Exported {len(selected_points)} points to {filename}")
            self.update_status(f"Exported {len(selected_points)} points")
        except Exception as e:
            print(f"Error exporting points: {e}")
            self.update_status("Export failed")

    def on_click(self, event):
        if self.plot_manager is None or event.inaxes != self.plot_manager.ax or event.button != 1:
            return
        if self.merge_mode:
            click_x, click_y = event.xdata, event.ydata
            distances = np.sqrt(
                (self.data_manager.data[:, 0] - click_x) ** 2 +
                (self.data_manager.data[:, 1] - click_y) ** 2
            )
            closest_idx = np.argmin(distances)
            lane_id = int(self.data_manager.data[closest_idx, -1])
            lane_indices = np.where(self.data_manager.data[:, -1] == lane_id)[0]
            lane_data = self.data_manager.data[lane_indices]
            min_idx = lane_data[:, 4].argmin()
            max_idx = lane_data[:, 4].argmax()
            point_type = 'start' if closest_idx == lane_indices[min_idx] else 'end' if closest_idx == lane_indices[max_idx] else None

            if point_type is None:
                print("Please select a start or end point")
                self.update_status("Select a start or end point")
                return

            if self.merge_point_1 is None:
                self.merge_point_1 = closest_idx
                self.merge_lane_1 = lane_id
                self.merge_point_1_type = point_type
                print(f"Selected {point_type} point in lane {lane_id} (index {closest_idx})")
                self.update_status("Select second point in different lane")
                self.update_point_sizes()
                self.plot_manager.fig.canvas.draw_idle()
            elif self.merge_point_2 is None and lane_id != self.merge_lane_1:
                self.merge_point_2 = closest_idx
                self.merge_lane_2 = lane_id
                self.merge_point_2_type = point_type
                print(f"Selected {point_type} point in lane {lane_id} (index {closest_idx})")
                self.finalize_merge()
            return
        if self.smoothing_point_selection:
            click_x, click_y = event.xdata, event.ydata
            selected_points = self.data_manager.data[self.smoothing_selected_indices, :2]
            distances = np.sqrt((selected_points[:, 0] - click_x) ** 2 + (selected_points[:, 1] - click_y) ** 2)
            closest_idx = np.argmin(distances)
            global_idx = self.smoothing_selected_indices[closest_idx]
            if self.smoothing_start_idx is None:
                self.smoothing_start_idx = global_idx
                self.update_point_sizes()
                print(f"Start point selected (index {self.smoothing_start_idx})")
                self.update_status("Confirm start or select end point")
            elif self.smoothing_end_idx is None:
                self.smoothing_end_idx = global_idx
                self.update_point_sizes()
                preview_points = self.curve_manager.preview_smooth(
                    self.smoothing_selected_indices,
                    self.smoothing_lane_id,
                    self.smoothing_start_idx,
                    self.smoothing_end_idx
                )
                if preview_points is not None:
                    if self.smoothing_preview_line:
                        self.smoothing_preview_line.remove()
                    self.smoothing_preview_line = self.plot_manager.ax.plot(
                        preview_points[:, 0], preview_points[:, 1], 'b--', alpha=0.5, label='Preview')[0]
                    self.plot_manager.ax.legend()
                    self.plot_manager.fig.canvas.draw_idle()
                print(f"End point selected (index {self.smoothing_end_idx})")
                self.update_status("Confirm end or cancel")
            return
        if self.draw_mode:
            self.curve_manager.add_draw_point(event.xdata, event.ydata)
            print(f"Added point to {'curve' if self.curve_manager.is_curve else 'line'}")
            self.update_status()
            return
        if self.selection_mode:
            return
        self.plot_manager.selected_indices = []
        self.update_point_sizes()
        self.data_manager.add_point(event.xdata, event.ydata, self.selected_id)
        self.plot_manager.update_plot(self.data_manager.data)
        print(f"Added point with ID {self.selected_id} ({self.data_manager.file_names[self.selected_id]})")
        self.update_status("Point added")

    def update_point_sizes(self):
        if self.plot_manager is None:
            print("Plot manager not set, skipping update_point_sizes")
            return
        try:
            for plot_idx, sc in enumerate(self.plot_manager.lane_scatter_plots):
                indices = self.plot_manager.indices[plot_idx]
                if len(indices) == 0:
                    continue
                lane_id = int(self.data_manager.data[indices[0], -1])
                base_size = 20 if self.plot_manager.highlighted_lane == lane_id else 10
                sizes = np.full(len(indices), base_size, dtype=float)
                for local_idx, global_idx in enumerate(indices):
                    if self.merge_mode:
                        if global_idx == self.merge_point_1:
                            sizes[local_idx] = 100
                        elif global_idx == self.merge_point_2:
                            sizes[local_idx] = 80
                    elif self.smoothing_point_selection:
                        if global_idx == self.smoothing_start_idx:
                            sizes[local_idx] = 100
                        elif global_idx == self.smoothing_end_idx:
                            sizes[local_idx] = 80
                        elif global_idx in self.smoothing_selected_indices:
                            sizes[local_idx] = 50
                    elif global_idx in self.plot_manager.selected_indices:
                        sizes[local_idx] = 30
                sc.set_sizes(sizes)
            self.plot_manager.fig.canvas.draw_idle()
            self.plot_manager.fig.canvas.flush_events()
            self.update_button_states()
        except Exception as e:
            print(f"Error updating point sizes: {e}")

    def on_pick(self, event):
        if self.plot_manager is None or event.mouseevent.button != 3 or self.plot_manager.rs.active:
            return
        if not self.selection_mode:
            self.plot_manager.selected_indices = []
            self.update_point_sizes()
        artist = event.artist
        if artist not in self.plot_manager.lane_scatter_plots:
            return
        ind = event.ind[0]
        file_index = self.plot_manager.lane_scatter_plots.index(artist)
        global_ind = self.plot_manager.indices[file_index][ind]
        self.data_manager.delete_points([global_ind])
        self.plot_manager.selected_indices = [i for i in self.plot_manager.selected_indices if i != global_ind]
        self.plot_manager.update_plot(self.data_manager.data)
        self.update_status("Point deleted")

    def on_select(self, eclick, erelease):
        if not self.selection_mode:
            return
        try:
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            self.plot_manager.selected_indices = [
                i for i in range(len(self.data_manager.data))
                if x_min <= self.data_manager.data[i, 0] <= x_max and
                   y_min <= self.data_manager.data[i, 1] <= y_max
            ]
            print(f"Selected {len(self.plot_manager.selected_indices)} points")
            self.update_point_sizes()
            self.update_status(f"Selected {len(self.plot_manager.selected_indices)} points")
        except Exception as e:
            print(f"Error during selection: {e}")

    def on_key(self, event):
        key = event.key.lower()
        if key == 'ctrl+z':
            self.on_undo(event)
        elif key in ('ctrl+shift+z', 'ctrl+y'):
            self.on_redo(event)
        elif key == 'tab':
            self.on_toggle_mode(event)
        elif key == 'd':
            self.on_toggle_draw_mode(event)
        elif key == 'escape':
            self.on_escape(event)
        elif key == 'delete':
            self.on_delete(event)
        elif key == 'enter':
            self.on_finalize_draw(event)
        elif key in '123456789':
            print("Lane ID selection disabled; use default ID 0 or Merge Lanes button")
            self.update_status("Lane ID selection disabled")

    def on_escape(self, event):
        if self.plot_manager is None:
            return
        self.selection_mode = False
        self.draw_mode = False
        self.id_set = True
        self.clear_smoothing_state()
        self.clear_merge_state()
        self.plot_manager.rs.set_active(False)
        self.curve_manager.draw_points = []
        if self.curve_manager.current_line:
            self.curve_manager.current_line.remove()
            self.curve_manager.current_line = None
            self.plot_manager.fig.canvas.draw_idle()
        if self.plot_manager.selected_indices:
            self.plot_manager.selected_indices = []
            self.update_point_sizes()
            print("Cleared selection")
        print("Entered add/delete mode")
        self.update_button_states()
        self.update_status("Entered add/delete mode")

    def on_delete(self, event):
        if self.plot_manager is None or not self.selection_mode or not self.plot_manager.selected_indices:
            return
        deleted_indices = self.plot_manager.selected_indices
        self.data_manager.delete_points(deleted_indices)
        self.plot_manager.selected_indices = []
        self.plot_manager.update_plot(self.data_manager.data)
        print(f"Deleted {len(deleted_indices)} points")
        self.update_status(f"Deleted {len(deleted_indices)} points")

    def on_undo(self, event):
        if self.plot_manager is None:
            return
        data, success = self.data_manager.undo()
        if success:
            self.plot_manager.selected_indices = []
            self.plot_manager.update_plot(data)
            print("Undo performed")
            self.update_status("Undo performed")
        else:
            self.update_status("Nothing to undo")

    def on_redo(self, event):
        if self.plot_manager is None:
            return
        data, success = self.data_manager.redo()
        if success:
            self.plot_manager.selected_indices = []
            self.plot_manager.update_plot(data)
            print("Redo performed")
            self.update_status("Redo performed")
        else:
            self.update_status("Nothing to redo")

    def on_finalize_draw(self, event):
        if not self.draw_mode:
            return
        self.curve_manager.finalize_draw(self.selected_id)
        self.plot_manager.selected_indices = []
        self.update_point_sizes()
        print(f"Finalized {'curve' if self.curve_manager.is_curve else 'line'} with ID {self.selected_id}")
        self.update_status("Drawing finalized")

    def update_status(self, message=""):
        try:
            self.plot_manager.update_status(message)
            self.last_status_time = time.time()
            if message:
                def clear_status():
                    if time.time() - self.last_status_time >= self.status_timeout:
                        self.plot_manager.update_status("")
                        self.fig.canvas.draw_idle()
                self.fig.canvas.manager.window.after(int(self.status_timeout * 1000), clear_status)
        except Exception as e:
            print(f"Error updating status: {e}")