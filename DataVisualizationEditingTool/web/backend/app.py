from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import sys

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from DataVisualizationEditingTool.utils.data_manager import DataManager
from DataVisualizationEditingTool.utils.curve_manager import CurveManager

app = Flask(__name__)
CORS(app)

# Global Data State
# In a real multi-user app, this would be in a database or session-keyed.
# For a single-user local tool, a global variable is fine.
data_manager = None
curve_manager = None

def initialize_data():
    global data_manager, curve_manager
    # Mock initialization based on main.py logic
    # We will look for temp data or load a dummy set if nothing exists

    # For now, let's try to load the same data as main.py,
    # but since we don't have the exact paths, we'll initialize empty or look for files.
    # Simplified for migration proof-of-concept:

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    # Try to load temp data first
    # This pathing might need adjustment based on where you run the server from
    # nodes = np.load(os.path.join(base_path, 'lanes/TEMP/nodes.npy')) if os.path.exists(...) else np.array([])

    # Initialize with empty data if files don't exist
    nodes = np.array([])
    edges = np.array([])
    file_names = []

    # Try loading sample data if available (replicating main.py partial logic)
    # For the purpose of this task, we will assume empty start if no files are found,
    # or user can use an upload endpoint (out of scope, but good for future).

    # Let's create a dummy node just so the frontend has something to show if empty
    # nodes = np.array([[0, 10, 10, 0, 0], [1, 20, 20, 0, 0]])
    # edges = np.array([[0, 1]])
    # file_names = ["Lane 0"]

    print("Initializing DataManager...")
    data_manager = DataManager(nodes, edges, file_names)

    # We need a dummy plot_manager and event_handler because CurveManager expects them
    # We will create a Mock wrapper
    class MockPlotManager:
        def __init__(self):
            self.slider_smooth = type('obj', (object,), {'val': 1.0})
            self.slider_weight = type('obj', (object,), {'val': 20.0})
            self.fig = type('obj', (object,), {'canvas': type('obj', (object,), {'draw_idle': lambda: None})})
            self.ax = type('obj', (object,), {'plot': lambda *args, **kwargs: [None]})

    class MockEventHandler:
        def __init__(self):
            self.update_status = lambda msg: print(f"Status: {msg}")
            self.smoothing_preview_line = None
            self.smoothing_path_ids = []

    plot_manager = MockPlotManager()
    event_handler = MockEventHandler()

    curve_manager = CurveManager(data_manager, plot_manager, event_handler)

initialize_data()


@app.route('/api/data', methods=['GET'])
def get_data():
    if data_manager is None:
        return jsonify({"nodes": [], "edges": [], "filenames": []})

    nodes_list = data_manager.nodes.tolist() if data_manager.nodes.size > 0 else []
    edges_list = data_manager.edges.tolist() if data_manager.edges.size > 0 else []

    return jsonify({
        "nodes": nodes_list,
        "edges": edges_list,
        "filenames": data_manager.file_names
    })

@app.route('/api/nodes/add', methods=['POST'])
def add_node():
    data = request.json
    x = data.get('x')
    y = data.get('y')
    lane_id = data.get('lane_id', 0)

    new_id = data_manager.add_node(x, y, lane_id)

    # If a 'connected_from' ID is provided, create an edge
    connected_from = data.get('connected_from')
    if connected_from is not None:
        data_manager.add_edge(connected_from, new_id)

    return jsonify({"success": True, "new_id": int(new_id)})

@app.route('/api/nodes/delete', methods=['POST'])
def delete_node():
    data = request.json
    node_ids = data.get('node_ids', [])
    data_manager.delete_points(node_ids)
    return jsonify({"success": True})

@app.route('/api/edges/add', methods=['POST'])
def add_edge():
    data = request.json
    from_id = data.get('from_id')
    to_id = data.get('to_id')
    data_manager.add_edge(from_id, to_id)
    return jsonify({"success": True})

@app.route('/api/edges/delete_for_node', methods=['POST'])
def delete_edges_for_node():
    data = request.json
    node_id = data.get('node_id')
    data_manager.delete_edges_for_node(node_id)
    return jsonify({"success": True})

@app.route('/api/undo', methods=['POST'])
def undo():
    nodes, edges, success = data_manager.undo()
    return jsonify({"success": success})

@app.route('/api/redo', methods=['POST'])
def redo():
    nodes, edges, success = data_manager.redo()
    return jsonify({"success": success})

@app.route('/api/save', methods=['POST'])
def save():
    filename = data_manager.save()
    return jsonify({"success": True, "filename": filename})

# --- Complex Operations ---

@app.route('/api/smooth/preview', methods=['POST'])
def preview_smooth():
    data = request.json
    start_id = data.get('start_id')
    end_id = data.get('end_id')
    smoothness = data.get('smoothness', 1.0)
    weight = data.get('weight', 20)

    # Update mock sliders
    curve_manager.plot_manager.slider_smooth.val = smoothness
    curve_manager.smoothing_weight = weight # CurveManager uses internal var for weight

    # Find path
    path_ids = curve_manager._find_path(start_id, end_id)
    if not path_ids:
        return jsonify({"success": False, "message": "No path found"})

    # Calculate smooth points (preview=True)
    # Note: _smooth_segment returns a numpy array of [x, y]
    # We need to temporally store the path_ids so apply knows what to do
    curve_manager.event_handler.smoothing_path_ids = path_ids

    smoothed_points = curve_manager._smooth_segment(path_ids, preview=True)

    if smoothed_points is None:
        return jsonify({"success": False, "message": "Smoothing calculation failed"})

    return jsonify({
        "success": True,
        "points": smoothed_points.tolist()
    })

@app.route('/api/smooth/apply', methods=['POST'])
def apply_smooth():
    # We assume preview was called and 'smoothing_path_ids' is set in the mock event handler.
    # In a stateless API, we should re-pass the IDs or store session state.
    # For safety, let's accept start/end again or path_ids.
    # But relying on server-state for a "transaction" (Preview -> Apply) is risky without sessions.
    # For this prototype, we'll re-calculate or use the stored global if available.

    if not curve_manager.event_handler.smoothing_path_ids:
        return jsonify({"success": False, "message": "No smooth preview active"})

    curve_manager.apply_smooth()
    return jsonify({"success": True})

@app.route('/api/remove_between', methods=['POST'])
def remove_between():
    data = request.json
    start_id = data.get('start_id')
    end_id = data.get('end_id')

    # Manually implement logic from EventHandler.finalize_remove_between
    path_ids = curve_manager._find_path(start_id, end_id)
    if not path_ids or len(path_ids) < 2:
        return jsonify({"success": False, "message": "No path found"})

    points_to_delete = path_ids[1:-1]
    if points_to_delete:
        data_manager.delete_points(points_to_delete)

    return jsonify({"success": True, "deleted_count": len(points_to_delete)})

@app.route('/api/reverse_path', methods=['POST'])
def reverse_path():
    data = request.json
    start_id = data.get('start_id')
    end_id = data.get('end_id')

    path_ids = curve_manager._find_path(start_id, end_id)
    if not path_ids or len(path_ids) < 2:
        return jsonify({"success": False, "message": "No path found"})

    data_manager.reverse_path(path_ids)
    return jsonify({"success": True})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
