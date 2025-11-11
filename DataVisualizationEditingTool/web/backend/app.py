from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import os
import sys

# This is to ensure the script can find the utility modules
# It assumes the script is run from the root of the repository
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))


from DataVisualizationEditingTool.utils.data_loader import DataLoader
from DataVisualizationEditingTool.utils.data_manager import DataManager
from DataVisualizationEditingTool.utils.curve_manager import CurveManager

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Globals to hold our data and managers
data_manager = None
curve_manager = None

def initialize_app():
    """Load initial data and initialize managers."""
    global data_manager, curve_manager

    # Construct a path to the data files relative to this script's location
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    lanes_path = os.path.join(base_path, 'DataVisualizationEditingTool/lanes/TEMP')

    if not os.path.isdir(lanes_path):
        print(f"Error: Data directory not found at {lanes_path}")
        # Initialize with empty data to allow the app to run
        nodes, edges, file_names = np.array([]), np.array([]), []
    else:
        loader = DataLoader(lanes_path)
        nodes, edges, file_names = loader.load_data()

    data_manager = DataManager(nodes, edges, file_names)

    # The original CurveManager is tightly coupled with Matplotlib's PlotManager.
    # We will need to refactor it or create a mock for the backend.
    # For now, we will work around it where necessary.
    # curve_manager = CurveManager(data_manager, None, None)

    print("Backend application initialized.")

@app.route('/api/data', methods=['GET'])
def get_data():
    """Endpoint to get all current node and edge data."""
    if data_manager is None:
        return jsonify({"error": "Data manager not initialized"}), 500

    nodes_list = data_manager.nodes.tolist() if data_manager.nodes.size > 0 else []
    edges_list = data_manager.edges.tolist() if data_manager.edges.size > 0 else []

    return jsonify({
        "nodes": nodes_list,
        "edges": edges_list,
        "file_names": data_manager.file_names
    })

@app.route('/api/add_node', methods=['POST'])
def add_node():
    data = request.json
    x = data.get('x')
    y = data.get('y')
    original_lane_id = data.get('original_lane_id')
    new_node_id = data_manager.add_node(x, y, original_lane_id)
    return jsonify({"message": "Node added successfully", "node_id": new_node_id})

@app.route('/api/add_edge', methods=['POST'])
def add_edge():
    data = request.json
    from_id = data.get('from_id')
    to_id = data.get('to_id')
    data_manager.add_edge(from_id, to_id)
    return jsonify({"message": "Edge added successfully"})

@app.route('/api/delete_nodes', methods=['POST'])
def delete_nodes():
    data = request.json
    node_ids = data.get('node_ids')
    data_manager.delete_points(node_ids)
    return jsonify({"message": "Nodes deleted successfully"})

@app.route('/api/undo', methods=['POST'])
def undo():
    nodes, edges, success = data_manager.undo()
    return jsonify({
        "success": success,
        "nodes": nodes.tolist(),
        "edges": edges.tolist()
    })

@app.route('/api/redo', methods=['POST'])
def redo():
    nodes, edges, success = data_manager.redo()
    return jsonify({
        "success": success,
        "nodes": nodes.tolist(),
        "edges": edges.tolist()
    })

@app.route('/api/save', methods=['POST'])
def save():
    """Saves the current nodes and edges to .npy files."""
    try:
        # Save nodes and edges for each lane
        unique_lane_ids = np.unique(data_manager.nodes[:, 4].astype(int))

        # Create a directory to save the files
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'lanes', 'TEMP_SAVED')
        os.makedirs(output_dir, exist_ok=True)

        for lane_id in unique_lane_ids:
            # Filter nodes by lane_id
            lane_nodes = data_manager.nodes[data_manager.nodes[:, 4] == lane_id]

            # Save nodes to .npy file
            file_name = f'lane-{lane_id}.npy'
            file_path = os.path.join(output_dir, file_name)
            np.save(file_path, lane_nodes)

        # Save all edges to a single file
        edges_path = os.path.join(output_dir, 'all_edges.json')
        edges_list = data_manager.edges.tolist()
        import json
        with open(edges_path, 'w') as f:
            json.dump(edges_list, f)

        return jsonify({"message": "Data saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    initialize_app()
    app.run(debug=True, port=5000)
