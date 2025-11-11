import os
import sys
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# These imports assume your package structure
from DataVisualizationEditingTool.utils.data_loader import DataLoader
from DataVisualizationEditingTool.utils.data_manager import DataManager
from DataVisualizationEditingTool.utils.operations import smooth_path
from flask import request

app = Flask(__name__)
CORS(app)

# Global DataManager instance
DATA_MANAGER = None

def load_data_globally():
    global DATA_MANAGER
    # The script is in web/backend, so we need to go up two levels to the project root
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    lanes_path = os.path.join(base_path, 'lanes/TEMP')

    if not os.path.isdir(lanes_path):
        raise FileNotFoundError(f"Directory does not exist: {lanes_path}")

    custom_order = ["lane-0.npy", "lane-3.npy", "lane-2.npy", "lane-1.npy"]
    loader = DataLoader(lanes_path, file_order=custom_order)
    nodes, edges, file_names = loader.load_data()
    DATA_MANAGER = DataManager(nodes, edges, file_names)

@app.route('/api/data')
def get_data():
    if DATA_MANAGER is None:
        try:
            load_data_globally()
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 500

    nodes_list = DATA_MANAGER.nodes.tolist() if isinstance(DATA_MANAGER.nodes, np.ndarray) else []
    edges_list = DATA_MANAGER.edges.tolist() if isinstance(DATA_MANAGER.edges, np.ndarray) else []

    response_data = {
        "nodes": nodes_list,
        "edges": edges_list,
        "file_names": DATA_MANAGER.file_names
    }

    return jsonify(response_data)

@app.route('/api/smooth', methods=['POST'])
def smooth_data():
    if DATA_MANAGER is None:
        return jsonify({"error": "Data not loaded. Please call /api/data first."}), 500

    data = request.json
    start_id = data.get('start_id')
    end_id = data.get('end_id')
    smoothness = data.get('smoothness', 0.5)
    smoothing_weight = data.get('smoothing_weight', 50)

    if start_id is None or end_id is None:
        return jsonify({"error": "start_id and end_id are required."}), 400

    new_points = smooth_path(DATA_MANAGER.nodes, DATA_MANAGER.edges, start_id, end_id, smoothness, smoothing_weight)

    if new_points is None:
        return jsonify({"error": "Smoothing failed."}), 500

    # Here you would typically update the nodes in the DATA_MANAGER
    # For now, we just return the smoothed points
    return jsonify({"smoothed_points": new_points.tolist()})

@app.route('/api/save', methods=['POST'])
def save_data():
    if DATA_MANAGER is None:
        return jsonify({"error": "Data not loaded. Please call /api/data first."}), 500

    filename = DATA_MANAGER.save()
    if filename:
        return jsonify({"message": f"Data saved to {filename}"}), 200
    else:
        return jsonify({"error": "Save failed"}), 500

@app.route('/api/delete_node', methods=['POST'])
def delete_node():
    if DATA_MANAGER is None:
        return jsonify({"error": "Data not loaded. Please call /api/data first."}), 500

    data = request.json
    node_id = data.get('node_id')

    if node_id is None:
        return jsonify({"error": "node_id is required."}), 400

    DATA_MANAGER.delete_points([node_id])

    return jsonify({"message": f"Node {node_id} deleted."}), 200

@app.route('/api/connect_nodes', methods=['POST'])
def connect_nodes():
    if DATA_MANAGER is None:
        return jsonify({"error": "Data not loaded."}), 500

    data = request.json
    start_id = data.get('start_id')
    end_id = data.get('end_id')

    if start_id is None or end_id is None:
        return jsonify({"error": "start_id and end_id are required."}), 400

    DATA_MANAGER.add_edge(start_id, end_id)
    return jsonify({"message": f"Nodes {start_id} and {end_id} connected."}), 200

@app.route('/api/remove_between', methods=['POST'])
def remove_between():
    if DATA_MANAGER is None:
        return jsonify({"error": "Data not loaded."}), 500

    data = request.json
    start_id = data.get('start_id')
    end_id = data.get('end_id')

    if start_id is None or end_id is None:
        return jsonify({"error": "start_id and end_id are required."}), 400

    path_ids = DATA_MANAGER._find_path(start_id, end_id)

    if not path_ids or len(path_ids) < 2:
        return jsonify({"error": "No path found between the specified nodes."}), 404

    edges_to_delete = []
    for i in range(len(path_ids) - 1):
        edges_to_delete.append((path_ids[i], path_ids[i+1]))
        edges_to_delete.append((path_ids[i+1], path_ids[i])) # Also check for reverse direction

    DATA_MANAGER.remove_edges(edges_to_delete)
    return jsonify({"message": "Path removed successfully."}), 200

@app.route('/api/reverse_path', methods=['POST'])
def reverse_path():
    if DATA_MANAGER is None:
        return jsonify({"error": "Data not loaded."}), 500

    data = request.json
    start_id = data.get('start_id')
    end_id = data.get('end_id')

    if start_id is None or end_id is None:
        return jsonify({"error": "start_id and end_id are required."}), 400

    path_ids = DATA_MANAGER._find_path(start_id, end_id)

    if not path_ids or len(path_ids) < 2:
        return jsonify({"error": "No path found between the specified nodes."}), 404

    DATA_MANAGER.reverse_path(path_ids)
    return jsonify({"message": "Path reversed successfully."}), 200

@app.route('/api/draw_node', methods=['POST'])
def draw_node():
    if DATA_MANAGER is None:
        return jsonify({"error": "Data not loaded."}), 500

    data = request.json
    x = data.get('x')
    y = data.get('y')
    original_lane_id = data.get('original_lane_id')

    if x is None or y is None or original_lane_id is None:
        return jsonify({"error": "x, y, and original_lane_id are required."}), 400

    new_node_id = DATA_MANAGER.add_node(x, y, original_lane_id)
    return jsonify({"message": "Node drawn successfully.", "node_id": new_node_id}), 200

if __name__ == '__main__':
    app.run(debug=True)
