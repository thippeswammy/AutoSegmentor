import os
import sys
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    base_path = os.getcwd()
    lanes_path = os.path.join(base_path, 'backup_lanes')

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

if __name__ == '__main__':
    app.run(debug=True)
