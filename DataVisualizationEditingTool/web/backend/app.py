import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# Construct path to the data directory relative to this script's location
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.abspath(os.path.join(APP_ROOT, '..', '..', 'lanes', 'TEMP'))

def load_data():
    """
    Loads lane data from .npy files in the specified directory.
    This is a simplified version of the DataLoader in the original application.
    """
    nodes_list = []
    file_names = []

    # A specific file order is mentioned in the original main.py
    custom_order = ["lane-0.npy", "lane-3.npy", "lane-2.npy", "lane-1.npy"]

    for filename in custom_order:
        if filename.endswith('.npy'):
            file_path = os.path.join(DATA_DIRECTORY, filename)
            if os.path.exists(file_path):
                lane_data = np.load(file_path)
                # Assuming lane_data is structured as [x, y, yaw]
                # We need to add point_id and original_lane_id
                num_points = lane_data.shape[0]
                lane_id = len(file_names)
                point_ids = np.arange(len(nodes_list), len(nodes_list) + num_points).reshape(-1, 1)
                lane_ids = np.full((num_points, 1), lane_id)

                # Create nodes: [point_id, x, y, yaw, original_lane_id]
                nodes = np.hstack([point_ids, lane_data, lane_ids])
                nodes_list.append(nodes)
                file_names.append(filename)

    if not nodes_list:
        return np.array([]), np.array([]), []

    nodes = np.vstack(nodes_list)

    # Create edges based on point proximity (a simple approach)
    edges = []
    for i in range(len(nodes) - 1):
        # Connect consecutive points within the same lane
        if nodes[i, 4] == nodes[i+1, 4]:
             edges.append([int(nodes[i, 0]), int(nodes[i+1, 0])])

    return nodes, np.array(edges), file_names

@app.route('/api/data')
def get_data():
    nodes, edges, file_names = load_data()

    if nodes.size == 0:
        return jsonify({"error": "No data found"}), 404

    # Convert numpy arrays to lists for JSON serialization
    nodes_list = nodes.tolist()
    edges_list = edges.tolist()

    return jsonify({
        "nodes": nodes_list,
        "edges": edges_list,
        "file_names": file_names
    })

@app.route('/api/save', methods=['POST'])
def save_data():
    data = request.get_json()
    if not data or 'nodes' not in data:
        return jsonify({"error": "Invalid data"}), 400

    nodes = np.array(data['nodes'])

    lanes_to_save = {}
    for node in nodes:
        lane_id = int(node[4])
        if lane_id not in lanes_to_save:
            lanes_to_save[lane_id] = []
        lanes_to_save[lane_id].append(node[1:4])

    custom_order = ["lane-0.npy", "lane-3.npy", "lane-2.npy", "lane-1.npy"]

    try:
        for lane_id, points in lanes_to_save.items():
            if lane_id < len(custom_order):
                filename = custom_order[lane_id]
                file_path = os.path.join(DATA_DIRECTORY, filename)

                lane_data = np.array(points)

                np.save(file_path, lane_data)
        return jsonify({"message": "Data saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
