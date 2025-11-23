import os

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Construct path to the data directory relative to this script's location
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.abspath(os.path.join(APP_ROOT, '..', '..', 'lanes', 'TEMP'))
print("DATA_DIRECTORY_DATA_DIRECTORY", DATA_DIRECTORY)
import json


def load_data():
    """
    Loads lane data from .npy files and edge data from .json files.
    """
    nodes_list = []
    edges_list = []
    file_names = []

    custom_order = ["lane-0.npy", "lane-3.npy", "lane-2.npy", "lane-1.npy"]

    node_id_offset = 0
    for filename in custom_order:
        if filename.endswith('.npy'):
            file_path = os.path.join(DATA_DIRECTORY, filename)
            if os.path.exists(file_path):
                lane_data = np.load(file_path)
                num_points = lane_data.shape[0]
                lane_id = len(file_names)

                point_ids = np.arange(node_id_offset, node_id_offset + num_points).reshape(-1, 1)
                lane_ids = np.full((num_points, 1), lane_id)

                nodes = np.hstack([point_ids, lane_data, lane_ids])
                nodes_list.append(nodes)
                file_names.append(filename)

                # Load corresponding edges if they exist
                edge_filename = filename.replace('.npy', '_edges.json')
                edge_file_path = os.path.join(DATA_DIRECTORY, edge_filename)
                if os.path.exists(edge_file_path):
                    with open(edge_file_path, 'r') as f:
                        edges_for_lane = json.load(f)
                        edges_list.extend(edges_for_lane)
                else:
                    # Default edges
                    for i in range(num_points - 1):
                        edges_list.append([
                            int(point_ids[i][0]),
                            int(point_ids[i + 1][0])
                        ])

                node_id_offset += num_points

    if not nodes_list:
        return np.array([]), np.array([]), []

    nodes = np.vstack(nodes_list)
    return nodes, np.array(edges_list), file_names


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
    if not data or 'nodes' not in data or 'edges' not in data:
        return jsonify({"error": "Invalid data"}), 400

    nodes = np.array(data['nodes'])
    edges = data['edges']

    lanes_to_save = {}
    node_id_to_lane_id = {}
    for node in nodes:
        lane_id = int(node[4])
        node_id = int(node[0])
        node_id_to_lane_id[node_id] = lane_id
        if lane_id not in lanes_to_save:
            lanes_to_save[lane_id] = []
        lanes_to_save[lane_id].append(node[1:4])

    custom_order = ["lane-0.npy", "lane-3.npy", "lane-2.npy", "lane-1.npy"]

    try:
        active_lane_ids = set(lanes_to_save.keys())

        for lane_id, filename in enumerate(custom_order):
            file_path = os.path.join(DATA_DIRECTORY, filename)
            edge_filename = filename.replace('.npy', '_edges.json')
            edge_file_path = os.path.join(DATA_DIRECTORY, edge_filename)

            if lane_id in active_lane_ids:
                points = lanes_to_save[lane_id]
                lane_data = np.array(points)
                np.save(file_path, lane_data)

                # Save edges for this lane
                edges_for_lane = [
                    edge for edge in edges
                    if node_id_to_lane_id.get(edge[0]) == lane_id and node_id_to_lane_id.get(edge[1]) == lane_id
                ]
                with open(edge_file_path, 'w') as f:
                    json.dump(edges_for_lane, f)

            else:
                if os.path.exists(file_path):
                    os.remove(file_path)
                if os.path.exists(edge_file_path):
                    os.remove(edge_file_path)

        return jsonify({"message": "Data saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
