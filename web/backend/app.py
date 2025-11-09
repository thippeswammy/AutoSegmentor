import os
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# --- Configuration ---
# The backend will look for lane data in a subdirectory of the project root.
DATA_DIRECTORY = 'lanes/TEMP'
# ---------------------

app = Flask(__name__)
CORS(app)

def load_data():
    base_path = os.getcwd()
    lanes_path = os.path.join(base_path, DATA_DIRECTORY)

    if not os.path.isdir(lanes_path):
        return None, None, None

    all_files = [f for f in os.listdir(lanes_path) if f.endswith('.npy')]
    # This is the default order from the original application
    custom_order = ["lane-0.npy", "lane-3.npy", "lane-2.npy", "lane-1.npy"]

    files = [f for f in custom_order if f in all_files]
    files += [f for f in all_files if f not in files]

    nodes_list = []
    edges_list = []
    file_names = []
    point_id_counter = 0

    for lane_idx, file in enumerate(files):
        file_path = os.path.join(lanes_path, file)
        try:
            points = np.load(file_path)
            if points.size == 0:
                continue

            if len(points.shape) == 1:
                points = points.reshape(-1, points.shape[0])

            if points.shape[1] < 2:
                continue

            N = points.shape[0]
            nodes = np.zeros((N, 5))
            edges = np.zeros((N - 1, 2), dtype=int)
            current_lane_point_ids = []

            nodes[:, 1:3] = points[:, 0:2]
            nodes[:, 4] = lane_idx

            for i in range(N):
                new_id = point_id_counter + i
                nodes[i, 0] = new_id
                current_lane_point_ids.append(new_id)

            for i in range(N - 1):
                edges[i, 0] = current_lane_point_ids[i]
                edges[i, 1] = current_lane_point_ids[i + 1]

            nodes_list.append(nodes)
            edges_list.append(edges)
            file_names.append(file)
            point_id_counter += N
        except Exception as e:
            print(f"Error loading file {file}: {e}")
            continue

    if not nodes_list:
        return np.array([]), np.array([]), []

    all_nodes = np.vstack(nodes_list)
    all_edges = np.vstack(edges_list)

    return all_nodes, all_edges, file_names

@app.route('/api/data')
def get_data():
    nodes, edges, file_names = load_data()
    if nodes is None:
        return jsonify({'error': f"Data directory '{DATA_DIRECTORY}' not found"}), 404

    return jsonify({
        'nodes': nodes.tolist(),
        'edges': edges.tolist(),
        'file_names': file_names
    })

@app.route('/api/save', methods=['POST'])
def save_data():
    data = request.json
    nodes = np.array(data['nodes'])
    edges = np.array(data['edges'])

    folder = "workspace-Temp"
    os.makedirs(folder, exist_ok=True)

    nodes_filename = os.path.join(folder, "graph_nodes.npy")
    edges_filename = os.path.join(folder, "graph_edges.npy")

    np.save(nodes_filename, nodes)
    np.save(edges_filename, edges)

    return jsonify({'message': 'Data saved successfully'})

if __name__ == '__main__':
    # Note: Debug mode is disabled for security.
    app.run(host='0.0.0.0', port=5000)
