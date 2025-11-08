import numpy as np
from scipy.interpolate import splprep, splev
from collections import deque

def _get_node_coords(nodes, point_id):
    node_mask = (nodes[:, 0] == point_id)
    if np.any(node_mask):
        return nodes[node_mask][0, 1:3]  # [x, y]
    return None

def _find_path(edges, start_id, end_id):
    if edges.size == 0:
        return None

    adj = {}
    for from_id, to_id in edges:
        from_id, to_id = int(from_id), int(to_id)
        adj.setdefault(from_id, []).append(to_id)
        adj.setdefault(to_id, []).append(from_id)

    if start_id not in adj:
        return None

    queue = deque([(start_id, [start_id])])
    visited = {start_id}

    while queue:
        current_id, path = queue.popleft()

        if current_id == end_id:
            return path

        for neighbor_id in adj.get(current_id, []):
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                new_path = path + [neighbor_id]
                queue.append((neighbor_id, new_path))

    return None

def smooth_path(nodes, edges, start_id, end_id, smoothness, smoothing_weight):
    path_ids = _find_path(edges, start_id, end_id)
    if not path_ids:
        return None

    if len(path_ids) < 2:
        return None

    points_xy = []
    for pid in path_ids:
        coords = _get_node_coords(nodes, pid)
        if coords is not None:
            points_xy.append(coords)
    points = np.array(points_xy)
    if len(points) < 2:
        return None

    original_start_point = points[0]
    original_end_point = points[-1]

    prev_point, next_point = None, None
    adj = {}
    for from_id, to_id in edges:
        from_id, to_id = int(from_id), int(to_id)
        adj.setdefault(from_id, []).append(to_id)
        adj.setdefault(to_id, []).append(from_id)

    if start_id in adj:
        for neighbor_id in adj[start_id]:
            if len(path_ids) > 1 and neighbor_id != path_ids[1]:
                prev_point = _get_node_coords(nodes, neighbor_id)
                break

    if end_id in adj:
        for neighbor_id in adj[end_id]:
            if len(path_ids) > 1 and neighbor_id != path_ids[-2]:
                next_point = _get_node_coords(nodes, neighbor_id)
                break

    fitting_points = points.copy()
    weights = np.ones(len(fitting_points)) * smoothing_weight

    segment_start_in_fitting = 0
    segment_end_in_fitting = len(fitting_points) - 1
    HIGH_WEIGHT = 100

    if prev_point is not None:
        fitting_points = np.vstack([prev_point, fitting_points])
        weights = np.concatenate(([1], weights))
        segment_start_in_fitting += 1
        segment_end_in_fitting += 1

    if next_point is not None:
        fitting_points = np.vstack([fitting_points, next_point])
        weights = np.concatenate((weights, [1]))

    weights[segment_start_in_fitting] = HIGH_WEIGHT
    weights[segment_end_in_fitting] = HIGH_WEIGHT

    try:
        x, y = fitting_points[:, 0], fitting_points[:, 1]

        if len(np.unique(x)) < 2 and len(np.unique(y)) < 2:
            return points

        distances = np.sqrt(np.sum(np.diff(fitting_points, axis=0) ** 2, axis=1))
        u = np.zeros(len(fitting_points))
        u[1:] = np.cumsum(distances)
        u = u / u[-1] if u[-1] > 0 else np.linspace(0, 1, len(fitting_points))

        smoothing_factor = len(points) * smoothness
        if smoothing_factor < 0.1: smoothing_factor = 0.1

        tck, u_fitted = splprep([x, y], u=u, s=smoothing_factor, k=3, w=weights)

        u_start_segment = u[segment_start_in_fitting]
        u_end_segment = u[segment_end_in_fitting]

        num_new_points = len(path_ids)
        u_fine = np.linspace(u_start_segment, u_end_segment, num_new_points)

        x_smooth, y_smooth = splev(u_fine, tck)
        new_points = np.stack((x_smooth, y_smooth), axis=1)

        new_points[0] = original_start_point
        new_points[-1] = original_end_point

        return new_points

    except ValueError as e:
        print(f"Spline fitting failed: {e}")
        return None
