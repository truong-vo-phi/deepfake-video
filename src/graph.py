import numpy as np


def _safe_index_pairs(edges, num_nodes: int):
    valid = []
    for i, j in edges:
        if 0 <= i < num_nodes and 0 <= j < num_nodes:
            valid.append((i, j))
    return valid


def get_face_edges(num_nodes: int = 468):
    try:
        import mediapipe as mp

        edges = set()
        for a, b in mp.solutions.face_mesh.FACEMESH_TESSELATION:
            edges.add((int(a), int(b)))
            edges.add((int(b), int(a)))
        return _safe_index_pairs(edges, num_nodes)
    except Exception:
        # Fallback graph if MediaPipe is unavailable.
        edges = []
        for i in range(num_nodes - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))
        return edges


def build_adjacency(num_nodes: int = 468):
    a = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in get_face_edges(num_nodes=num_nodes):
        a[i, j] = 1.0
    return a


def normalize_adjacency(a: np.ndarray):
    i = np.eye(a.shape[0], dtype=np.float32)
    a_tilde = a + i
    d = np.sum(a_tilde, axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-6)))
    return d_inv_sqrt @ a_tilde @ d_inv_sqrt
