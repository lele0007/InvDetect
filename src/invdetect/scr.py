from __future__ import annotations

import numpy as np


def unary_costs(anomaly_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equation (3): cross-entropy costs for normal and abnormal labels."""
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    if not np.isfinite(anomaly_map).all():
        raise ValueError("Anomaly map contains NaN or infinite values.")
    normal_cost = np.logaddexp(0.0, anomaly_map).astype(np.float32)
    abnormal_cost = np.logaddexp(0.0, -anomaly_map).astype(np.float32)
    return normal_cost, abnormal_cost


def spatial_contiguity_refinement(
    anomaly_map: np.ndarray,
    horizontal_weights: np.ndarray,
    vertical_weights: np.ndarray,
) -> np.ndarray:
    """Solve Equations (2)-(4) using an s-t max-flow/min-cut graph."""
    try:
        import maxflow
    except ImportError as exc:  
        raise ImportError("Spatial refinement requires PyMaxflow: pip install PyMaxflow") from exc

    normal_cost, abnormal_cost = unary_costs(anomaly_map)
    height, width = normal_cost.shape
    if horizontal_weights.shape != (height, max(width - 1, 0)):
        raise ValueError("Horizontal edge weights have the wrong shape.")
    if vertical_weights.shape != (max(height - 1, 0), width):
        raise ValueError("Vertical edge weights have the wrong shape.")
    if np.any(horizontal_weights < 0) or np.any(vertical_weights < 0):
        raise ValueError("Pairwise weights must be non-negative.")

    graph = maxflow.Graph[float]()
    nodes = graph.add_grid_nodes((height, width))

    graph.add_grid_tedges(nodes, abnormal_cost, normal_cost)

    for y in range(height):
        for x in range(width - 1):
            weight = float(horizontal_weights[y, x])
            if weight > 0:
                graph.add_edge(nodes[y, x], nodes[y, x + 1], weight, weight)
    for y in range(height - 1):
        for x in range(width):
            weight = float(vertical_weights[y, x])
            if weight > 0:
                graph.add_edge(nodes[y, x], nodes[y + 1, x], weight, weight)

    graph.maxflow()
    return graph.get_grid_segments(nodes).astype(np.uint8)

