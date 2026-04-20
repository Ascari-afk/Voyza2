"""
src/routing_engine.py  (patched)
- build_node_spatial_index : ใช้ sklearn BallTree (haversine) — ถูกต้องแล้ว
- get_routes_from_latlon   : signature ปรับให้ตรงกับ dashboard_app2.py
                             (รับ node_index dict ตัวเดียว ไม่แยก tree/node_ids)
- K=3 fixed, weight="length" fixed ตามที่ dashboard ต้องการ
"""
from typing import Any, Dict, List

import numpy as np
import networkx as nx
from sklearn.neighbors import BallTree


# ---------- 1) Spatial index ----------

def build_node_spatial_index(G: nx.Graph) -> Dict[str, Any]:
    """
    เตรียม BallTree สำหรับค้นหา node ใกล้สุดจากพิกัด lat/lon
    Node ใน G ต้องมี attribute 'lat' และ 'lon' (WGS84)
    ✅ ถูกต้องแล้ว — ใช้ haversine บน radian
    """
    node_ids: list = []
    coords: list = []

    for n, data in G.nodes(data=True):
        lat = data.get("lat")
        lon = data.get("lon")
        if lat is not None and lon is not None:
            node_ids.append(n)
            coords.append([lat, lon])

    if not coords:
        raise ValueError(
            "Graph has no nodes with 'lat'/'lon'. "
            "ตรวจสอบว่า build_road_graph_from_roads_gdf() เก็บ lat/lon ไว้ใน node attribute"
        )

    coords_arr = np.array(coords, dtype=np.float64)
    coords_rad = np.deg2rad(coords_arr)
    tree = BallTree(coords_rad, metric="haversine")

    return {
        "tree": tree,
        "node_ids": np.array(node_ids),
        "coords_rad": coords_rad,
    }


def nearest_node_from_latlon(
    index: Dict[str, Any],
    lat: float,
    lon: float,
) -> Any:
    """หา node ใกล้สุดจาก (lat, lon) — คืน node id"""
    query = np.deg2rad([[lat, lon]])
    dist, ind = index["tree"].query(query, k=1)
    return index["node_ids"][ind[0, 0]]


# ---------- 2) K shortest paths ----------

def k_shortest_paths(
    G: nx.Graph,
    source: Any,
    target: Any,
    k: int = 3,
    weight: str = "length",
) -> List[List[Any]]:
    """
    คืน list ของ path สั้นสุด k เส้น
    ใช้ nx.shortest_simple_paths (Yen's algorithm ใน NetworkX)
    """
    try:
        gen = nx.shortest_simple_paths(G, source=source, target=target, weight=weight)
        paths = []
        for i, p in enumerate(gen):
            if i >= k:
                break
            paths.append(p)
        return paths
    except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
        return []


def compute_route_cost(G: nx.Graph, path: List[Any], weight: str = "length") -> float:
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v, default={})
        total += data.get(weight, 0.0)
    return total


def get_routes_for_od(
    G: nx.Graph,
    src_node: Any,
    dst_node: Any,
    k: int = 3,
    weight: str = "length",
) -> List[Dict[str, Any]]:
    paths = k_shortest_paths(G, src_node, dst_node, k=k, weight=weight)
    return [
        {
            "path": p,
            "cost": compute_route_cost(G, p, weight=weight),
            "edges": list(zip(p[:-1], p[1:])),
        }
        for p in paths
    ]


# ---------- 3) Main entry point ที่ dashboard เรียก ----------

def get_routes_from_latlon(
    G: nx.Graph,
    node_index: Dict[str, Any],
    src_lat: float,
    src_lon: float,
    dst_lat: float,
    dst_lon: float,
    k: int = 3,           # ✅ default = 3 ตาม requirement
    weight: str = "length",  # ✅ fixed weight
) -> Dict[str, Any]:
    """
    รับ lat/lon ต้นทาง–ปลายทาง → snap → K=3 shortest paths
    คืน dict ที่ dashboard ใช้:
    {
        "src_node": ...,
        "dst_node": ...,
        "routes": [{"path": [...], "cost": float, "edges": [...]}, ...]
    }
    """
    src_node = nearest_node_from_latlon(node_index, src_lat, src_lon)
    dst_node = nearest_node_from_latlon(node_index, dst_lat, dst_lon)
    routes   = get_routes_for_od(G, src_node, dst_node, k=k, weight=weight)

    return {
        "src_node": src_node,
        "dst_node": dst_node,
        "routes":   routes,
    }
