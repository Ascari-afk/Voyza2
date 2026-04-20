"""
src/routing_engine.py  (v5 — Penalty-Based Diverse K-Shortest Paths)
======================================================================

แทนที่ Yen's algorithm (nx.shortest_simple_paths) ด้วย
Penalty-Based approach:
  1. หา Route 1 = shortest path ปกติ
  2. Penalize ทุก edge ที่ Route 1 ใช้ (คูณ PENALTY_FACTOR)
  3. หา Route 2 บน graph ที่ penalized แล้ว → บังคับให้ใช้ถนนต่างกัน
  4. Penalize edges ของ Route 1 + Route 2 รวมกัน
  5. หา Route 3 → บังคับต่างจากทั้งสองเส้นก่อนหน้า
  6. Cost ที่รายงานคือ cost จริงบน original graph (ไม่รวม penalty)

PENALTY_FACTOR = 10.0 → edge ที่เคยใช้มีราคาแพงขึ้น 10x
  ปรับได้ใน config หรือส่งเป็น argument

ข้อดีเทียบ Yen's:
  - ได้ path ที่ต่างกันทางกายภาพ ไม่ใช่แค่ต่างนิดเดียว
  - เร็วกว่า Yen's O(kn(m + n log n)) สำหรับ k เล็กๆ
  - ง่ายต่อการปรับ diversity
"""

from typing import Any, Dict, List, Set, Tuple

import numpy as np
import networkx as nx
from sklearn.neighbors import BallTree

# ---------------------------------------------------------------
# Tunable parameter
# ---------------------------------------------------------------
PENALTY_FACTOR = 10.0   # edge ที่เคยใช้จะแพงขึ้น 10x


# ---------------------------------------------------------------
# 1) Spatial index
# ---------------------------------------------------------------

def build_node_spatial_index(G: nx.Graph) -> Dict[str, Any]:
    """
    BallTree (haversine) สำหรับ nearest-node lookup
    รองรับ node เป็น int osmid หรือ tuple (lon, lat)
    """
    node_ids: list = []
    coords:   list = []

    for n, data in G.nodes(data=True):
        lat = data.get("lat")
        lon = data.get("lon")
        # fallback: node = tuple(lon, lat)
        if (lat is None or lon is None) and isinstance(n, tuple) and len(n) == 2:
            lon, lat = float(n[0]), float(n[1])
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

    node_ids_arr = np.empty(len(node_ids), dtype=object)
    for i, nid in enumerate(node_ids):
        node_ids_arr[i] = nid

    return {
        "tree":       tree,
        "node_ids":   node_ids_arr,
        "coords_rad": coords_rad,
    }


def nearest_node_from_latlon(index: Dict[str, Any], lat: float, lon: float) -> Any:
    query = np.deg2rad([[lat, lon]])
    dist, ind = index["tree"].query(query, k=1)
    return index["node_ids"][ind[0, 0]]


# ---------------------------------------------------------------
# 2) Cost helpers
# ---------------------------------------------------------------

def compute_route_cost(G: nx.Graph, path: List[Any], weight: str = "length") -> float:
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v, default={})
        total += data.get(weight, 0.0)
    return total


def _edges_of_path(path: List[Any]) -> Set[Tuple[Any, Any]]:
    """คืน set ของ frozenset edge จาก path (undirected)"""
    return {frozenset((u, v)) for u, v in zip(path[:-1], path[1:])}


# ---------------------------------------------------------------
# 3) Penalty-Based Diverse K-Shortest Paths
# ---------------------------------------------------------------

def _penalize_graph(
    G: nx.Graph,
    used_edges: Set[frozenset],
    weight: str = "length",
    factor: float = PENALTY_FACTOR,
) -> nx.Graph:
    """
    คืน graph ใหม่ที่ edge ใน used_edges ถูกคูณ weight ด้วย factor
    ไม่แก้ไข G ต้นฉบับ
    """
    H = G.copy()
    for u, v, data in H.edges(data=True):
        key = frozenset((u, v))
        if key in used_edges:
            H[u][v][weight] = data.get(weight, 0.0) * factor
    return H


def penalty_based_k_paths(
    G: nx.Graph,
    source: Any,
    target: Any,
    k: int = 3,
    weight: str = "length",
    factor: float = PENALTY_FACTOR,
) -> List[Dict[str, Any]]:
    """
    Penalty-Based Diverse K-Shortest Paths

    Algorithm:
      สำหรับ i = 1..k:
        1. หา shortest path บน current graph H
        2. คำนวณ cost จริงบน G (ไม่รวม penalty)
        3. เพิ่ม edges ของ path นี้เข้า used_edges
        4. Penalize used_edges บน H สำหรับ route ถัดไป

    คืน list of dicts: {"path", "cost", "edges"}
    cost = ระยะทางจริง (เมตร) ไม่รวม penalty
    """
    results:    List[Dict[str, Any]] = []
    used_edges: Set[frozenset]       = set()
    H = G  # เริ่มต้นจาก original graph

    for _ in range(k):
        try:
            path = nx.shortest_path(H, source=source, target=target, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            break

        # cost จริงบน original graph
        real_cost = compute_route_cost(G, path, weight=weight)

        results.append({
            "path":  path,
            "cost":  real_cost,
            "edges": list(zip(path[:-1], path[1:])),
        })

        # เพิ่ม edges ของ route นี้เข้า penalty set
        new_edges = _edges_of_path(path)
        used_edges |= new_edges

        # สร้าง penalized graph สำหรับ route ถัดไป
        H = _penalize_graph(G, used_edges, weight=weight, factor=factor)

    return results


# ---------------------------------------------------------------
# 4) get_routes_for_od (backward compat wrapper)
# ---------------------------------------------------------------

def get_routes_for_od(
    G: nx.Graph,
    src_node: Any,
    dst_node: Any,
    k: int = 3,
    weight: str = "length",
) -> List[Dict[str, Any]]:
    return penalty_based_k_paths(G, src_node, dst_node, k=k, weight=weight)


# ---------------------------------------------------------------
# 5) Main entry point ที่ dashboard เรียก
# ---------------------------------------------------------------

def get_routes_from_latlon(
    G: nx.Graph,
    node_index: Dict[str, Any],
    src_lat: float,
    src_lon: float,
    dst_lat: float,
    dst_lon: float,
    k: int = 3,
    weight: str = "length",
) -> Dict[str, Any]:
    """
    รับ lat/lon → snap → Penalty-Based K=3 diverse shortest paths
    คืน:
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