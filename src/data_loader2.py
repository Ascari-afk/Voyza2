"""
src/data_loader2.py  (OSMnx edition — v4)
==========================================
✅ FIX v4 — Nearest node snap ผิดตำแหน่ง:

Root cause: _spatial_sample(roads, 50000) ตัด roads ในพื้นที่สยาม/สุวรรณภูมิออก
→ node ที่ nearest neighbor หาได้อยู่คนละพื้นที่ (ปทุมธานี/รังสิต)

แก้โดยแยก 2 ชุดข้อมูล:
- roads_display : spatial sample 50K → ใช้แค่ Tab 1 Folium map
- roads_full    : โหลดทั้งหมด 366K → ใช้ build graph เท่านั้น

ผลคือ graph มี node ครอบคลุมทุกพื้นที่กรุงเทพ
→ nearest node snap ถูกต้อง → route map แสดงถูกที่
"""

import math
import numpy as np
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, MultiLineString, Point

from .config import (
    ROADS_PATH,
    BUILDINGS_PATH,
    ROAD_BUFFER_METERS,
    MAX_ROADS_ROWS,
    MAX_BUILDINGS_ROWS,
)

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROAD_NODES_PATH = os.path.join(_ROOT, "data", "raw", "road_nodes.geojson")


# ---------------------------------------------------------------
# Haversine helper (degree → meters)
# ---------------------------------------------------------------

def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _linestring_length_m(coords) -> float:
    total = 0.0
    pts = list(coords)
    for i in range(len(pts) - 1):
        total += _haversine_m(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
    return total


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _to_4326(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf
    if gdf.crs.to_epsg() != 4326:
        return gdf.to_crs(epsg=4326)
    return gdf


def _drop_invalid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    mask = (
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
        & gdf.geometry.is_valid
    )
    n_drop = (~mask).sum()
    if n_drop:
        print(f"  [data_loader] dropped {n_drop} invalid geometries")
    return gdf[mask].copy()


def _spatial_sample(gdf: gpd.GeoDataFrame, max_rows: int, seed: int = 42) -> gpd.GeoDataFrame:
    """
    สุ่มแบบ spatial-aware (10×10 grid)
    max_rows=0 → คืนทั้งหมด
    """
    if max_rows <= 0 or len(gdf) <= max_rows:
        return gdf

    try:
        proj = gdf.to_crs(epsg=32647)
        cx   = proj.geometry.centroid.x
        cy   = proj.geometry.centroid.y

        n_bins  = 10
        x_edges = np.linspace(cx.min(), cx.max(), n_bins + 1)
        y_edges = np.linspace(cy.min(), cy.max(), n_bins + 1)

        gdf = gdf.copy()
        gdf["_gx"] = np.clip(np.digitize(cx, x_edges) - 1, 0, n_bins - 1)
        gdf["_gy"] = np.clip(np.digitize(cy, y_edges) - 1, 0, n_bins - 1)

        per_cell = max(1, max_rows // (n_bins * n_bins))
        sampled = (
            gdf.groupby(["_gx", "_gy"], group_keys=False)
               .apply(lambda g: g.sample(min(len(g), per_cell), random_state=seed))
        )
        if len(sampled) < max_rows:
            remaining = gdf.drop(sampled.index)
            need  = max_rows - len(sampled)
            extra = remaining.sample(min(need, len(remaining)), random_state=seed)
            sampled = gpd.GeoDataFrame(
                gpd.pd.concat([sampled, extra]), crs=gdf.crs
            )
        return sampled.drop(columns=["_gx", "_gy"]).reset_index(drop=True)

    except Exception as e:
        print(f"  [data_loader] spatial_sample fallback to random: {e}")
        return gdf.sample(n=min(max_rows, len(gdf)), random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------
# ✅ NEW v4: load_roads_buildings_split
# คืน (roads_display, roads_full, buildings)
# - roads_display = sample MAX_ROADS_ROWS → ใช้ Tab 1 map
# - roads_full    = ทั้งหมด 366K → ใช้ build graph (node ครอบคลุมทั่วกรุงเทพ)
# ---------------------------------------------------------------

def load_roads_buildings_split():
    """
    โหลด roads 2 ชุด:
    - roads_display: spatial sample สำหรับ Folium map (เร็ว)
    - roads_full:    ทั้งหมด สำหรับ build graph (ครอบคลุมทุกพื้นที่)
    """
    print("  [data_loader] reading roads from", ROADS_PATH)
    roads_raw = gpd.read_file(ROADS_PATH)
    print(f"  [data_loader] roads loaded: {len(roads_raw):,} | CRS: {roads_raw.crs}")

    print("  [data_loader] reading buildings from", BUILDINGS_PATH)
    buildings = gpd.read_file(BUILDINGS_PATH)
    print(f"  [data_loader] buildings loaded: {len(buildings):,} | CRS: {buildings.crs}")

    # กรอง invalid + แปลง CRS
    roads_raw  = _drop_invalid(roads_raw)
    buildings  = _drop_invalid(buildings)
    roads_raw  = _to_4326(roads_raw)
    buildings  = _to_4326(buildings)

    # ---- roads_display (สำหรับ Tab 1 map) ----
    roads_display = _spatial_sample(roads_raw, MAX_ROADS_ROWS)
    print(f"  [data_loader] roads_display (map): {len(roads_display):,}")

    # ---- roads_full (สำหรับ build graph) ----
    # ✅ ไม่ sample — ใช้ทั้งหมดเพื่อให้ node ครอบคลุมทุกพื้นที่
    roads_full = roads_raw.copy()
    print(f"  [data_loader] roads_full (graph): {len(roads_full):,}")

    # clip buildings ใน bbox ของ roads_display (แสดงเฉพาะในพื้นที่ map)
    xmin, ymin, xmax, ymax = roads_display.total_bounds
    if (xmax - xmin) > 1e-8 and (ymax - ymin) > 1e-8:
        buildings = buildings.cx[xmin:xmax, ymin:ymax]

    buildings = _spatial_sample(buildings, MAX_BUILDINGS_ROWS)
    print(f"  [data_loader] buildings after sample: {len(buildings):,}")

    return roads_display, roads_full, buildings


# ---------------------------------------------------------------
# load_roads_buildings (backward compat — ใช้ roads_display)
# ---------------------------------------------------------------

def load_roads_buildings():
    roads_display, _, buildings = load_roads_buildings_split()
    return roads_display, buildings


# ---------------------------------------------------------------
# add_building_features_to_roads
# ---------------------------------------------------------------

def add_building_features_to_roads(roads_gdf, buildings_gdf):
    """STRtree bulk query — เร็วกว่า sjoin ~8-10x"""
    from shapely.strtree import STRtree

    print("  [features] projecting to EPSG:32647...")
    roads_proj     = roads_gdf.to_crs(epsg=32647)
    buildings_proj = buildings_gdf.to_crs(epsg=32647)

    roads_proj     = _drop_invalid(roads_proj).reset_index(drop=True)
    buildings_proj = _drop_invalid(buildings_proj).reset_index(drop=True)

    print(f"  [features] STRtree on {len(buildings_proj):,} building centroids...")
    b_centroids = list(buildings_proj.geometry.centroid)
    tree = STRtree(b_centroids)

    col_name = f"num_buildings_{ROAD_BUFFER_METERS}m"
    road_geoms = list(roads_proj.geometry)
    print(f"  [features] querying {len(road_geoms):,} road buffers (buffer={ROAD_BUFFER_METERS}m)...")

    counts = [len(tree.query(g.buffer(ROAD_BUFFER_METERS), predicate="intersects"))
              for g in road_geoms]

    roads_proj[col_name] = counts
    print(f"  [features] '{col_name}' added | total matches: {sum(counts):,}")
    return roads_proj.to_crs(roads_gdf.crs)


# ---------------------------------------------------------------
# build_road_graph_from_roads_gdf
# ---------------------------------------------------------------

def build_road_graph_from_roads_gdf(roads_gdf, road_nodes_gdf=None):
    """
    สร้าง NetworkX Graph
    ✅ v4: รับ roads_gdf ที่ไม่ได้ sample (366K) → node ครอบคลุมทุกพื้นที่
    col_name (num_buildings) ไม่จำเป็นต้องมี — จะ default เป็น 0
    """
    print("  [graph] building NetworkX graph...")
    G = nx.Graph()

    roads_4326 = _to_4326(roads_gdf).reset_index(drop=True)

    col_name = next(
        (c for c in roads_4326.columns
         if str(c).startswith("num_buildings_") and str(c).endswith("m")),
        None,
    )

    has_uv = ("u" in roads_4326.columns and "v" in roads_4326.columns)
    print(f"  [graph] roads has u/v columns: {has_uv} | total roads: {len(roads_4326):,}")

    if has_uv and road_nodes_gdf is not None:
        # OSMnx mode
        nodes_4326 = _to_4326(road_nodes_gdf).reset_index(drop=True)
        if "osmid" in nodes_4326.columns:
            osmids = nodes_4326["osmid"].values
        else:
            osmids = np.arange(len(nodes_4326))

        lats = nodes_4326.geometry.y.values
        lons = nodes_4326.geometry.x.values
        for nid, lat, lon in zip(osmids, lats, lons):
            G.add_node(int(nid), lat=float(lat), lon=float(lon),
                       x=float(lon), y=float(lat))
        print(f"  [graph] added {G.number_of_nodes():,} nodes from road_nodes")

        u_arr      = roads_4326["u"].values.astype(np.int64)
        v_arr      = roads_4326["v"].values.astype(np.int64)
        length_arr = roads_4326["length"].values.astype(float) \
                     if "length" in roads_4326.columns \
                     else np.ones(len(roads_4326))
        num_b_arr  = roads_4326[col_name].values.astype(int) \
                     if col_name else np.zeros(len(roads_4326), dtype=int)

        node_set = set(G.nodes())
        added = 0
        for u, v, length, num_b in zip(u_arr, v_arr, length_arr, num_b_arr):
            if u in node_set and v in node_set:
                G.add_edge(int(u), int(v), length=float(length),
                           num_buildings=int(num_b))
                added += 1
        print(f"  [graph] added {added:,} edges from u/v columns")

    else:
        # QGIS mode — geometry endpoints, haversine length
        print("  [graph] no u/v columns — geometry endpoints mode (node = tuple)")
        _add_edges_from_geometry(G, roads_4326, col_name)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"  [graph] done → nodes: {n_nodes:,}, edges: {n_edges:,}")

    if n_nodes > 0:
        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        print(f"  [graph] components: {len(components):,} | "
              f"largest: {len(largest):,} nodes ({len(largest)/n_nodes*100:.1f}%)")

    return G


def _add_edges_from_geometry(G: nx.Graph, roads_4326: gpd.GeoDataFrame, col_name):
    """
    Node = tuple(lon, lat) — consistent type
    Length = haversine meters (✅ ไม่ใช้ part.length องศา)
    """
    geoms  = roads_4326.geometry.values
    num_bs = roads_4326[col_name].values.astype(int) if col_name \
             else [0] * len(geoms)

    for geom, num_b in zip(geoms, num_bs):
        if geom is None or geom.is_empty or not geom.is_valid:
            continue
        parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            if part.geom_type != "LineString":
                continue
            coords = list(part.coords)
            if len(coords) < 2:
                continue
            s, e = coords[0], coords[-1]
            u = (round(s[0], 7), round(s[1], 7))
            v = (round(e[0], 7), round(e[1], 7))
            length_m = _linestring_length_m(coords)
            G.add_node(u, x=s[0], y=s[1], lon=s[0], lat=s[1])
            G.add_node(v, x=e[0], y=e[1], lon=e[0], lat=e[1])
            G.add_edge(u, v, length=length_m, num_buildings=int(num_b))


def get_largest_component(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    largest = max(nx.connected_components(G), key=len)
    return G.subgraph(largest).copy()


def load_with_osmnx_nodes():
    import os as _os
    import geopandas as gpd

    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    road_nodes_path = _os.path.join(root, "data", "raw", "road_nodes.geojson")

    if not _os.path.exists(road_nodes_path):
        print("  [data_loader] road_nodes.geojson not found — using edge endpoints")
        return None

    print("  [data_loader] loading road_nodes.geojson...")
    gdf = gpd.read_file(road_nodes_path)
    valid = gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.is_valid
    gdf = gdf[valid].copy()
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    print(f"  [data_loader] road_nodes loaded: {len(gdf):,}")
    return gdf
