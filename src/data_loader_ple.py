import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString

from .config import (
    ROADS_PATH,
    BUILDINGS_PATH,
    ROAD_BUFFER_METERS,
    MAX_ROADS_ROWS,
    MAX_BUILDINGS_ROWS,
)


def load_roads_buildings():
    roads     = gpd.read_file(ROADS_PATH)
    buildings = gpd.read_file(BUILDINGS_PATH)

    # ✅ FIX: shuffle ก่อน truncate เพื่อให้กระจายทั่วพื้นที่ (ไม่ cluster)
    if MAX_ROADS_ROWS and MAX_ROADS_ROWS > 0 and len(roads) > MAX_ROADS_ROWS:
        roads = roads.sample(n=MAX_ROADS_ROWS, random_state=42).reset_index(drop=True)

    roads_bbox = roads.total_bounds

    buildings = buildings.cx[
        roads_bbox[0]:roads_bbox[2],
        roads_bbox[1]:roads_bbox[3]
    ]

    # ✅ FIX: shuffle buildings ก่อน truncate เช่นกัน
    if MAX_BUILDINGS_ROWS and MAX_BUILDINGS_ROWS > 0 and len(buildings) > MAX_BUILDINGS_ROWS:
        buildings = buildings.sample(n=MAX_BUILDINGS_ROWS, random_state=42).reset_index(drop=True)

    if roads.crs != buildings.crs:
        buildings = buildings.to_crs(roads.crs)

    return roads, buildings


def add_building_features_to_roads(roads_gdf, buildings_gdf):
    print("  [features] projecting to meters (EPSG:32647)...")
    roads_proj = roads_gdf.to_crs(epsg=32647)
    buildings_proj = buildings_gdf.to_crs(epsg=32647)

    print("  [features] buffering roads...")
    roads_proj = roads_proj.copy()
    roads_proj["geometry_buffer"] = roads_proj.geometry.buffer(ROAD_BUFFER_METERS)

    roads_buffer_gdf = roads_proj[["geometry_buffer"]].rename(
        columns={"geometry_buffer": "geometry"}
    )
    roads_buffer_gdf = gpd.GeoDataFrame(
        roads_buffer_gdf, geometry="geometry", crs=roads_proj.crs
    )

    print("  [features] spatial join buildings within buffer...")
    joined = gpd.sjoin(
        buildings_proj,
        roads_buffer_gdf,
        how="inner",
        predicate="intersects",
    )
    print("  [features] join rows:", len(joined))

    counts = joined.groupby("index_right").size()
    col_name = f"num_buildings_{ROAD_BUFFER_METERS}m"
    roads_proj[col_name] = roads_proj.index.map(counts).fillna(0).astype(int)

    print(f"  [features] feature '{col_name}' added to roads")
    roads_with_features = roads_proj.to_crs(roads_gdf.crs)
    return roads_with_features


def build_road_graph_from_roads_gdf(roads_gdf):
    """
    สร้าง NetworkX graph จาก roads GeoDataFrame
    FIX: แปลงเป็น EPSG:4326 ก่อน เพื่อให้ node ได้ lat/lon จริง
    และเก็บ lat/lon ตั้งแต่ต้น (ไม่ต้อง patch ทีหลัง)
    """
    print("  [graph] building NetworkX graph from roads...")
    G = nx.Graph()

    # ✅ FIX: แปลงเป็น 4326 ก่อนดึง coords → lat/lon ถูกต้อง
    roads_4326 = (
        roads_gdf.to_crs(epsg=4326)
        if (roads_gdf.crs and roads_gdf.crs.to_epsg() != 4326)
        else roads_gdf
    )

    col_name = next(
        (c for c in roads_4326.columns
         if str(c).startswith("num_buildings_") and str(c).endswith("m")),
        None,
    )

    for idx, row in roads_4326.iterrows():
        geom = row.geometry
        if not isinstance(geom, LineString):
            continue

        coords = list(geom.coords)
        if len(coords) < 2:
            continue

        start = coords[0]   # (lon, lat) ใน 4326
        end   = coords[-1]

        u = (start[0], start[1])
        v = (end[0],   end[1])

        length = geom.length
        num_buildings = int(row.get(col_name, 0)) if col_name else 0

        # ✅ FIX: เก็บ lat/lon ตั้งแต่ตอนสร้าง node ไม่ต้อง attach ทีหลัง
        G.add_node(u, x=start[0], y=start[1], lon=start[0], lat=start[1])
        G.add_node(v, x=end[0],   y=end[1],   lon=end[0],   lat=end[1])

        G.add_edge(u, v, length=length, num_buildings=num_buildings)

    print("  [graph] done. nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
    return G
