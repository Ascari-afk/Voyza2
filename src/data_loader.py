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
    print("  [data_loader] reading roads from", ROADS_PATH)
    roads = gpd.read_file(ROADS_PATH)
    print("  [data_loader] roads loaded:", len(roads))

    print("  [data_loader] reading buildings from", BUILDINGS_PATH)
    buildings = gpd.read_file(BUILDINGS_PATH)
    print("  [data_loader] buildings loaded:", len(buildings))

    # เลือกแค่บางส่วนของ roads เพื่อให้ไม่หนักเกินไป
    if len(roads) > MAX_ROADS_ROWS:
        roads = roads.head(MAX_ROADS_ROWS)
        print(f"  [data_loader] roads truncated to first {MAX_ROADS_ROWS} rows")

    # ตัด buildings ตาม bounding box ของ roads ชุดนี้
    roads_bbox = roads.total_bounds  # [minx, miny, maxx, maxy]
    print("  [data_loader] roads bbox:", roads_bbox)

    buildings = buildings.cx[roads_bbox[0]:roads_bbox[2], roads_bbox[1]:roads_bbox[3]]
    print("  [data_loader] buildings clipped to bbox, rows:", len(buildings))

    if len(buildings) > MAX_BUILDINGS_ROWS:
        buildings = buildings.head(MAX_BUILDINGS_ROWS)
        print(f"  [data_loader] buildings truncated to first {MAX_BUILDINGS_ROWS} rows")

    if roads.crs != buildings.crs:
        print("  [data_loader] reprojecting buildings to match roads CRS")
        buildings = buildings.to_crs(roads.crs)

    print("  [data_loader] CRS roads:", roads.crs, "buildings:", buildings.crs)
    return roads, buildings


def add_building_features_to_roads(roads_gdf, buildings_gdf):
    """
    สร้างฟีเจอร์: num_buildings_<R>m = จำนวนตึกในรัศมี ROAD_BUFFER_METERS รอบแต่ละ segment ถนน
    """
    print("  [features] projecting to meters (EPSG:32647)...")
    roads_proj = roads_gdf.to_crs(epsg=32647)
    buildings_proj = buildings_gdf.to_crs(epsg=32647)

    print("  [features] buffering roads...")
    roads_proj["geometry_buffer"] = roads_proj.geometry.buffer(ROAD_BUFFER_METERS)

    print("  [features] building buffer GeoDataFrame...")
    roads_buffer_gdf = roads_proj[["geometry_buffer"]].rename(
        columns={"geometry_buffer": "geometry"}
    )
    roads_buffer_gdf = gpd.GeoDataFrame(
        roads_buffer_gdf, geometry="geometry", crs=roads_proj.crs
    )

    print("  [features] spatial join buildings within buffer...")
    # predicate="intersects" สำหรับ GeoPandas >=0.10 [web:99][web:102]
    joined = gpd.sjoin(
        buildings_proj,
        roads_buffer_gdf,
        how="inner",
        predicate="intersects",
    )
    print("  [features] join rows:", len(joined))

    print("  [features] counting buildings per road segment...")
    counts = joined.groupby("index_right").size()

    col_name = f"num_buildings_{ROAD_BUFFER_METERS}m"
    roads_proj[col_name] = roads_proj.index.map(counts).fillna(0).astype(int)

    print(f"  [features] feature '{col_name}' added to roads")

    print("  [features] reproject back to original CRS...")
    roads_with_features = roads_proj.to_crs(roads_gdf.crs)

    return roads_with_features


def build_road_graph_from_roads_gdf(roads_gdf):
    G = nx.Graph()

    # แปลงเป็น 4326 ก่อน เพื่อให้ coords เป็น lon/lat จริง
    roads_4326 = roads_gdf.to_crs(epsg=4326) if (
        roads_gdf.crs and roads_gdf.crs.to_epsg() != 4326
    ) else roads_gdf

    col_name = next(
        (c for c in roads_4326.columns
         if str(c).startswith("num_buildings_") and str(c).endswith("m")),
        None
    )

    for idx, row in roads_4326.iterrows():
        geom = row.geometry
        if not isinstance(geom, LineString):
            continue
        coords = list(geom.coords)
        if len(coords) < 2:
            continue

        start = coords[0]   # (lon, lat)
        end   = coords[-1]  # (lon, lat)

        u = (start[0], start[1])
        v = (end[0],   end[1])

        length = geom.length
        num_buildings = int(row.get(col_name, 0)) if col_name else 0

        # ✅ เก็บ lat/lon ตั้งแต่ต้น
        G.add_node(u, x=start[0], y=start[1], lon=start[0], lat=start[1])
        G.add_node(v, x=end[0],   y=end[1],   lon=end[0],   lat=end[1])
        G.add_edge(u, v, length=length, num_buildings=num_buildings)

    return G