ROADS_PATH     = "data/raw/Roads.geojson"
BUILDINGS_PATH = "data/raw/Buildings.geojson"

# -------------------------------------------------------
# Sampling limits
# 0 = โหลดทั้งหมด (ช้า แต่ครอบคลุม)
# แนะนำสำหรับ development: roads=20000, buildings=40000
# แนะนำสำหรับ demo/presentation: roads=50000, buildings=80000
# ✅ FIX: เพิ่มเป็น 50000/80000 เพื่อให้แผนที่ครอบทั่วกรุงเทพมากขึ้น
# -------------------------------------------------------
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # ขึ้นจาก src/ ไป root
ROADS_PATH = ROOT / "data" / "raw" / "Roads.geojson"
BUILDINGS_PATH = ROOT / "data" / "raw" / "Buildings.geojson"

MAX_ROADS_ROWS     = 50000   # ~8-12 วิ (จาก 366K → spatial sample กระจายทั่วกรุงเทพ)
MAX_BUILDINGS_ROWS = 80000   # ~2-3 วิ

ROAD_BUFFER_METERS    = 200
GRID_SIZE_METERS      = 500
MAX_DEGREE            = 6
N_SAMPLES_SHORTEST_PATH = 50
