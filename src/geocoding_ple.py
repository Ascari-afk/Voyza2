"""
src/geocoding.py
geocode ชื่อสถานที่ → (lat, lon) โดยใช้ Nominatim (geopy)
FIX: เพิ่ม timeout, retry, Bangkok suffix
"""
import time
from typing import Optional, Tuple

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# singleton เดียว ไม่สร้างซ้ำทุกครั้ง
_geolocator = Nominatim(user_agent="traffic_blockchain_app_v2", timeout=10)


def geocode_place(query: str, retry: int = 2) -> Optional[Tuple[float, float]]:
    """
    รับชื่อสถานที่ (ไทย/อังกฤษ) → คืน (lat, lon) หรือ None

    FIX จากเดิม:
    - เพิ่ม timeout=10 (เดิมไม่มี → timeout ค่า default ของ geopy = 1 วิ ซึ่งสั้นเกินไป)
    - ลอง query ด้วย suffix ", Bangkok, Thailand" ก่อน ช่วยให้ Nominatim pinpoint ได้แม่น
    - retry เมื่อ GeocoderTimedOut
    """
    if not query or not query.strip():
        return None

    queries_to_try = [
        f"{query.strip()}, Bangkok, Thailand",
        query.strip(),
    ]

    for attempt in range(retry + 1):
        for q in queries_to_try:
            try:
                loc = _geolocator.geocode(q)
                if loc is not None:
                    return (loc.latitude, loc.longitude)
            except GeocoderTimedOut:
                if attempt < retry:
                    time.sleep(1.5)
                    continue
                return None
            except GeocoderServiceError:
                return None

    return None
