from typing import Optional, Tuple
from geopy.geocoders import Nominatim

_geolocator = Nominatim(user_agent="traffic_blockchain_app")

def geocode_place(query: str) -> Optional[Tuple[float, float]]:
    if not query:
        return None
    location = _geolocator.geocode(query)
    if location is None:
        return None
    return (location.latitude, location.longitude)