"""
src/industrial_data.py
Industrial data layer for logistics demo.
"""
import random
import numpy as np
from dataclasses import dataclass
from typing import List
import pandas as pd

WAREHOUSES = [
    {"id": "WH-01", "name": "คลังสินค้า ลาดกระบัง",      "lat": 13.7320, "lon": 100.7750, "capacity": 500},
    {"id": "WH-02", "name": "คลังสินค้า บางนา",          "lat": 13.6670, "lon": 100.6050, "capacity": 350},
    {"id": "WH-03", "name": "คลังสินค้า บางพลี",         "lat": 13.5970, "lon": 100.6870, "capacity": 400},
    {"id": "WH-04", "name": "คลังสินค้า มีนบุรี",        "lat": 13.8100, "lon": 100.7560, "capacity": 300},
    {"id": "WH-05", "name": "Distribution Center รังสิต", "lat": 14.0280, "lon": 100.6160, "capacity": 600},
]

DELIVERY_ZONES = [
    {"id": "DZ-01", "name": "สยาม / ราชประสงค์",  "lat": 13.7456, "lon": 100.5342, "demand": "high"},
    {"id": "DZ-02", "name": "อโศก / สุขุมวิท",    "lat": 13.7368, "lon": 100.5601, "demand": "high"},
    {"id": "DZ-03", "name": "ลาดพร้าว / รัชดา",  "lat": 13.8021, "lon": 100.5700, "demand": "medium"},
    {"id": "DZ-04", "name": "ดอนเมือง",            "lat": 13.9132, "lon": 100.6070, "demand": "medium"},
    {"id": "DZ-05", "name": "พระรามเก้า / เอกมัย", "lat": 13.7220, "lon": 100.5760, "demand": "high"},
    {"id": "DZ-06", "name": "บางรัก / สีลม",       "lat": 13.7230, "lon": 100.5230, "demand": "high"},
    {"id": "DZ-07", "name": "ปทุมวัน / จุฬาฯ",    "lat": 13.7380, "lon": 100.5290, "demand": "medium"},
    {"id": "DZ-08", "name": "ประตูน้ำ / พญาไท",   "lat": 13.7540, "lon": 100.5360, "demand": "medium"},
    {"id": "DZ-09", "name": "บางซื่อ / นนทบุรี",  "lat": 13.8060, "lon": 100.5220, "demand": "low"},
    {"id": "DZ-10", "name": "ตลิ่งชัน / บางแค",   "lat": 13.7730, "lon": 100.4390, "demand": "low"},
]

PRIORITY_LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
PRODUCT_TYPES   = ["อิเล็กทรอนิกส์", "เครื่องจักร", "วัตถุดิบ", "สินค้าอุปโภค", "อาหารแช่แข็ง"]

@dataclass
class DeliveryOrder:
    order_id: str
    product_type: str
    weight_kg: float
    priority: str
    time_window_hr: float
    warehouse_id: str
    delivery_zone_id: str
    status: str = "PENDING"

def generate_normal_orders(n: int = 12, seed: int = 42) -> List[DeliveryOrder]:
    rng = random.Random(seed)
    orders = []
    for i in range(n):
        wh = rng.choice(WAREHOUSES)
        dz = rng.choice(DELIVERY_ZONES)
        priority = rng.choices(PRIORITY_LEVELS, weights=[0.05, 0.20, 0.50, 0.25])[0]
        time_window = {"CRITICAL": 1.0, "HIGH": 2.0, "MEDIUM": 4.0, "LOW": 8.0}[priority]
        orders.append(DeliveryOrder(
            order_id=f"ORD-{1000+i}",
            product_type=rng.choice(PRODUCT_TYPES),
            weight_kg=round(rng.uniform(10, 500), 1),
            priority=priority,
            time_window_hr=time_window,
            warehouse_id=wh["id"],
            delivery_zone_id=dz["id"],
        ))
    return orders

def generate_surge_orders(n: int = 20, seed: int = 99) -> List[DeliveryOrder]:
    rng = random.Random(seed)
    orders = []
    for i in range(n):
        wh = rng.choice(WAREHOUSES)
        dz = rng.choice([z for z in DELIVERY_ZONES if z["demand"] == "high"])
        priority = rng.choices(PRIORITY_LEVELS, weights=[0.25, 0.45, 0.20, 0.10])[0]
        time_window = {"CRITICAL": 0.5, "HIGH": 1.5, "MEDIUM": 3.0, "LOW": 6.0}[priority]
        orders.append(DeliveryOrder(
            order_id=f"SURGE-{2000+i}",
            product_type=rng.choice(PRODUCT_TYPES),
            weight_kg=round(rng.uniform(50, 800), 1),
            priority=priority,
            time_window_hr=time_window,
            warehouse_id=wh["id"],
            delivery_zone_id=dz["id"],
        ))
    return orders

PRIORITY_SCORE = {"CRITICAL": 100, "HIGH": 70, "MEDIUM": 40, "LOW": 10}

def score_order(order: DeliveryOrder) -> float:
    p  = PRIORITY_SCORE[order.priority]
    tw = max(order.time_window_hr, 0.1)
    w  = np.log1p(order.weight_kg)
    return round((p / tw) * w, 2)

def prioritize_orders(orders: List[DeliveryOrder]) -> List[DeliveryOrder]:
    return sorted(orders, key=lambda o: score_order(o), reverse=True)

def orders_to_df(orders: List[DeliveryOrder]) -> pd.DataFrame:
    wh_map = {w["id"]: w["name"] for w in WAREHOUSES}
    dz_map = {d["id"]: d["name"] for d in DELIVERY_ZONES}
    rows = []
    for o in orders:
        rows.append({
            "Order ID":        o.order_id,
            "สินค้า":          o.product_type,
            "น้ำหนัก (kg)":   o.weight_kg,
            "Priority":        o.priority,
            "Time Window (hr)": o.time_window_hr,
            "Urgency Score":   score_order(o),
            "คลังสินค้า":     wh_map.get(o.warehouse_id, o.warehouse_id),
            "จุดส่ง":         dz_map.get(o.delivery_zone_id, o.delivery_zone_id),
            "Status":          o.status,
        })
    return pd.DataFrame(rows)

@dataclass
class ScenarioKPI:
    total_orders: int
    critical_orders: int
    avg_urgency_score: float
    estimated_fleet_size: int
    at_risk_orders: int
    baseline_avg_time_min: float
    optimized_avg_time_min: float

    @property
    def time_saved_pct(self) -> float:
        if self.baseline_avg_time_min == 0:
            return 0.0
        return round((1 - self.optimized_avg_time_min / self.baseline_avg_time_min) * 100, 1)

    @property
    def throughput_gain_pct(self) -> float:
        return round(self.time_saved_pct * 1.15, 1)

def compute_kpi(orders: List[DeliveryOrder], scenario: str = "normal") -> ScenarioKPI:
    n        = len(orders)
    critical = sum(1 for o in orders if o.priority == "CRITICAL")
    avg_score = np.mean([score_order(o) for o in orders]) if orders else 0
    fleet    = max(3, n // 4)
    at_risk  = sum(1 for o in orders if o.time_window_hr <= 1.0)
    if scenario == "normal":
        baseline, optimized = 42.0, 31.0
    else:
        baseline, optimized = 58.0, 38.0
    return ScenarioKPI(
        total_orders=n, critical_orders=critical,
        avg_urgency_score=round(avg_score, 1),
        estimated_fleet_size=fleet, at_risk_orders=at_risk,
        baseline_avg_time_min=baseline, optimized_avg_time_min=optimized,
    )
