import numpy as np
import networkx as nx

# มิติของ hypervector
D = 10000

# random seed เพื่อ reproducible
rng = np.random.default_rng(42)

def random_hv():
    """สร้าง bipolar hypervector ขนาด D: ค่า +1/-1"""
    hv = rng.choice([-1, 1], size=D)
    return hv

def bind(a, b):
    """binding: ใช้ element-wise multiplication (เทียบเท่า XOR ใน bipolar space)"""
    return a * b

def bundle(vectors):
    """bundling: รวมหลาย hypervector โดยการ sum แล้ว sign"""
    if len(vectors) == 0:
        return np.zeros(D)
    s = np.sum(vectors, axis=0)
    # majority sign
    s[s > 0] = 1
    s[s < 0] = -1
    # ถ้า 0 ให้สุ่มเล็กน้อย
    zeros = s == 0
    s[zeros] = rng.choice([-1, 1], size=zeros.sum())
    return s

# -------------------------
# สร้างฐาน hypervector สำหรับ feature ต่าง ๆ
# -------------------------

# role vectors (ตำแหน่ง)
ROLE_LENGTH = random_hv()
ROLE_BUILDINGS = random_hv()
ROLE_ORIENTATION = random_hv()

# bin vectors
N_LEN_BINS = 8
N_BLDG_BINS = 8
N_ORI_BINS = 8

LEN_BINS = [random_hv() for _ in range(N_LEN_BINS)]
BLDG_BINS = [random_hv() for _ in range(N_BLDG_BINS)]
ORI_BINS = [random_hv() for _ in range(N_ORI_BINS)]


def digitize_with_max(value, bins):
    """ตัด value เข้าช่อง bin ตามขอบ bins แต่ถ้ามากกว่าบินสุดท้ายก็ใส่ bin สุดท้าย"""
    idx = np.digitize([value], bins)[0]
    if idx >= len(bins):
        idx = len(bins) - 1
    return idx


def compute_edge_orientation(u, v):
    """คำนวณมุมทิศทางของถนน u->v (radians)"""
    x1, y1 = u
    x2, y2 = v
    dx = x2 - x1
    dy = y2 - y1
    angle = np.arctan2(dy, dx)  # -pi .. pi
    return angle


def edge_to_hv(u, v, data, len_bins_edges, bldg_bins_edges):
    """
    แปลง edge หนึ่งเส้น (u, v, data dict จาก NetworkX) เป็น hypervector
    data ควรมี key: 'length', 'num_buildings_XXXm'
    """
    hv_parts = []

    # 1) Length
    length = float(data.get("length", 0.0))
    len_bin_idx = digitize_with_max(length, len_bins_edges)
    hv_len = bind(ROLE_LENGTH, LEN_BINS[len_bin_idx])
    hv_parts.append(hv_len)

    # 2) Number of buildings
    # หา key ที่เป็น num_buildings_...m
    bldg_keys = [k for k in data.keys() if str(k).startswith("num_buildings_") and str(k).endswith("m")]
    if len(bldg_keys) > 0:
        bkey = bldg_keys[0]
        num_b = float(data.get(bkey, 0.0))
    else:
        num_b = 0.0
    bldg_bin_idx = digitize_with_max(num_b, bldg_bins_edges)
    hv_bldg = bind(ROLE_BUILDINGS, BLDG_BINS[bldg_bin_idx])
    hv_parts.append(hv_bldg)

    # 3) Orientation
    angle = compute_edge_orientation(u, v)
    # แปลงจาก -pi..pi เป็น 0..2pi
    angle_01 = (angle + np.pi) / (2 * np.pi)  # 0..1
    ori_bin_idx = int(angle_01 * N_ORI_BINS)
    if ori_bin_idx >= N_ORI_BINS:
        ori_bin_idx = N_ORI_BINS - 1
    hv_ori = bind(ROLE_ORIENTATION, ORI_BINS[ori_bin_idx])
    hv_parts.append(hv_ori)

    # รวมทั้งหมด
    hv_edge = bundle(hv_parts)
    return hv_edge


def build_edge_hv_dict(G: nx.Graph):
    """
    สร้าง dict: (u, v) -> hypervector สำหรับทุก edge ในกราฟ G
    """
    # เตรียม bin edges จาก distribution จริงในกราฟ
    lengths = []
    bcounts = []
    for u, v, data in G.edges(data=True):
        lengths.append(float(data.get("length", 0.0)))
        bkeys = [k for k in data.keys() if str(k).startswith("num_buildings_") and str(k).endswith("m")]
        if len(bkeys) > 0:
            bcounts.append(float(data.get(bkeys[0], 0.0)))
        else:
            bcounts.append(0.0)

    lengths = np.array(lengths)
    bcounts = np.array(bcounts)

    # สร้าง quantile-based bins เพื่อให้กระจายดี [web:38]
    # ป้องกัน edge-case: ถ้า length หรือ buildings เป็นค่าซ้ำ ให้ fallback เป็น linspace
    try:
        len_bins_edges = np.quantile(lengths, np.linspace(0, 1, N_LEN_BINS))
    except Exception:
        len_bins_edges = np.linspace(lengths.min(), lengths.max(), N_LEN_BINS)

    try:
        bldg_bins_edges = np.quantile(bcounts, np.linspace(0, 1, N_BLDG_BINS))
    except Exception:
        bldg_bins_edges = np.linspace(bcounts.min(), bcounts.max(), N_BLDG_BINS)

    edge_hv = {}
    for u, v, data in G.edges(data=True):
        hv = edge_to_hv(u, v, data, len_bins_edges, bldg_bins_edges)
        edge_hv[(u, v)] = hv

    return edge_hv