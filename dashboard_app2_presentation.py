import sys
from pathlib import Path
import time
import random
import geopandas as gpd
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data_loader2 import (
    load_roads_buildings_split,
    add_building_features_to_roads,
    build_road_graph_from_roads_gdf,
    load_with_osmnx_nodes,
    get_largest_component,
)
from src.graph_blockchain_style import (
    build_blockchain_style_graph,
    apply_degree_constraint,
)
from src.hdc_encoding import build_edge_hv_dict, D
from src.config import N_SAMPLES_SHORTEST_PATH
from src.routing_engine import build_node_spatial_index, get_routes_from_latlon
# ─────────────────────────────────────────────
# Cached load
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_data_and_graphs():
    roads_display, roads_full, buildings = load_roads_buildings_split()
    roads_feat     = add_building_features_to_roads(roads_display, buildings)
    road_nodes_gdf = load_with_osmnx_nodes()
    G_real         = build_road_graph_from_roads_gdf(roads_full, road_nodes_gdf=road_nodes_gdf)
    G_real_conn    = get_largest_component(G_real)
    G_abs, node_to_block = build_blockchain_style_graph(G_real_conn)
    G_abs_constrained    = apply_degree_constraint(G_abs)
    return roads_display, buildings, roads_feat, G_real, G_real_conn, G_abs_constrained, node_to_block
@st.cache_resource(show_spinner=True)
def build_spatial_index(_G_real_conn: nx.Graph):
    return build_node_spatial_index(_G_real_conn)
@st.cache_data(show_spinner=False)
def build_hdc_dict(_G_real_conn):
    """
    สร้าง edge hypervector dict จาก NetworkX Graph
    build_edge_hv_dict รับ Graph ไม่ใช่ GeoDataFrame
    """
    return build_edge_hv_dict(_G_real_conn)
# ─────────────────────────────────────────────
# Folium helpers
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_roads_geojson(_roads: gpd.GeoDataFrame) -> str:
    r = _roads.to_crs(epsg=4326) if (_roads.crs and _roads.crs.to_epsg() != 4326) else _roads
    return r[[r.geometry.name]].to_json()
@st.cache_data(show_spinner=False)
def build_buildings_geojson(_buildings: gpd.GeoDataFrame) -> str:
    b = _buildings[_buildings.geometry.notna() & _buildings.geometry.is_valid].copy()
    b = b.to_crs(epsg=4326) if (b.crs and b.crs.to_epsg() != 4326) else b
    return b[[b.geometry.name]].to_json()
def plot_roads_buildings_map_folium(roads, buildings):
    roads_4326 = roads.to_crs(epsg=4326) if (roads.crs and roads.crs.to_epsg() != 4326) else roads
    bounds     = roads_4326.total_bounds
    m = folium.Map(
        location=[float((bounds[1]+bounds[3])/2), float((bounds[0]+bounds[2])/2)],
        zoom_start=12, tiles="OpenStreetMap", prefer_canvas=True,
    )
    folium.GeoJson(build_roads_geojson(roads), name="Roads",
                   style_function=lambda _: {"color":"#1a73e8","weight":1.5,"opacity":0.75},
                   tooltip=None).add_to(m)
    folium.GeoJson(build_buildings_geojson(buildings), name="Buildings",
                   style_function=lambda _: {"color":"#e53935","weight":0.5,
                                             "fillColor":"#e53935","fillOpacity":0.4},
                   tooltip=None).add_to(m)
    folium.LayerControl().add_to(m)
    return m
# ─────────────────────────────────────────────
# Tab 2 helpers — Graph Stats + Metrics
# ─────────────────────────────────────────────
def plot_graph_sizes(G_real, G_abs):
    labels = ["Real graph", "Blockchain-style graph"]
    fig = go.Figure(data=[
        go.Bar(name="Nodes", x=labels, y=[G_real.number_of_nodes(), G_abs.number_of_nodes()],
               marker_color=["#1a73e8","#e53935"]),
        go.Bar(name="Edges", x=labels, y=[G_real.number_of_edges(), G_abs.number_of_edges()],
               marker_color=["#42a5f5","#ef9a9a"]),
    ])
    fig.update_layout(barmode="group", title="Graph Size Comparison",
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font_color="white")
    return fig
def render_graph_metrics(G_real, G_abs):
    """
    แสดง Metrics Summary Cards
    - Node compression ratio
    - Edge compression ratio
    - avg degree ทั้งสอง graph
    """
    n_real  = G_real.number_of_nodes()
    n_abs   = G_abs.number_of_nodes()
    e_real  = G_real.number_of_edges()
    e_abs   = G_abs.number_of_edges()
    node_compression = (1 - n_abs / n_real) * 100 if n_real > 0 else 0
    edge_compression = (1 - e_abs / e_real) * 100 if e_real > 0 else 0
    avg_deg_real = (2 * e_real / n_real) if n_real > 0 else 0
    avg_deg_abs  = (2 * e_abs  / n_abs)  if n_abs  > 0 else 0
    st.markdown("### Graph Compression Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Node Compression", f"{node_compression:.1f}%",
              help=f"ลด {n_real:,} → {n_abs:,} nodes")
    c2.metric("Edge Compression", f"{edge_compression:.1f}%",
              help=f"ลด {e_real:,} → {e_abs:,} edges")
    c3.metric("Avg Degree (Real)", f"{avg_deg_real:.2f}")
    c4.metric("Avg Degree (Blockchain)", f"{avg_deg_abs:.2f}")
    st.info(
        f"Blockchain-style graph ลด node จาก **{n_real:,}** เหลือ **{n_abs:,}** "
        f"(**{node_compression:.1f}% reduction**) โดยยังคง topology ของเครือข่ายไว้"
    )
# ─────────────────────────────────────────────
# Tab 3 helpers — Shortest Path Comparison
# ─────────────────────────────────────────────
def compute_shortest_path_stats(G_real, G_abs, node_to_block, n_samples=N_SAMPLES_SHORTEST_PATH):
    nodes = list(G_real.nodes())
    if len(nodes) < 2:
        return None
    pairs = random.sample(nodes, min(n_samples * 2, len(nodes)))
    pairs = [(pairs[i], pairs[i+1]) for i in range(0, len(pairs)-1, 2)][:n_samples]
    real_times, abs_times, real_lengths, abs_lengths = [], [], [], []
    for s_real, t_real in pairs:
        if s_real == t_real:
            continue
        t0 = time.perf_counter()
        try:
            length_real = nx.shortest_path_length(G_real, s_real, t_real, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        t1 = time.perf_counter()
        s_block = node_to_block.get(s_real)
        t_block = node_to_block.get(t_real)
        if s_block is None or t_block is None:
            continue
        if s_block == t_block:
            length_abs = length_real
            dt_abs = 0.0
        else:
            t2 = time.perf_counter()
            try:
                length_abs = nx.shortest_path_length(G_abs, s_block, t_block, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            dt_abs = time.perf_counter() - t2
        real_lengths.append(length_real)
        abs_lengths.append(length_abs)
        real_times.append(t1 - t0)
        abs_times.append(dt_abs)
    if not real_times or not abs_times:
        return None
    rt_mean  = float(np.mean(real_times))
    at_mean  = float(np.mean(abs_times))
    speedup  = rt_mean / at_mean if at_mean > 1e-12 else float("inf")
    rl_mean  = float(np.mean(real_lengths))
    al_mean  = float(np.mean(abs_lengths))
    len_diff = (al_mean - rl_mean) / rl_mean * 100 if rl_mean > 0 else 0
    return {
        "n_pairs":        len(real_times),
        "real_len_mean":  rl_mean,
        "abs_len_mean":   al_mean,
        "len_diff_pct":   len_diff,
        "real_time_mean": rt_mean,
        "abs_time_mean":  at_mean,
        "speedup":        speedup,
    }
def plot_time_comparison(stats):
    labels = ["Real graph", "Blockchain-style graph"]
    times  = [stats["real_time_mean"], stats["abs_time_mean"]]
    fig = go.Figure(data=[
        go.Bar(
            x=labels, y=times,
            text=[f"{t*1e3:.3f} ms" for t in times],
            textposition="auto",
            marker_color=["#1a73e8", "#e53935"],
        ),
    ])
    fig.update_layout(
        title=f"Avg Shortest Path Time  (Speedup: {stats['speedup']:.1f}×)",
        yaxis_title="Time (seconds)",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
    )
    return fig
def render_path_metrics(stats):
    """Cards สรุปผล experiment"""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pairs tested",    stats["n_pairs"])
    c2.metric("Speedup",         f"{stats['speedup']:.1f}×",
              help="Blockchain / Real graph query time ratio")
    c3.metric("Path length diff", f"{stats['len_diff_pct']:+.2f}%",
              help="Blockchain path ยาวกว่า Real path กี่ % (+ = ยาวกว่า)")
    c4.metric("Real graph avg path", f"{stats['real_len_mean']/1000:.2f} km")
    if stats["speedup"] > 1:
        st.success(
            f"Blockchain-style graph **เร็วกว่า {stats['speedup']:.1f}×** "
            f"โดยเสียความยาวเส้นทางเพิ่มเพียง **{stats['len_diff_pct']:+.2f}%** "
            f"— acceptable trade-off สำหรับ large-scale routing"
        )
    else:
        st.info("Blockchain graph ใช้เวลาใกล้เคียงกับ Real graph ในชุดข้อมูลนี้")
# ─────────────────────────────────────────────
# Tab 4 helpers — Dynamic Routing
# ─────────────────────────────────────────────
def _node_latlon(G: nx.Graph, node) -> tuple:
    data = G.nodes.get(node, {})
    lat  = data.get("lat")
    lon  = data.get("lon")
    if (lat is None or lon is None) and isinstance(node, tuple) and len(node) == 2:
        lon, lat = float(node[0]), float(node[1])
    return lat, lon
def _route_quality_label(cost_km: float, base_km: float) -> str:
    diff = cost_km - base_km
    if diff < 0.5:
        return "★★★  Most Direct"
    elif diff < 2.0:
        return f"★★☆  Alternative (+{diff:.1f} km)"
    else:
        return f"★☆☆  Detour (+{diff:.1f} km)"
def build_routes_map(G_real: nx.Graph, routes_info: dict):
    routes  = routes_info.get("routes", [])
    if not routes:
        return None
    colors  = ["#e53935", "#43a047", "#1e88e5"]
    weights = [7, 5, 3]
    offsets = [0.00004, 0.0, -0.00004]
    center_lat, center_lon = 13.75, 100.50
    for r in routes:
        lats, lons = [], []
        for node in r["path"]:
            lat, lon = _node_latlon(G_real, node)
            if lat is not None:
                lats.append(lat)
            if lon is not None:
                lons.append(lon)
        if lats and lons:
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
            break
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12,
                   tiles="OpenStreetMap", prefer_canvas=True)
    for ridx, r in enumerate(routes):
        coords = []
        for node in r["path"]:
            lat, lon = _node_latlon(G_real, node)
            if lat is not None and lon is not None:
                coords.append((float(lat) + offsets[ridx % len(offsets)], float(lon)))
        if not coords:
            continue
        cost_km = r["cost"] / 1000.0
        color   = colors[ridx % len(colors)]
        weight  = weights[ridx % len(weights)]
        label   = _route_quality_label(cost_km, routes[0]["cost"] / 1000.0)
        folium.PolyLine(
            locations=coords, color=color, weight=weight, opacity=0.9,
            tooltip=f"Route {ridx+1} {label}: {cost_km:.2f} km ({len(r['path'])} nodes)",
        ).add_to(m)
        folium.CircleMarker(location=coords[0],  radius=8, color=color,
                            fill=True, fill_opacity=1.0,
                            tooltip=f"Route {ridx+1} — Start").add_to(m)
        folium.CircleMarker(location=coords[-1], radius=8, color=color,
                            fill=True, fill_opacity=1.0,
                            tooltip=f"Route {ridx+1} — End").add_to(m)
    return m
def render_route_cards(routes: list):
    """Route Quality Score cards ใต้แผนที่"""
    if not routes:
        return
    base_km = routes[0]["cost"] / 1000.0
    cols = st.columns(len(routes))
    colors_hex = ["#e53935", "#43a047", "#1e88e5"]
    labels_short = ["Most Direct", "Alternative", "Detour"]
    for i, (r, col) in enumerate(zip(routes, cols)):
        cost_km  = r["cost"] / 1000.0
        diff_km  = cost_km - base_km
        n_nodes  = len(r["path"])
        color    = colors_hex[i % len(colors_hex)]
        badge    = labels_short[i] if i < len(labels_short) else f"Route {i+1}"
        extra    = "" if i == 0 else f" · +{diff_km:.2f} km longer"
        sub_line = f"{n_nodes} nodes{extra}"
        html = (
            '<div style="border-left:4px solid ' + color + '; padding:10px 14px;'
            ' border-radius:6px; background:#1a1d23; margin-bottom:6px;">'
            '<div style="font-size:0.75rem; color:' + color + '; font-weight:700;'
            ' letter-spacing:0.05em; margin-bottom:4px;">'
            'ROUTE ' + str(i + 1) + ' — ' + badge.upper() + '</div>'
            '<div style="font-size:1.6rem; font-weight:700; color:white;">'
            + f"{cost_km:.2f} km" +
            '</div>'
            '<div style="font-size:0.85rem; color:#9e9e9e; margin-top:4px;">'
            + sub_line +
            '</div>'
            '<div style="font-size:0.75rem; color:#616161; margin-top:2px;">'
            'Penalty-based diverse routing'
            '</div></div>'
        )
        col.markdown(html, unsafe_allow_html=True)
# ─────────────────────────────────────────────
# Tab 5 — RL Training
# ─────────────────────────────────────────────
def plot_reward_curve(rewards: np.ndarray):
    episodes = list(range(1, len(rewards) + 1))
    window   = min(10, len(rewards))
    ma       = np.convolve(rewards, np.ones(window) / window, mode="valid")
    ma_ep    = list(range(window, len(rewards) + 1))
    # หา convergence episode = episode แรกที่ moving avg > 80th percentile ของ ma
    converge_ep = None
    if len(ma) > 0:
        threshold = np.percentile(ma, 80)
        idx = np.where(ma >= threshold)[0]
        if len(idx) > 0:
            converge_ep = ma_ep[idx[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=episodes, y=rewards.tolist(), mode="lines",
                             line=dict(color="#90caf9", width=1.5),
                             name="Episode reward", opacity=0.7))
    fig.add_trace(go.Scatter(x=ma_ep, y=ma.tolist(), mode="lines",
                             line=dict(color="#e53935", width=2.5),
                             name=f"Moving avg (window={window})"))
    if converge_ep:
        fig.add_vline(x=converge_ep, line_dash="dash", line_color="#ffd600",
                      annotation_text=f"Converge ~ep.{converge_ep}",
                      annotation_position="top right",
                      annotation_font_color="#ffd600")
    fig.update_layout(
        title="DQN Training — Reward per Episode",
        xaxis_title="Episode", yaxis_title="Total Reward",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
        height=420,
    )
    return fig, converge_ep
def render_rl_explanation(rewards: np.ndarray, converge_ep):
    """คำอธิบาย reward สำหรับคนฟังที่ไม่คุ้นเคย RL"""
    best  = float(rewards.max())
    last  = float(rewards[-1])
    worst = float(rewards.min())
    st.markdown("#### ทำความเข้าใจ Reward")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Episodes",    len(rewards))
    c2.metric("Best Reward",       f"{best:.4f}")
    c3.metric("Last Reward",       f"{last:.4f}")
    c4.metric("Converge ~Episode", converge_ep if converge_ep else "N/A")
    st.markdown(
        f"""
        **Reward Interpretation**
        - Reward ใกล้ **0** = agent หาเส้นทางได้สั้นมาก (ดีที่สุด)
        - Reward ลบมาก = agent วนหลายขั้นตอนหรือ timeout
        - Best reward = **{best:.4f}** หมายถึง agent ใช้ขั้นตอนน้อยมากในการถึงเป้าหมาย
        - Moving average **เพิ่มขึ้น** จาก {worst:.3f} → {best:.3f} แสดงว่า agent เรียนรู้ได้จริง
        {f"- Agent เริ่ม converge ที่ประมาณ **Episode {converge_ep}**" if converge_ep else ""}
        > HDC state encoding (10,000 มิติ) ช่วยให้ DQN agent แยกแยะ edge ที่คล้ายกันได้แม่นยำขึ้น
        > เทียบกับ one-hot encoding ที่ขนาด state space ใหญ่เกินไป
        """
    )
# ─────────────────────────────────────────────
# Tab 6 — HDC Similarity Heatmap (ใหม่)
# ─────────────────────────────────────────────
def plot_hdc_heatmap(hv_dict: dict, n_samples: int = 30):
    """
    แสดง cosine similarity matrix ระหว่าง edge hypervectors
    เลือก n_samples edges แบบ random → matrix n×n
    """
    keys = list(hv_dict.keys())
    if len(keys) < 2:
        return None
    sample_keys = random.sample(keys, min(n_samples, len(keys)))
    vecs = np.stack([hv_dict[k].astype(np.float32) for k in sample_keys])
    # cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    vecs_n = vecs / norms
    sim    = vecs_n @ vecs_n.T  # (n, n)
    # labels สั้นๆ
    labels = [f"e{i}" for i in range(len(sample_keys))]
    fig = go.Figure(data=go.Heatmap(
        z=sim,
        x=labels, y=labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        colorbar=dict(title="Cosine Similarity"),
    ))
    fig.update_layout(
        title=f"HDC Edge Hypervector Similarity ({D:,}-dim) — {len(sample_keys)} sample edges",
        xaxis=dict(tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8)),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
        height=500,
    )
    return fig, sim, sample_keys
def render_hdc_stats(sim: np.ndarray):
    """สถิติ similarity ที่น่าสนใจ"""
    # off-diagonal เท่านั้น
    mask    = ~np.eye(sim.shape[0], dtype=bool)
    off_sim = sim[mask]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HDC Dimensions",      f"{D:,}")
    c2.metric("Avg Similarity",       f"{off_sim.mean():.4f}",
              help="ใกล้ 0 = edges แยกจากกันได้ดี")
    c3.metric("Max Off-diag Sim",     f"{off_sim.max():.4f}")
    c4.metric("% Near-orthogonal",
              f"{(np.abs(off_sim) < 0.1).mean()*100:.1f}%",
              help="คู่ที่ similarity < 0.1 ถือว่า near-orthogonal (แยกกันได้ดีมาก)")
    st.markdown(
        f"""
        **ทำไม HDC {D:,} มิติถึงสำคัญ**
        - Hypervector แต่ละมิติเป็น random ±1 → vectors คู่ใดก็แทบจะ
          **orthogonal กัน** (similarity ≈ 0) ด้วยความน่าจะเป็นสูง
        - กราฟแสดงว่า {(np.abs(off_sim) < 0.1).mean()*100:.1f}% ของ edge pairs
          มี |similarity| < 0.1 → DQN state encoder แยกแยะถนนแต่ละสายได้แม่นยำ
        - ถ้าใช้ {D:,} มิติน้อยกว่า เช่น 128 มิติ จำนวน near-orthogonal pairs
          จะลดลงมาก → agent สับสนระหว่าง edge ที่คล้ายกัน
        """
    )
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Presentation CSS (inject once)
# ─────────────────────────────────────────────
# Font import URL — fetched at runtime
_FONT_URL = (
    "https://api.fontshare.com/v2/css?f[]=cabinet-grotesk@400,700,800,900"
    "&f[]=satoshi@400,500,700&display=swap"
)
def _inject_present_css() -> None:
    """Inject presentation CSS into the Streamlit page via JS parent-window injection.
    Uses components.html with a JS snippet that appends <style> to window.parent.document.head,
    which bypasses Streamlit's HTML sanitizer that strips <style> from st.markdown.
    """
    css_file = Path(__file__).with_name("presentation.css")
    raw_css = css_file.read_text(encoding="utf-8") if css_file.exists() else ""
    # Escape backticks for JS template literal
    raw_css_js = raw_css.replace("\\", "\\\\").replace("`", "\\`")
    inject_html = (
        "<script>\n"
        "(function(){\n"
        "  var p = window.parent || window;\n"
        "  if(p.document.getElementById('voyza-pres-css')) return;\n"
        "  var lnk = p.document.createElement('link');\n"
        f"  lnk.href = '{_FONT_URL}';\n"
        "  lnk.rel = 'stylesheet'; lnk.id = 'voyza-pres-font';\n"
        "  p.document.head.appendChild(lnk);\n"
        "  var s = p.document.createElement('style');\n"
        "  s.id = 'voyza-pres-css';\n"
        f"  s.textContent = `{raw_css_js}`;\n"
        "  p.document.head.appendChild(s);\n"
        "})();\n"
        "</script>"
    )
    components.html(inject_html, height=0, scrolling=False)
def render_presentation_tab(G_real, G_abs):
    """Tab พิเศษสำหรับ Presentation — ไม่แตะ logic เดิม"""
    # Inject presentation CSS via JS into parent window head
    # (bypasses Streamlit's HTML sanitizer that strips <style> from st.markdown)
    _inject_present_css()
    n_real  = G_real.number_of_nodes()
    n_abs   = G_abs.number_of_nodes()
    e_real  = G_real.number_of_edges()
    e_abs   = G_abs.number_of_edges()
    node_compression = (1 - n_abs / n_real) * 100 if n_real > 0 else 0
    edge_compression = (1 - e_abs / e_real) * 100 if e_real > 0 else 0
    # ── HERO ──────────────────────────────────────
    st.markdown("""
    <div class="pres-wrap">
    <div class="pres-hero">
      <div class="pres-badge"><span class="pres-badge-dot"></span>Bangkok Traffic Routing System</div>
      <div class="pres-title">
        Smart Routing via<br>
        <span class="pres-gradient">Blockchain Graph &amp; AI</span>
      </div>
      <p class="pres-sub">
      <p class="pres-sub" style="text-align:center;margin:0 auto 1.4rem;max-width:580px;">
        ระบบนำทางกรุงเทพฯ ที่ใช้ Blockchain Graph Abstraction ลดโหนดถนน 97%
        ร่วมกับ Hyperdimensional Computing 10,000 มิติ และ DQN Reinforcement Learning
        เพื่อหา 3 เส้นทางที่เหมาะสมและแตกต่างกัน
      </p>
      <div class="pres-tech-row">
        <span class="pres-tech">OSMnx / QGIS</span>
        <span class="pres-tech">Blockchain Graph</span>
        <span class="pres-tech">HDC · D=10,000</span>
        <span class="pres-tech">DQN RL</span>
        <span class="pres-tech">Folium / Streamlit</span>
        <span class="pres-tech">Bangkok Roads</span>
      </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    # Live stat cards (from real graph data)
    c1, c2, c3, c4 = st.columns(4)
    for col, cls, val, lbl, sub in [
        (c1, "kpi-c",  f"{node_compression:.1f}%", "Node Compression",   f"{n_real:,} → {n_abs:,} nodes"),
        (c2, "kpi-p",  "10,000",                   "HDC Dimensions",     "Bipolar ±1 encoding"),
        (c3, "kpi-g",  "3",                        "Diverse Routes",     "Penalty-based K-shortest"),
        (c4, "kpi-gr", f"{n_real:,}",              "Graph Nodes (Real)", f"{e_real:,} edges"),
    ]:
        with col:
            st.markdown(
                f'<div class="kpi-card {cls}">'
                f'<div class="kpi-val">{val}</div>'
                f'<div class="kpi-lbl">{lbl}</div>'
                f'<div class="kpi-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    st.markdown("<hr style='border:none;border-top:1px solid #21262d;margin:2rem 0'>", unsafe_allow_html=True)
    # ── ARCHITECTURE ───────────────────────────────
    st.markdown('<div class="sec-label">System Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">4 Pillars of the System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-desc">ออกแบบให้แต่ละส่วนทำงานร่วมกันอย่างมีประสิทธิภาพ ตั้งแต่ประมวลผลข้อมูลถนนจริงจนถึงการตัดสินใจแบบ AI</div>', unsafe_allow_html=True)
    def pillar(num, pcls, icon, lbl, title, desc, tags):
        tag_html = "".join(f'<span class="pillar-tag">{t}</span>' for t in tags)
        st.markdown(
            f'<div class="pillar-card {pcls}">'
            f'<div class="pillar-accent"></div>'
            f'<span class="pillar-num">{num}</span>'
            f'<div class="pillar-icon">{icon}</div>'
            f'<div class="pillar-lbl">{lbl}</div>'
            f'<div class="pillar-title">{title}</div>'
            f'<p class="pillar-desc">{desc}</p>'
            f'<div class="pillar-tags">{tag_html}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        pillar("01", "p-cyan", "🔗", "Layer 1", "Blockchain Graph Abstraction",
               f"แปลง Road Graph จาก QGIS/OSMnx ที่มี {n_real:,} โหนด ให้กลายเป็น Blockchain-style "
               f"Abstract Graph เพียง {n_abs:,} โหนด ({node_compression:.1f}% compression) "
               "โดยจัดกลุ่มถนนในบริเวณใกล้เคียงกันเป็น Block เพื่อเพิ่มความเร็วค้นหาเส้นทาง",
               [f"{n_real:,} → {n_abs:,} nodes", f"{node_compression:.1f}% compression", "NetworkX", "EPSG:4326"])
    with col2:
        pillar("02", "p-purple", "🧠", "Layer 2", "Hyperdimensional Computing",
               "เข้ารหัส Edge แต่ละเส้นในกราฟด้วย Hypervector ขนาด 10,000 มิติ "
               "ใช้ฟีเจอร์ความยาวถนน (Haversine), จำนวนอาคารใกล้เคียง และมุมถนน "
               "ค่า Cosine similarity บอกความสัมพันธ์ระหว่างเส้นทาง",
               ["D = 10,000", "Bipolar ±1", "Cosine similarity", "3 features / edge"])
    col3, col4 = st.columns(2, gap="medium")
    with col3:
        pillar("03", "p-gold", "🤖", "Layer 3", "DQN Reinforcement Learning",
               "Deep Q-Network Agent เรียนรู้การนำทางบน Bangkok Road Graph "
               "รับ state จาก HDC encoding เลือก action (node ถัดไป) รับ reward จากระยะทางที่ลดลง "
               "ฝึก replay buffer ให้ converge",
               ["Experience Replay", "ε-greedy", "Target Network", "PyTorch"])
    with col4:
        pillar("04", "p-green", "🗺️", "Layer 4", "OSMnx + QGIS Data",
               "โหลดข้อมูลถนนกรุงเทพฯ จาก Roads.geojson (366,029 แถว) และ Buildings.geojson (187,909 แถว) "
               "แปลง CRS จาก EPSG:32647 เป็น EPSG:4326 พร้อม spatial sampling ด้วย 10×10 grid",
               ["366K road rows", "187K buildings", "EPSG:4326", "STRtree join"])
    st.markdown("<hr style='border:none;border-top:1px solid #21262d;margin:2rem 0'>", unsafe_allow_html=True)
    # ── GRAPH COMPRESSION VISUAL ───────────────────
    st.markdown('<div class="sec-label">Performance Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Graph Compression Results</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")
    with left:
        fig_bar = go.Figure(data=[
            go.Bar(name="Nodes",
                   x=["Real Graph", "Blockchain Graph"],
                   y=[n_real, n_abs],
                   marker_color=["#4a5568", "#4f98a3"],
                   text=[f"{n_real:,}", f"{n_abs:,}"],
                   textposition="auto"),
            go.Bar(name="Edges",
                   x=["Real Graph", "Blockchain Graph"],
                   y=[e_real, e_abs],
                   marker_color=["#2d3748", "#2c7a85"],
                   text=[f"{e_real:,}", f"{e_abs:,}"],
                   textposition="auto"),
        ])
        fig_bar.update_layout(
            barmode="group", title="Node & Edge Compression",
            plot_bgcolor="#0e1117", paper_bgcolor="#161b22",
            font_color="#e2e8f0", height=320,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(bgcolor="#161b22", borderwidth=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown(
            f'<div class="kpi-card kpi-c" style="margin-bottom:.75rem">'
            f'<div class="kpi-val">{node_compression:.1f}%</div>'
            f'<div class="kpi-lbl">Node Compression Rate</div>'
            f'<div class="kpi-sub">{n_real:,} → {n_abs:,} nodes</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="kpi-card kpi-p" style="margin-bottom:.75rem">'
            f'<div class="kpi-val">{edge_compression:.1f}%</div>'
            f'<div class="kpi-lbl">Edge Compression Rate</div>'
            f'<div class="kpi-sub">{e_real:,} → {e_abs:,} edges</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        avg_deg_real = (2 * e_real / n_real) if n_real > 0 else 0
        avg_deg_abs  = (2 * e_abs  / n_abs)  if n_abs  > 0 else 0
        st.markdown(
            f'<div class="kpi-card kpi-g" style="margin-bottom:.75rem">'
            f'<div class="kpi-val">{avg_deg_real:.2f}</div>'
            f'<div class="kpi-lbl">Avg Degree — Real Graph</div>'
            f'<div class="kpi-sub">vs Blockchain: {avg_deg_abs:.2f}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="kpi-card kpi-gr">'
            '<div class="kpi-val">3</div>'
            '<div class="kpi-lbl">Diverse Routes Output</div>'
            '<div class="kpi-sub">Penalty factor = 10×</div>'
            '</div>',
            unsafe_allow_html=True
        )
    # ── PIPELINE ────────────────────────────────────
    st.markdown(
        '<div class="pipe-wrap">'
        '<div class="pipe-hd">Full System Pipeline</div>'
        '<div class="pipe-steps">'
        '<div class="pipe-step"><div class="pipe-ico">🗺️</div><div class="pipe-lbl">QGIS / OSMnx Data Load</div></div>'
        '<div class="pipe-step"><div class="pipe-ico">🔗</div><div class="pipe-lbl">Build Road Graph (366K rows)</div></div>'
        '<div class="pipe-step"><div class="pipe-ico">📦</div><div class="pipe-lbl">Blockchain Abstraction</div></div>'
        '<div class="pipe-step"><div class="pipe-ico">🧠</div><div class="pipe-lbl">HDC Edge Encoding (10K dim)</div></div>'
        '<div class="pipe-step"><div class="pipe-ico">🤖</div><div class="pipe-lbl">DQN RL Training</div></div>'
        '<div class="pipe-step"><div class="pipe-ico">🛤️</div><div class="pipe-lbl">K=3 Route Output</div></div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.caption("Voyza Traffic Ai — Blockchain Graph + HDC 10K + DQN RL | Bangkok Road Network")
def main():
    st.set_page_config(page_title="Voyza Traffic Ai", layout="wide", page_icon="🚦")
    # ── Sidebar: branding + mode toggle ──────────────
    with st.sidebar:
        st.markdown(
            "<div style='font-family:sans-serif;font-size:1.05rem;font-weight:800;"
            "color:#e2e8f0;margin-bottom:.25rem'>🚦 Voyza Traffic Ai</div>"
            "<div style='font-size:.75rem;color:#4f98a3;margin-bottom:1.5rem'>"
            "Bangkok Road Network</div>",
            unsafe_allow_html=True
        )
        present_mode = st.toggle("🎯 Presentation Mode", value=False)
        if present_mode:
            st.info("Presentation tab จะแสดง overview สวยงาม \nTabs อื่นยังใช้งานได้ปกติ")
        st.divider()
        st.caption("Blockchain Graph + HDC 10K + DQN RL")
    # ── Header (ย่อเล็กลง) ───────────────────────────
    st.markdown(
        "<h3 style='margin-bottom:.1rem;font-size:1.3rem'>Traffic Blockchain-Style Graph Dashboard</h3>",
        unsafe_allow_html=True
    )
    st.caption("Blockchain-style graph abstraction · HDC 10,000-dim · DQN Reinforcement Learning")
    with st.spinner("Loading data & building graphs..."):
        roads_display, buildings, roads_feat, G_real, G_real_conn, G_abs, node_to_block = load_data_and_graphs()
    node_index = build_spatial_index(G_real_conn)
    # ── Tab list (เพิ่ม Presentation tab หน้าสุด) ───
    tab_labels = [
        "🗺 Map", "📊 Graph Stats", "⚡ Path Comparison",
        "🧭 Dynamic Routing", "🤖 RL Training", "🧠 HDC Encoding",
    ]
    if present_mode:
        tab_labels = ["🎯 Presentation"] + tab_labels
    tabs = st.tabs(tab_labels)
    # ── Index offset ─────────────────────────────────
    off = 1 if present_mode else 0
    if present_mode:
        with tabs[0]:
            render_presentation_tab(G_real_conn, G_abs)
    tab_map      = tabs[0 + off]
    tab_graph    = tabs[1 + off]
    tab_path     = tabs[2 + off]
    tab_routing  = tabs[3 + off]
    tab_rl       = tabs[4 + off]
    tab_hdc      = tabs[5 + off]
    # ══════════════════════════════
    # Tab 1 — Map
    # ══════════════════════════════
    with tab_map:
        st.subheader("Road Network + Buildings (Folium)")
        st.info("ถนน = เส้นสีน้ำเงิน | ตึก = polygon สีแดง")
        m = plot_roads_buildings_map_folium(roads_display, buildings)
        st_folium(m, width=None, height=600, returned_objects=[])
    # ══════════════════════════════
    # Tab 2 — Graph Stats
    # ══════════════════════════════
    with tab_graph:
        st.subheader("Graph Size Comparison")
        st.plotly_chart(plot_graph_sizes(G_real, G_abs), use_container_width=True)
        col1, col2 = st.columns(2)
        col1.metric("Real Graph — Nodes", f"{G_real.number_of_nodes():,}")
        col1.metric("Real Graph — Edges", f"{G_real.number_of_edges():,}")
        col2.metric("Blockchain Graph — Nodes", f"{G_abs.number_of_nodes():,}")
        col2.metric("Blockchain Graph — Edges", f"{G_abs.number_of_edges():,}")
        conn_nodes = G_real_conn.number_of_nodes()
        conn_pct   = conn_nodes / G_real.number_of_nodes() * 100 if G_real.number_of_nodes() > 0 else 0
        st.caption(f"Largest Connected Component: {conn_nodes:,} nodes ({conn_pct:.1f}%) — ใช้สำหรับ routing")
        st.divider()
        render_graph_metrics(G_real, G_abs)
    # ══════════════════════════════
    # Tab 3 — Shortest Path Comparison
    # ══════════════════════════════
    with tab_path:
        st.subheader("Shortest Path: Real Graph vs Blockchain Graph")
        st.markdown(
            f"สุ่ม **{N_SAMPLES_SHORTEST_PATH} คู่ OD** แบบ random จาก largest connected component "
            "แล้วเปรียบเทียบเวลาและความยาวเส้นทางระหว่าง 2 graph"
        )
        if st.button("Run Experiment", key="run_sp"):
            with st.spinner(f"Computing {N_SAMPLES_SHORTEST_PATH} pairs..."):
                stats = compute_shortest_path_stats(G_real_conn, G_abs, node_to_block)
            if stats is None:
                st.warning("ไม่มีคู่ที่หา path ได้")
            else:
                render_path_metrics(stats)
                st.plotly_chart(plot_time_comparison(stats), use_container_width=True)
                with st.expander("Raw stats"):
                    st.json(stats)
    # ══════════════════════════════
    # Tab 4 — Dynamic Routing
    # ══════════════════════════════
    with tab_routing:
        st.subheader("Dynamic Routing — K=3 Diverse Shortest Paths")
        st.markdown("ใส่ **Latitude / Longitude** แล้วกด **Find Routes**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**จุดเริ่มต้น (Origin)**")
            c1, c2 = st.columns(2)
            orig_lat = c1.number_input("Latitude",  value=13.7456, format="%.6f",
                                       min_value=-90.0,  max_value=90.0,  key="orig_lat")
            orig_lon = c2.number_input("Longitude", value=100.5342, format="%.6f",
                                       min_value=-180.0, max_value=180.0, key="orig_lon")
        with col_b:
            st.markdown("**จุดหมาย (Destination)**")
            c3, c4 = st.columns(2)
            dest_lat = c3.number_input("Latitude",  value=13.6900, format="%.6f",
                                       min_value=-90.0,  max_value=90.0,  key="dest_lat")
            dest_lon = c4.number_input("Longitude", value=100.7501, format="%.6f",
                                       min_value=-180.0, max_value=180.0, key="dest_lon")
        st.caption("ตัวอย่าง: สยาม = 13.7456, 100.5342 | สนามบินสุวรรณภูมิ = 13.6900, 100.7501")
        if st.button("Find Routes", key="find_routes"):
            if abs(orig_lat - dest_lat) < 1e-6 and abs(orig_lon - dest_lon) < 1e-6:
                st.warning("จุดเริ่มต้นและจุดหมายเป็นตำแหน่งเดียวกัน")
            else:
                st.info(f"Origin: {orig_lat:.6f}, {orig_lon:.6f} → Dest: {dest_lat:.6f}, {dest_lon:.6f}")
                with st.spinner("Finding K=3 diverse routes (penalty-based)..."):
                    routes_info = get_routes_from_latlon(
                        G_real_conn, node_index,
                        src_lat=orig_lat, src_lon=orig_lon,
                        dst_lat=dest_lat, dst_lon=dest_lon,
                        k=3, weight="length",
                    )
                if not routes_info["routes"]:
                    st.warning("ไม่พบเส้นทาง")
                else:
                    st.success(f"พบ {len(routes_info['routes'])} เส้นทางที่แตกต่างกัน")
                    # Route Quality Cards
                    render_route_cards(routes_info["routes"])
                    # Debug snap (เล็ก ไม่รกสายตา)
                    src_node = routes_info["src_node"]
                    dst_node = routes_info["dst_node"]
                    slat, slon = _node_latlon(G_real_conn, src_node)
                    dlat, dlon = _node_latlon(G_real_conn, dst_node)
                    if slat and dlat:
                        st.caption(
                            f"Snap — origin: {slat:.5f}, {slon:.5f} | "
                            f"dest: {dlat:.5f}, {dlon:.5f}"
                        )
                    # Folium map
                    m_routes = build_routes_map(G_real_conn, routes_info)
                    if m_routes:
                        st_folium(m_routes, width=None, height=520, returned_objects=[])
    # ══════════════════════════════
    # Tab 5 — RL Training
    # ══════════════════════════════
    with tab_rl:
        st.subheader("DQN Reinforcement Learning")
        rewards_path = ROOT / "data" / "rewards_dqn.npy"
        if not rewards_path.exists():
            st.info("ยังไม่มีไฟล์ `data/rewards_dqn.npy`\n\nรัน: `python scripts/run_train_dqn.py`")
        else:
            rewards = np.load(str(rewards_path))
            fig_rl, converge_ep = plot_reward_curve(rewards)
            render_rl_explanation(rewards, converge_ep)
            st.plotly_chart(fig_rl, use_container_width=True)
            if st.checkbox("Show raw reward values"):
                st.dataframe({"episode": list(range(1, len(rewards)+1)),
                              "reward": rewards.tolist()})
    # ══════════════════════════════
    # Tab 6 — HDC Encoding (ใหม่)
    # ══════════════════════════════
    with tab_hdc:
        st.subheader(f"HDC Edge Encoding — {D:,} Dimensions")
        st.markdown(
            "แต่ละ edge ในกราฟถูก encode เป็น **hypervector ขนาด 10,000 มิติ** "
            "ด้วยการ bind และ bundle ของ node hypervectors "
            "ซึ่งใช้เป็น state input ของ DQN agent"
        )
        n_slider = st.slider("จำนวน edges ที่ sample สำหรับ heatmap",
                             min_value=10, max_value=50, value=30, step=5)
        if st.button("Generate HDC Heatmap", key="hdc_btn"):
            with st.spinner("Building edge hypervectors..."):
                hv_dict = build_hdc_dict(G_real_conn)
            if not hv_dict:
                st.warning("ไม่สามารถสร้าง hypervectors ได้ — graph ไม่มี edges")
            else:
                result = plot_hdc_heatmap(hv_dict, n_samples=n_slider)
                if result:
                    fig_hdc, sim, sample_keys = result
                    render_hdc_stats(sim)
                    st.plotly_chart(fig_hdc, use_container_width=True)
                    st.caption(
                        f"Diagonal = 1.0 (self-similarity) | "
                        f"Off-diagonal ≈ 0 = near-orthogonal → "
                        f"edges แยกแยะกันได้ดี"
                    )
if __name__ == "__main__":
    main()
