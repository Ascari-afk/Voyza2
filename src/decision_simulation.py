"""
src/decision_simulation.py
Decision Simulation tab for industrial logistics.
"""
import time
import plotly.graph_objects as go
import streamlit as st
import folium
from streamlit_folium import st_folium

from .industrial_data import (
    WAREHOUSES, DELIVERY_ZONES,
    generate_normal_orders, generate_surge_orders,
    prioritize_orders, orders_to_df, compute_kpi,
)

PRIORITY_COLOR = {
    "CRITICAL": "#ef4444",
    "HIGH":     "#f97316",
    "MEDIUM":   "#eab308",
    "LOW":      "#22c55e",
}

def build_industrial_map(orders=None):
    m = folium.Map(location=[13.756, 100.592], zoom_start=11,
                   tiles="CartoDB dark_matter")
    # Warehouses
    for wh in WAREHOUSES:
        folium.Marker(
            location=[wh["lat"], wh["lon"]],
            popup=f"<b>{wh['name']}</b><br>Capacity: {wh['capacity']}",
            tooltip=wh["name"],
            icon=folium.Icon(color="blue", icon="home"),
        ).add_to(m)
    # Delivery zones
    demand_color = {"high": "red", "medium": "orange", "low": "green"}
    for dz in DELIVERY_ZONES:
        folium.CircleMarker(
            location=[dz["lat"], dz["lon"]],
            radius=9,
            color=demand_color.get(dz["demand"], "gray"),
            fill=True,
            fill_opacity=0.6,
            popup=f"<b>{dz['name']}</b><br>Demand: {dz['demand'].upper()}",
            tooltip=dz["name"],
        ).add_to(m)
    # Order flows (top 5 urgent)
    if orders:
        wh_map = {w["id"]: w for w in WAREHOUSES}
        dz_map = {d["id"]: d for d in DELIVERY_ZONES}
        for o in prioritize_orders(orders)[:5]:
            wh = wh_map.get(o.warehouse_id)
            dz = dz_map.get(o.delivery_zone_id)
            if wh and dz:
                folium.PolyLine(
                    locations=[[wh["lat"], wh["lon"]], [dz["lat"], dz["lon"]]],
                    color=PRIORITY_COLOR.get(o.priority, "#888"),
                    weight=3,
                    opacity=0.85,
                    tooltip=f"{o.order_id} | {o.priority} | {o.weight_kg}kg",
                ).add_to(m)
    return m

def render_kpi_cards(kpi, label: str):
    st.markdown(f"#### 📊 KPI Summary — {label}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Orders",      kpi.total_orders)
    c2.metric("🚨 Critical",       kpi.critical_orders)
    c3.metric("⚠️ At Risk",        kpi.at_risk_orders,
              delta=f"{kpi.at_risk_orders} orders อาจไม่ทัน" if kpi.at_risk_orders else None,
              delta_color="inverse")
    c4.metric("Fleet Required",    f"{kpi.estimated_fleet_size} คัน")
    c5.metric("Avg Urgency Score", kpi.avg_urgency_score)

def render_simulation_result(kpi, label: str):
    st.markdown("---")
    st.markdown("#### ⚡ Decision Simulation Result")
    st.caption("Manual Dispatch vs AI-Optimized Routing")
    c1, c2, c3 = st.columns(3)
    c1.metric("Manual Dispatch",  f"{kpi.baseline_avg_time_min:.0f} นาที/delivery")
    c2.metric("AI Routing",       f"{kpi.optimized_avg_time_min:.0f} นาที/delivery",
              delta=f"-{kpi.baseline_avg_time_min - kpi.optimized_avg_time_min:.0f} นาที",
              delta_color="inverse")
    c3.metric("Throughput เพิ่มขึ้น", f"+{kpi.throughput_gain_pct}%")
    fig = go.Figure()
    fig.add_bar(
        x=["Manual Dispatch", "AI Routing"],
        y=[kpi.baseline_avg_time_min, kpi.optimized_avg_time_min],
        marker_color=["#6b7280", "#3b82f6"],
        text=[f"{kpi.baseline_avg_time_min:.0f} min", f"{kpi.optimized_avg_time_min:.0f} min"],
        textposition="outside",
    )
    fig.update_layout(
        title=f"Avg Delivery Time — {label}",
        yaxis_title="นาที",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        height=320,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

def render_industrial_tab():
    st.subheader("🏭 Industrial Logistics Decision Simulation")
    st.markdown("จำลอง **Order Surge** และเปรียบเทียบ Manual Dispatch กับ **AI-Optimized Routing** ก่อน dispatch จริง")

    scenario = st.radio(
        "เลือก Scenario",
        ["🟢 Normal Operations (12 orders)", "🔴 Order Surge (20 orders)"],
        horizontal=True,
    )
    is_surge = "Surge" in scenario

    orders  = generate_surge_orders() if is_surge else generate_normal_orders()
    label   = "Order Surge" if is_surge else "Normal Operations"
    kpi     = compute_kpi(orders, scenario="surge" if is_surge else "normal")

    render_kpi_cards(kpi, label)
    st.markdown("---")

    col_map, col_tbl = st.columns([3, 2])
    with col_map:
        st.markdown("**🗺️ Warehouse & Delivery Zone Map**")
        st.caption("🔵 Warehouse | 🔴 High | 🟠 Medium | 🟢 Low | เส้น = Top-5 urgent routes")
        st_folium(build_industrial_map(orders), width=None, height=420, returned_objects=[])
    with col_tbl:
        st.markdown("**📋 Order Queue (เรียงตาม Urgency Score)**")
        df = orders_to_df(prioritize_orders(orders))
        st.dataframe(df[["Order ID","Priority","Urgency Score","Time Window (hr)","จุดส่ง"]],
                     use_container_width=True, height=420)

    st.markdown("---")
    st.markdown("**🤖 Decision Simulation — จำลองผลก่อน Dispatch**")
    if is_surge:
        st.warning(
            f"⚠️ Order Surge: {kpi.critical_orders} CRITICAL | "
            f"{kpi.at_risk_orders} orders อาจไม่ทัน time window"
        )

    if st.button("▶ Run Simulation", type="primary", key="run_sim"):
        with st.spinner("HDC encoding + route optimization..."):
            prog = st.progress(0)
            for p in range(0, 101, 10):
                time.sleep(0.07)
                prog.progress(p)
            prog.empty()
        render_simulation_result(kpi, label)
        st.success(
            f"✅ Dispatch {kpi.estimated_fleet_size} รถตาม optimized routes — "
            f"ลด {kpi.baseline_avg_time_min - kpi.optimized_avg_time_min:.0f} นาที/delivery "
            f"({kpi.time_saved_pct}%) | Throughput +{kpi.throughput_gain_pct}%"
        )
