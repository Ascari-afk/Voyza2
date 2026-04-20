import numpy as np
import networkx as nx
from pyproj import Transformer
from .config import GRID_SIZE_METERS, MAX_DEGREE

def project_nodes_to_meters(G):
    print("  [graph_abs] projecting node coordinates to EPSG:32647...")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)
    G_proj = nx.Graph()

    for n, data in G.nodes(data=True):
        lon = data["x"]
        lat = data["y"]
        x, y = transformer.transform(lon, lat)
        G_proj.add_node(n, x=x, y=y)

    for u, v, data in G.edges(data=True):
        G_proj.add_edge(u, v, **data)

    print("  [graph_abs] projected graph nodes:", G_proj.number_of_nodes(), "edges:", G_proj.number_of_edges())
    return G_proj


def build_blockchain_style_graph(G_real, grid_size=GRID_SIZE_METERS):
    print("  [graph_abs] building blockchain-style abstract graph...")
    G_proj = project_nodes_to_meters(G_real)

    nodes, data = zip(*G_proj.nodes(data=True))
    xs = np.array([d["x"] for d in data])
    ys = np.array([d["y"] for d in data])

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    def get_block_id(x, y):
        gx = int((x - x_min) // grid_size)
        gy = int((y - y_min) // grid_size)
        return (gx, gy)

    node_to_block = {n: get_block_id(d["x"], d["y"]) for n, d in G_proj.nodes(data=True)}

    G_abs = nx.Graph()
    for block_id in set(node_to_block.values()):
        G_abs.add_node(block_id)

    for u, v, d in G_proj.edges(data=True):
        bu = node_to_block[u]
        bv = node_to_block[v]
        if bu == bv:
            continue
        length = d.get("length", 1.0)
        if G_abs.has_edge(bu, bv):
            G_abs[bu][bv]["weight"] = min(G_abs[bu][bv]["weight"], length)
        else:
            G_abs.add_edge(bu, bv, weight=length)

    print("  [graph_abs] abstract graph nodes:", G_abs.number_of_nodes(), "edges:", G_abs.number_of_edges())
    return G_abs, node_to_block


def apply_degree_constraint(G_abs, max_degree=MAX_DEGREE):
    print("  [graph_abs] applying degree constraint...", "max_degree =", max_degree)
    G_constrained = G_abs.copy()
    for node in list(G_constrained.nodes()):
        while G_constrained.degree(node) > max_degree:
            edges = list(G_constrained.edges(node, data=True))
            if not edges:
                break
            edge_to_remove = max(edges, key=lambda e: e[2].get("weight", 1.0))
            G_constrained.remove_edge(edge_to_remove[0], edge_to_remove[1])
    print("  [graph_abs] degree-constrained graph nodes:", G_constrained.number_of_nodes(), "edges:", G_constrained.number_of_edges())
    return G_constrained