import os
import sys
from pprint import pprint
import sumolib

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
NET_PATH = os.path.join(PROJECT_ROOT, "envs", "grid2x2", "net.net.xml")

def build_tls_graph(net_path):
    net = sumolib.net.readNet(net_path)

    # Map: tls_id -> node_id(s)
    tls_nodes = {}  # TLS to controlling nodes
    node_tls = {}   # node_id -> tls_id (if controlling)

    for node in net.getNodes():
        if node.getType() == "traffic_light":
            tls_id = node.getID()
            tls_nodes[tls_id] = [tls_id]  # simple 1:1 mapping
            node_tls[tls_id] = tls_id

    graph = {tid: set() for tid in tls_nodes.keys()}

    # For each edge, check if both ends are controlled by TLS
    for edge in net.getEdges():
        from_node = edge.getFromNode().getID()
        to_node   = edge.getToNode().getID()

        if from_node in node_tls and to_node in node_tls:
            a = node_tls[from_node]
            b = node_tls[to_node]
            if a != b:
                graph[a].add(b)
                graph[b].add(a)

    return {k: list(v) for k, v in graph.items()}


def main():
    print(f"Reading net: {NET_PATH}")
    graph = build_tls_graph(NET_PATH)

    print("\nTLS Graph (neighbors):")
    pprint(graph)


if __name__ == "__main__":
    main()

