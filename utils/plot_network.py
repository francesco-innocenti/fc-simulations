import numpy as np
import networkx as nx


def plot_network(adj_net):
    """Plots graph network from adjacency matrix.

    Args:
        adj_net (array): (2D) matrix representing graph/network.

    """
    if not isinstance(adj_net, np.ndarray):
        adj_net = np.array(adj_net)

    if len(adj_net.shape) != 2:
        raise ValueError("Adjacency matrix must have two dimensions")

    g = nx.convert_matrix.from_numpy_matrix(adj_net)
    labels = {n: n + 1 for n in range(len(adj_net))}
    nx.draw(g, arrows=True, with_labels=True, labels=labels, node_size=1400,
            node_color='red', alpha=0.8, font_size=15, edgecolors='black',
            arrowsize=12, pos=nx.circular_layout(g))
