import numpy as np
def assert_network_close(actual, desired, atol=0, rtol=1e-7):
    """Asserts if the given two ``Networks`` are not close. For two
    ``Networks`` to be close  there needs to be a one-to-one map between their
    nodes such that the tensors that these nodes represent are close to each
    other. These ``Networks`` must also have same structure regarding to its
    connections.

    Parameters:
    -----------
    actual : list of nodes
        ``Network`` obtained.

    desired : list of nodes
        Desired ``Network``.

    atol: float, optional
        Relative tolerance.

    rtol: float, optional
        Absolute tolerance.

    Raises
    ------
    AssertionError
        If actual and desired do not represent the same map.
    ValueError
        If more than one node in the network is close to each other.

    See Also
    --------
    numpy.testing.assert_allclose

    Notes
    -----
    The implementation of this function is not optimal as it transverses the
    graph multiple times. Furthermore, the current implementation is not able
    to handle the case where more than two nodes in the network are the same.
    This is a severe limitation that will be addressed in later updates.

    In its currect form this function does not check the dangling edges either.
    """
    # We first find a bijective map, if any, between nodes.
    node_dict = _search_map(actual.nodes, desired.nodes, atol, rtol)

    # We now compare Edges to see if they are properly connected.
    for node in actual.nodes:
        for edge in node.edges:
            if not edge.is_dangling():
                # This is a very basic test. We need to improve it to check
                # whether the two edges are really the same: same dimentions
                # and connected to same index.
                assert_no_edge_between(node_dict[edge.node1], node_dict[edge.node2])

def assert_no_edge_between(node1, node2):
    """Raise an AssertionError if the given nodes do not have an edge connectin
    them."""
    for edge in node1.edges:
        if edge in node2.edges:
            return None

    raise AssertionError("Node 1 "+ str(node1) + " and Node 2 " + str(node2) +
                         " have no edges in common.")

def _search_map(nodes_actual, nodes_desired, atol, rtol):
    """It attempts to create a map between the nodes of two networks such
    that the mapped nodes have tensors that are numerically close.

    Raises
    ------
    AssertionError
        If actual and desired do not represent the same map.
    ValueError
        If more than one node in the network is close to each other. Hence, no
        bijection was possible.
    """

    node_dict = {}
    for node in nodes_actual:
        node_dict[node] = _search_close(node, nodes_desired, atol, rtol)

    return node_dict

def _search_close(node_actual, nodes_desired, atol, rtol):
    """Search `nodes_desired` to find which nodes are close to the given
    `node_actual`.
    """
    found = []
    for node in nodes_desired:
        if np.allclose(node.tensor, node_actual.tensor, atol=atol, rtol=rtol):
            found.append(node)

    if len(found) == 0:
        raise AssertionError("No node was found in desired nodes that was"
                             "close to " + str(node_actual))
    if len(found) > 1:
        raise ValueError("Multiple nodes found that were close to"
                             + str(node_actual))
    return found[0]

