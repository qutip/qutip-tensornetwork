from qutip_tensornetwork.core.data.network import Network, _match_edges_by_split
import tensornetwork as tn
from itertools import chain

__all__ = [
    "FiniteTT",
]


class FiniteTT(Network):
    """Represent finite tensor-trains. This can be either MPS or MPO. A
    schematic representation of such tensor trains are:
        |   |   |
        * - * - *
        |   |   |
    where "-" represents an edge and "*" a node. The above graph represents an
    MPO but MPS can also be represented with this class. The dangling edges
    represent the ``in_edges`` and ``out_edges`` (which correspond to rows and
    columns of a matrix when the network is contracted into a 2D array). The
    bonds connecting the nodes can be accessed with ``bond_edges``. Nodes can
    also be accessed in a shorted list format using ``node_list``.

    Notes
    -----
    Nodes in this class have been named from 0 to L, with L the number of
    nodes, as they can be sorted (from left to right). The axes of each node
    have also been named for easy access. These are named as: {"in", "out",
    "rbond", "lbond"}.

    Parameters
    ----------
    out_edges: List of Edges
        The edges of the network to be used as the output edges.

    in_edges: List of Edges
        The edges of the network to be used as the input edges.

    nodes: None or List of Nodes
        Nodes of the network. If None, the nodes are obtained
        by finding all the nodes that belong to the graphs that include
        ``in_edges`` and ``out_edges``.

    copy: bool, default True
        Whether to copy all the ``Nodes``/``Edges`` involved in the
        network. At this moment a copy is always returned.

    Attributes
    ----------
    nodes : set of Nodes
        Nodes that belong to the Network. These can either be reachable from
        in_edges and out_edges or scalar nodes, in which case they have no
        edges.

    train_nodes: List of Nodes
        Nodes of the tensor-train sorted from left to right.
    out_edges : list of Edges
        List of ``Edges`` to be used. When the network is considered as a
        matrix, these edges represent the rows.

    in_edges : list of Edges
        List of ``Edges`` to be used. When the network is considered as a
        matrix, these edges represent the columns.

    bond_edges: List of Edges
        The edges between the nodes of the tensor-train. These are sorted from
        left to right (i.e. ``train_nodes[i]["rbond"] == bond_edges[i]``).

    dims : list of int
        Dimension of the system as a list of lists. dims[0] represents the
        out dimensions whereas dims[1] represents the in dimension.

    shape : tuple of int
        Shape that the matrix would have if the network is represented with a
        matrix.
    """

    def __init__(self, out_edges, in_edges, nodes=None, copy=True):
        out_dims = [e.dimension for e in out_edges]
        in_dims = [e.dimension for e in in_edges]
        if in_dims != out_dims and in_edges and out_edges:
            raise NotImplementedError(
                " At this moment this class can"
                " only represent square matrices, kets"
                " and bras."
            )

        super().__init__(out_edges, in_edges, nodes, copy)
        self._to_tt_format()

    @property
    def train_nodes(self):
        """Return the nodes of the train as an ordered list. The order goes
        from left to right in the tensor-train."""
        if self.in_edges:
            return [edge.node1 for edge in self.in_edges]
        else:
            return [edge.node1 for edge in self.out_edges]

    @property
    def bond_dimension(self):
        """Return a list that represent the dimension of each bond edges from
        left to right."""
        return [e.dimension for e in self.bond_edges]

    @classmethod
    def from_nodes(cls, nodes):
        """Create a tensor-train from a list of nodes.

        By default we assume that the input is a ket (MPS) if the first node has
        two dimension and we assume it is a square operator (MPO) if it has
        three dimensions.

        Parameters
        ----------
        nodes: List of Node
            The nodes are assumed to have the following ranks (MPS):
                nodes[0].shape = (out_dim, bond_dim)
                nodes[1:-1].shape = (out_dim, bond_dim, bond_dim)
                nodes[-1].shape = (out_dim, bond_dim)
            or (MPO):
                nodes[0].shape = (out_dim, in_dim, bond_dim)
                nodes[1:-1].shape = (out_dim, in_dim, bond_dim, bond_dim)
                nodes[-1].shape = (out_dim, in_dim, bond_dim)
        """
        nodes = [tn.Node(node) for node in nodes]

        is_ket = len(nodes[0].shape) == 2
        _check_shape(nodes)

        for i, node in enumerate(nodes):
            node.name = f"node_{i}"

        if is_ket:
            nodes[0].add_axis_names(["out", "rbond"])
            nodes[-1].add_axis_names(["out", "lbond"])
            for node in nodes[1:-1]:
                node.add_axis_names(["out", "lbond", "rbond"])
        else:
            nodes[0].add_axis_names(["out", "in", "rbond"])
            nodes[-1].add_axis_names(["out", "in", "lbond"])
            for node in nodes[1:-1]:
                node.add_axis_names(["out", "in", "lbond", "rbond"])

        for i in range(len(nodes) - 1):
            bond_edge = nodes[i]["rbond"] ^ nodes[i + 1]["lbond"]

        out_edges = [node["out"] for node in nodes]
        in_edges = [] if is_ket else [node["in"] for node in nodes]

        network = cls._fast_constructor(out_edges, in_edges, nodes)
        return network

    @property
    def bond_edges(self):
        """Returns the bond edges as a list sorted from left to right."""
        return [node["rbond"] for node in self.train_nodes[:-1]]

    def _to_tt_format(self):
        """This function is used to transform an arbitrary network into a
        tensor train. This is done by first contracting the whole network into
        a single tensor an then splitting it into a tensor-train by repeatedly
        applying an SVD transformation to the nodes. No truncation is done by
        this function. Hence, the output tensor-train represents exactly the
        input network.

        For a more detailed explanation of the algorithm see [1]_.

        Note that this method was created to be used in the init method of the
        FiniteTT class. It performs an in-place modification of the nodes.

        References
        ---------- .. [1] Paeckel, S., Köhler, T., Swoboda, A., Manmana, S. R.,
        Schollwöck, U., & Hubig, C. (2019). Time-evolution methods for
        matrix-product states.  Annals of Physics, 411, 167998.
        """
        self.contract(copy=False)

        n_nodes = max(len(self.in_edges), len(self.out_edges))
        if self.in_edges:
            in_edges = [[e] for e in self.in_edges]
        else:
            in_edges = [[] for _ in range(n_nodes)]

        if self.out_edges:
            out_edges = [[e] for e in self.out_edges]
        else:
            out_edges = [[] for _ in range(n_nodes)]

        axes_names = ["out"] if self.out_edges else []
        axes_names += ["in"] if self.in_edges else []

        nodes = []
        lbond = []
        for i in range(n_nodes - 1):
            left_edges = out_edges[i]
            left_edges += in_edges[i]
            left_edges += lbond

            right_edges = out_edges[i + 1 :]
            right_edges += in_edges[i + 1 :]
            # We flatten the right edges as it is a list of lists
            right_edges = list(chain(*right_edges))

            node = left_edges[0].node1

            lnode, rnode, _ = tn.split_node(node, left_edges, right_edges)

            rbond = [rnode[0]]
            lnode.name = f"node_{i}"
            lnode.reorder_edges(left_edges + rbond)
            lnode.add_axis_names(axes_names + ["lbond"] * len(lbond) + ["rbond"])
            nodes.append(lnode)

            lbond = rbond

        # For a single node we do not go through the for loop so we accommodate the
        # variables here
        if n_nodes == 1:
            right_edges = out_edges[0]
            right_edges += in_edges[0]
            node = right_edges[0].node1
            node.name = f"node_0"
            node.reorder_edges(right_edges)
            node.add_axis_names(axes_names)
            nodes.append(node)
        # For when the network is actually a scalar
        elif n_nodes == 0:
            nodes = self.nodes
        else:
            rnode.name = f"node_{n_nodes-1}"
            rnode.reorder_edges(right_edges + lbond)
            rnode.add_axis_names(axes_names + ["lbond"])
            nodes.append(rnode)

        self._nodes = set(nodes)

    def truncate(self, bond_dimension=None, max_truncation_err=None):
        """Truncate in-place the bond dimension of the tensor train according
        to ``bond_dimension``. This is done from left to right and once
        finished the network is right normalized.

        Notes
        -----
        The truncation method consists on performing a svd decomposition from
        left to right and truncating each nodes svd by the given parameters.

        Parameters
        ----------
        bond_dimension: List of int or int
            List of integers that define, from left to right, the target bond
            dimension. If a single integer is provided is understood as the
            same bond dimension for all bonds. None is understood as
            "infinite" bond dimension and hence no truncation is done.

        max_truncation_err: List of float or float
            Maximum truncation error for each individual node. If a single
            value is provided instead of a list, it is the assumed that the
            same value applies for every node.

        Returns
        -------
        error: float
            List of truncated singular values.

        See also
        --------
        tensornetwork.split_node:
            Function employed to split individual nodes. ``bond_dimension``
            and ``max_truncation_err`` directly translate into
            ``max_singular_values`` and ``max_truncation_err``.
        """
        if not isinstance(bond_dimension, list):
            bond_dimension = [bond_dimension] * len(self.bond_edges)

        if not isinstance(max_truncation_err, list):
            max_truncation_err = [max_truncation_err] * len(self.bond_edges)

        if len(self.train_nodes) == 1:
            return []

        total_error = []
        for i, bond_edge in enumerate(self.bond_edges):

            node = self.train_nodes[i]
            next_node = self.train_nodes[i+1]
            dim = bond_dimension[i]
            err = max_truncation_err[i]

            # We begin by performing the following (svd) transformation:
            #   |   |             |           |
            # - * - * -   --->  - * - * - * - * -
            #   |   |             |           |
            # where the edge between the nodes (left fig) is the bond_edge.
            # Note that the svd truncates the dimension of the new nodes.
            left_edges = [edge for edge in node if edge is not bond_edge]
            right_edges = [bond_edge]
            lnode, s, rnode, error = tn.split_node_full_svd(
                node,
                left_edges,
                right_edges,
                max_singular_values=dim,
                max_truncation_err=err,
            )
            total_error += error.tolist()
            lnode.name = node.name
            lnode.reorder_edges(left_edges + [s[0]])
            lnode.add_axis_names(node.axis_names)

            # The next step is to contract the last two nodes to obtain the
            # network again in the tt format:
            #   |           |            |   |
            # - * - * - * - * -  ----> - * - * -
            #   |           |            |   |
            # Note that bond edge connects now rnode and next_node
            axis_names = next_node.axis_names
            edge_order = [next_node["out"]] if "out" in axis_names else []
            edge_order += [next_node["in"]] if "in" in axis_names else []
            edge_order += [s[0]]  # This is the new bond edge
            edge_order += [next_node["rbond"]] if "rbond" in axis_names else []

            tn.contract(s[1])
            new_next_node = tn.contract(bond_edge)
            new_next_node.name = next_node.name
            new_next_node.reorder_edges(edge_order)
            new_next_node.add_axis_names(axis_names)

        # We append the last node after the for loop as it does not need to be
        # truncated.
        new_nodes = [edge.node1 for edge in self.bond_edges]
        new_nodes += [self.bond_edges[-1].node2]

        self._nodes = set(new_nodes)

        return total_error

    @classmethod
    def _fast_constructor(cls, out_edges, in_edges, nodes):
        """Fast constructor for a TensorTrain. This is unsafe and should only be
        used if it is known with absolute certainty that the input edges and
        nodes form a correct TensorTrain. For example, after a matmul operation
        with two valid networks.
        """
        out = cls.__new__(cls)
        out.in_edges = in_edges
        out.out_edges = out_edges
        out._nodes = nodes

        return out


def _check_shape(nodes):
    """Check that the nodes have the appropriate shape for the `from_node_list`
    method."""
    if len(nodes[0].shape) != 2 and len(nodes[0].shape) != 3:
        raise ValueError(
            " the shape of the input nodes is not correct. The"
            f" first node has rank {len(nodes[0].shape)} but can"
            " only be 2 or 3."
        )

    previous_lbond_dim = nodes[0].shape[-1]
    for i, node in enumerate(nodes[1:-1], start=2):
        if len(node.shape) != 1 + len(nodes[0].shape):
            raise ValueError(
                f" the shape of the {i}-th node is not correct. It"
                f" has rank {len(node.shape)} but was expecting"
                f" {len(nodes[0].shape) + 1}."
            )
        # Checking bond_dimension is not strictly necessary but we do it to
        # raise a clearer error message.
        if node.shape[-2] != previous_lbond_dim:
            raise ValueError(
                f" the bond shape between the {i-1}-th and {i}-th"
                f" nodes is different ({previous_lbond_dim} and"
                f" {node.shape[-2]} respectively)."
            )
        previous_lbond_dim = node.shape[-1]

    if len(nodes[-1].shape) != len(nodes[0].shape):
        raise ValueError(
            " the shape of the last node is not correct. It"
            f" has rank {len(nodes[-1].shape)} but was expecting"
            f" {len(nodes[0].shape)}."
        )

    if nodes[-1].shape[-1] != previous_lbond_dim:
        raise ValueError(
            f" the bond shape between the last and the previoust"
            " to last node is"
            f" different ({previous_lbond_dim} and"
            f" {nodes[-1].shape[-1]} respectively)."
        )
