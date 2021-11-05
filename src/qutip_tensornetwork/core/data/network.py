import numbers
from typing import Any, Union, Callable, Optional, Sequence, Collection, Text
from typing import Tuple, Set, List, Type

import qutip
import numpy as np
import tensornetwork as tn
from tensornetwork.visualization.graphviz import to_graphviz
from tensornetwork.network_components import AbstractNode, Node, Edge
from tensornetwork.network_components import CopyNode
from tensornetwork.network_operations import get_all_nodes, copy, reachable
from tensornetwork.network_operations import get_subgraph_dangling
from tensornetwork.contractors import greedy
Tensor = Any

__all__ = ['copy_quoperator', 'Network']

def _is_shape_compatible_with_dims(shape, dims):
    return shape[0]==np.prod(dims[0]) and shape[1]==np.prod(dims[1])

def copy_quoperator(quoperator, copy=True, conjugate=False):
    """Copies nodes of QuOperator and returns a new QuOperator with the copied nodes and new edges linked."""
    if copy==False:
        return quoperator

    nodes_dict, edge_dict = tn.copy(quoperator.nodes, conjugate=conjugate)
    out_edges = [edge_dict[e] for e in quoperator.out_edges]
    in_edges = [edge_dict[e] for e in quoperator.in_edges]
    ref_nodes = [nodes_dict[n] for n in quoperator.ref_nodes]
    ignore_edges = [edge_dict[e] for e in quoperator.ignore_edges]
    return QuOperator(out_edges, in_edges, ref_nodes, ignore_edges)

def _check_valid_shape(shape, error_msg):
    """Check that shape is a tuple of len 2 with positive integer numbers."""
    if not (
        isinstance(shape, tuple)
        and len(shape) == 2
        and isinstance(shape[0], numbers.Integral)
        and isinstance(shape[1], numbers.Integral)
        and shape[0] > 0
        and shape[1] > 0
    ):
        raise ValueError(error_msg)

class Network(qutip.core.data.Data):
    """Represents arbitrary quantum objects as tensor networks.

    Tensor networks will be composed of `tensornetwork.Nodes` and
    `tensornetwork.Edges`. Nodes represent n-dimensional arrays whose dimensions
    are connected with edges to other arrays. An edge thus represents the
    contraction that needs to be carried out between two nodes. The contraction
    of nodes is done in a `lazy` way, which means that is only carried out
    when necessary or explicitly required with the contract method.

    Edges can be either dangling (one of its ends is not connected to any node)
    or fully connected. To represents arbitrary quantum objects, we
    employ the dangling edges `out_edges` and `in_edges`. Considered as a matrix, the
    dangling `out_edges` and `in_edges` represent the rows and columns of the
    matrix respectively.

    Examples:
    --------

    Notes:
    -----
    Most of the operations and logic of this class has been borrowed from
    `tensornetwork.QuOperator`. This class is by no means compatible with
    `QuOperator` but we probide a method, `as_quoperator()` that returns a view
    of this class as a `QuOperator`.
    """

    def __init__(
        self,
        out_edges: Sequence[Edge],
        in_edges: Sequence[Edge],
        ref_nodes: Optional[Collection[AbstractNode]] = None,
        ignore_edges: Optional[Collection[Edge]] = None) -> None:
        """Creates a new `QuOperator` from a tensor network.
        This encapsulates an existing tensor network, interpreting it as a linear
        operator.

        The network is checked for consistency: All dangling edges must either be
        in `out_edges`, `in_edges`, or `ignore_edges`.

        Args:
            out_edges: The edges of the network to be used as the output edges.
            in_edges: The edges of the network to be used as the input edges.
            ref_nodes: Nodes used to refer to parts of the tensor network that are
            not connected to any input or output edges (for example: a scalar
            factor).
            ignore_edges: Optional collection of dangling edges to ignore when
            performing consistency checks.
        """
        if len(in_edges) == 0 and len(out_edges) == 0:
            raise ValueError("Scalaras are not valid Networs in"
                             "qutip-tensornetwork.")
        out_dimension = [e for e in out_edges]
        in_dimension = [e for e in in_edges]
        if (len(out_dimension)!=len(in_dimension)
            and len(in_dimension)!=0
            and len(out_dimension)!=0):
            raise ValueError("Not a valid Networ: it must have same dimension"
                             "for in_edges and out_edges")

        self.out_edges = list(out_edges)
        self.in_edges = list(in_edges)
        self.ignore_edges = set(ignore_edges) if ignore_edges else set()
        self.ref_nodes = set(ref_nodes) if ref_nodes else set()
        self._check_network()


    @property
    def shape(self):
        return (np.prod(self.dims[0], dtype=int),
                np.prod(self.dims[1], dtype=int))

    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor,
        out_axes: Sequence[int],
        in_axes: Sequence[int],
        backend: Optional[Text] = None,
        name: Optional[Text] = None) -> "QuOperator":
        """Construct a `QuOperator` directly from a single tensor.

        This first wraps the tensor in a `Node`, then constructs the `QuOperator`
        from that `Node`.

        Args:
            tensor: The tensor.
            out_axes: The axis indices of `tensor` to use as `out_edges`.
            in_axes: The axis indices of `tensor` to use as `in_edges`.
            backend: Optionally specify the backend to use for computations.
        Returns:
            The new operator.
        """
        n = Node(tensor, name=name, backend=backend)
        out_edges = [n[i] for i in out_axes]
        in_edges = [n[i] for i in in_axes]
        return cls(out_edges, in_edges, set([n]))

    @property
    def nodes(self) -> Set[AbstractNode]:
        """All tensor-network nodes involved in the operator."""
        return reachable(
            get_all_nodes(self.out_edges + self.in_edges) | self.ref_nodes)


    @property
    def dims(self) -> List[List[int]]:
        out_space = [e.dimension for e in self.out_edges]
        in_space = [e.dimension for e in self.in_edges]
        return [out_space, in_space]

    def _check_network(self) -> None:
        """Check that the network has the expected dimensionality.

        This checks that all input and output edges are dangling and that
        there are no other dangling edges (except any specified in
        `ignore_edges`). If not, an exception is raised.
        """
        for (i, e) in enumerate(self.out_edges):
          if not e.is_dangling():
            raise ValueError("Output edge {} is not dangling!".format(i))
        for (i, e) in enumerate(self.in_edges):
          if not e.is_dangling():
            raise ValueError("Input edge {} is not dangling!".format(i))
        for e in self.ignore_edges:
          if not e.is_dangling():
            raise ValueError("ignore_edges contains non-dangling edge: {}".format(
                str(e)))

        known_edges = set(self.in_edges) | set(self.out_edges) | self.ignore_edges
        all_dangling_edges = get_subgraph_dangling(self.nodes)
        if known_edges != all_dangling_edges:
          raise ValueError("The network includes unexpected dangling edges (that "
                           "are not members of ignore_edges).")

    def conj(self) -> "Network":
        raise NotImplementedError()

    def transopose(self) -> "Network":
        raise NotImplementedError()

    # TODO: this does not return a copy
    def copy(self) -> "Network":
        return self

    def from_np_array(self, array) -> "Network":
        raise NotImplementedError()

    def _repr_svg_(self):
        # Dot engine seems to be prettier than neato to draw our graphs.
        return to_graphviz(self.nodes, engine='dot')._repr_svg_()

    def adjoint(self) -> "Network":
        """The adjoint of the operator.

        This creates a new `QuOperator` with complex-conjugate copies of all
        tensors in the network and with the input and output edges switched.
        """
        nodes_dict, edge_dict = copy(self.nodes, True)
        out_edges = [edge_dict[e] for e in self.in_edges]
        in_edges = [edge_dict[e] for e in self.out_edges]
        ref_nodes = [nodes_dict[n] for n in self.ref_nodes]
        ignore_edges = [edge_dict[e] for e in self.ignore_edges]
        return Network(out_edges, in_edges, ref_nodes, ignore_edges)

    def trace(self) -> "scalar":
        """The trace of the operator."""
        return self.partial_trace(range(len(self.in_edges))).to_array()

    def norm(self) -> "scalar":
        """The norm of the operator.

        This is the 2-norm (also known as the Frobenius or Hilbert-Schmidt
        norm).
        """
        return (self.adjoint() @ self).trace()

    def partial_trace(
        self,
        subsystems_to_trace_out: Collection[int]) -> "Network":
        """The partial trace of the operator.

        Subsystems to trace out are supplied as indices, so that dangling edges
        are connected to eachother as:
          `out_edges[i] ^ in_edges[i] for i in subsystems_to_trace_out`

        This does not modify the original network. The original ordering of the
        remaining subsystems is maintained.

        Args:
          subsystems_to_trace_out: Indices of subsystems to trace out.
        Returns:
          A new QuOperator or QuScalar representing the result.
        """
        out_edges_trace = [self.out_edges[i] for i in subsystems_to_trace_out]
        in_edges_trace = [self.in_edges[i] for i in subsystems_to_trace_out]

        check_spaces(in_edges_trace, out_edges_trace)

        nodes_dict, edge_dict = copy(self.nodes, False)
        for (e1, e2) in zip(out_edges_trace, in_edges_trace):
          edge_dict[e1] = edge_dict[e1] ^ edge_dict[e2]

        # get leftover edges in the original order
        out_edges_trace = set(out_edges_trace)
        in_edges_trace = set(in_edges_trace)
        out_edges = [
            edge_dict[e] for e in self.out_edges if e not in out_edges_trace
        ]
        in_edges = [edge_dict[e] for e in self.in_edges if e not in in_edges_trace]
        ref_nodes = [n for _, n in nodes_dict.items()]
        ignore_edges = [edge_dict[e] for e in self.ignore_edges]

        return Network(out_edges, in_edges, ref_nodes, ignore_edges)

    def __matmul__(self, other: "QuOperator") -> "QuOperator":
        """The action of this operator on another.

        Given `QuOperator`s `A` and `B`, produces a new `QuOperator` for `A @ B`,
        where `A @ B` means: "the action of A, as a linear operator, on B".

        Under the hood, this produces copies of the tensor networks defining `A`
        and `B` and then connects the copies by hooking up the `in_edges` of
        `A.copy()` to the `out_edges` of `B.copy()`.
        """
        check_spaces(self.in_edges, other.out_edges)

        # Copy all nodes involved in the two operators.
        # We must do this separately for self and other, in case self and other
        # are defined via the same network components (e.g. if self === other).
        nodes_dict1, edges_dict1 = copy(self.nodes, False)
        nodes_dict2, edges_dict2 = copy(other.nodes, False)

        # connect edges to create network for the result
        for (e1, e2) in zip(self.in_edges, other.out_edges):
          _ = edges_dict1[e1] ^ edges_dict2[e2]

        in_edges = [edges_dict2[e] for e in other.in_edges]
        out_edges = [edges_dict1[e] for e in self.out_edges]
        ref_nodes = ([n for _, n in nodes_dict1.items()] +
                     [n for _, n in nodes_dict2.items()])
        ignore_edges = ([edges_dict1[e] for e in self.ignore_edges] +
                        [edges_dict2[e] for e in other.ignore_edges])

        return Network(out_edges, in_edges, ref_nodes, ignore_edges)

    def tensor(self, other: "Network") -> "Network":
        """Tensor product with another operator.

        Given two operators `A` and `B`, produces a new operator `AB` representing
        `A` ⊗ `B`. The `out_edges` (`in_edges`) of `AB` is simply the
        concatenation of the `out_edges` (`in_edges`) of `A.copy()` with that of
        `B.copy()`:

        `new_out_edges = [*out_edges_A_copy, *out_edges_B_copy]`
        `new_in_edges = [*in_edges_A_copy, *in_edges_B_copy]`

        Args:
        -----
          other: The other operator (`B`).

        Returns:
        --------
          The result (`AB`).
        """
        nodes_dict1, edges_dict1 = copy(self.nodes, False)
        nodes_dict2, edges_dict2 = copy(other.nodes, False)

        in_edges = ([edges_dict1[e] for e in self.in_edges] +
                    [edges_dict2[e] for e in other.in_edges])
        out_edges = ([edges_dict1[e] for e in self.out_edges] +
                     [edges_dict2[e] for e in other.out_edges])
        ref_nodes = ([n for _, n in nodes_dict1.items()] +
                     [n for _, n in nodes_dict2.items()])
        ignore_edges = ([edges_dict1[e] for e in self.ignore_edges] +
                        [edges_dict2[e] for e in other.ignore_edges])

        return Network(out_edges, in_edges, ref_nodes, ignore_edges)

    def contract(
        self,
        contractor: Callable = greedy,
        final_edge_order: Optional[Sequence[Edge]] = None) -> "QuOperator":
        """Contract the tensor network in place.

        This modifies the tensor network representation of the operator (or vector,
        or scalar), reducing it to a single tensor, without changing the value.

        Args:
          contractor: A function that performs the contraction. Defaults to
            `greedy`, which uses the greedy algorithm from `opt_einsum` to
            determine a contraction order.
          final_edge_order: Manually specify the axis ordering of the final tensor.
        Returns:
          The present object.
        """
        nodes_dict, dangling_edges_dict = eliminate_identities(self.nodes)
        self.in_edges = [dangling_edges_dict[e] for e in self.in_edges]
        self.out_edges = [dangling_edges_dict[e] for e in self.out_edges]
        self.ignore_edges = set(dangling_edges_dict[e] for e in self.ignore_edges)
        self.ref_nodes = set(
            nodes_dict[n] for n in self.ref_nodes if n in nodes_dict)
        self._check_network()

        if final_edge_order:
          final_edge_order = [dangling_edges_dict[e] for e in final_edge_order]
          self.ref_nodes = set(
              [contractor(self.nodes, output_edge_order=final_edge_order)])
        else:
          self.ref_nodes = set([contractor(self.nodes, ignore_edge_order=True)])
        return self




    def to_array(self,
        contractor: Callable = greedy,
        final_edge_order: Optional[Sequence[Edge]] = None) -> Tensor:
        """Contracts the tensor network in place and returns the final tensor.

        Note that this modifies the tensor network representing the operator.

        The default ordering for the axes of the final tensor is:
          `*out_edges, *in_edges`.

        If there are any "ignored" edges, their axes come first:
          `*ignored_edges, *out_edges, *in_edges`.

        Args:
          contractor: A function that performs the contraction. Defaults to
            `greedy`, which uses the greedy algorithm from `opt_einsum` to
            determine a contraction order.
          final_edge_order: Manually specify the axis ordering of the final tensor.
            The default ordering is determined by `out_edges` and `in_edges` (see
            above).
        Returns:
          The final tensor representing the operator.
        """
        if not final_edge_order:
          final_edge_order = (
              list(self.ignore_edges) + self.out_edges + self.in_edges)
        self.contract(contractor, final_edge_order)
        nodes = self.nodes
        if len(nodes) != 1:
          raise ValueError("Node count '{}' > 1 after contraction!".format(
              len(nodes)))
        array = list(nodes)[0].tensor

        return array.reshape(self.shape)

    def as_quoperator(self):
        return NotImplementedError()

def check_spaces(edges_1: Sequence[Edge], edges_2: Sequence[Edge]) -> None:
    """Check the vector spaces represented by two lists of edges are compatible.

    The number of edges must be the same and the dimensions of each pair of edges
    must match. Otherwise, an exception is raised.

    Args:
    edges_1: List of edges representing a many-body Hilbert space.
    edges_2: List of edges representing a many-body Hilbert space.
    """
    if len(edges_1) != len(edges_2):
        raise ValueError("Hilbert-space mismatch: Cannot connect {} subsystems "
                        "with {} subsystems.".format(len(edges_1), len(edges_2)))

    for (i, (e1, e2)) in enumerate(zip(edges_1, edges_2)):
        if e1.dimension != e2.dimension:
            raise ValueError("Hilbert-space mismatch on subsystems {}: Input "
                            "dimension {} != output dimension {}.".format(
                            i, e1.dimension, e2.dimension))
    def reshape(self, dims):
        """Reshapes the in_edges and out_edges both input and outputs to the
        given shape. It does so by spliting the `in_edges` and `out_edges`
        and reshaping the nodes connected to those edges.

        Only spliting of edges is allowed in the reshape.
        """
        return NotImplementedError()

def _check_matmul_compatible(dims_out, dims_in):
    """For two dimensions to be compatible they must have same shape and a mcm."""


def eliminate_identities(nodes: Collection[AbstractNode]) -> Tuple[dict, dict]:
    """Eliminates any connected CopyNodes that are identity matrices.

    This will modify the network represented by `nodes`.
    Only identities that are connected to other nodes are eliminated.

    Args:
    nodes: Collection of nodes to search.
    Returns:
    nodes_dict: Dictionary mapping remaining Nodes to any replacements.
    dangling_edges_dict: Dictionary specifying all dangling-edge replacements.
    """
    nodes_dict = {}
    dangling_edges_dict = {}
    for n in nodes:
        if (isinstance(n, CopyNode)
            and n.get_rank() == 2
            and not (n[0].is_dangling() and n[1].is_dangling())):
            old_edges = [n[0], n[1]]
            _, new_edges = remove_node(n)
            if 0 in new_edges and 1 in new_edges:
                e = connect(new_edges[0], new_edges[1])
            elif 0 in new_edges:  # 1 was dangling
                dangling_edges_dict[old_edges[1]] = new_edges[0]
            elif 1 in new_edges:  # 0 was dangling
                dangling_edges_dict[old_edges[0]] = new_edges[1]
            else:
                # Trace of identity, so replace with a scalar node!
                d = n.get_dimension(0)
                # NOTE: Assume CopyNodes have numpy dtypes.
                nodes_dict[n] = Node(np.array(d, dtype=n.dtype), backend=n.backend)
        else:
            for e in n.get_all_dangling():
                dangling_edges_dict[e] = e
            nodes_dict[n] = n

    return nodes_dict, dangling_edges_dict
