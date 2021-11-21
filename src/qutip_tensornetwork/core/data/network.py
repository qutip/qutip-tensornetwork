import numbers
from typing import Any, Union, Callable, Optional, Sequence, Collection, Text
from typing import Tuple, Set, List, Type

import qutip
import numpy as np
import tensornetwork as tn
from tensornetwork.visualization.graphviz import to_graphviz
from tensornetwork.network_components import AbstractNode, Node, Edge
from tensornetwork.network_components import CopyNode
from tensornetwork.network_operations import (get_all_nodes, reachable,
                                              get_all_edges
                                             )
from tensornetwork.network_operations import get_subgraph_dangling
from tensornetwork.contractors import greedy
Tensor = Any

__all__ = ['Network']

class Network(qutip.core.data.Data):
    """Represents arbitrary quantum objects as tensor networks.

    Tensor networks will be composed of ``tensornetwork.Nodes`` and
    `tensornetwork.Edges`. Nodes represent n-dimensional arrays whose dimensions
    are connected with edges to other arrays. An edge represents the
    contraction that needs to be carried out between two nodes. The contraction
    of nodes is done in a `lazy` way, which means that is only carried out
    when necessary or explicitly required with the contract method.

    Edges can be either dangling (one of its ends is not connected to any node)
    or fully connected. To represents arbitrary quantum objects, we
    employ the dangling edges `out_edges` and `in_edges`. Considered as a matrix, the
    dangling `out_edges` and `in_edges` represent the rows and columns of the
    matrix respectively.

    Attributes
    ----------
    nodes : set of Nodes
        Nodes that belong to the Network. These can either be reachable from
        in_edges and out_edges or scalar nodes, in which case they have no
        edges.

    out_edges : list of Edges
        List of 

    in_edges : list of Edges
        List of the Edges that represent 

    dims : list of int

    shape : tuple of int

    Notes
    -----
    Most of the operations and logic of this class has been derived from
    ``tensornetwork.quantum.QuOperator``. However, this class is not compatible with
    ``QuOperator`` but we provide a method, ``as_quoperator()`` that returns a view
    of this class as a `QuOperator`.
    """

    def __init__(
        self,
        out_edges: Sequence[Edge],
        in_edges: Sequence[Edge],
        nodes: Optional[Collection[AbstractNode]] = None,
        copy: Optional[bool] = True) -> None:
        """Creates a new `Network` from a tensor network.
        This encapsulates an existing tensor network, interpreting it as a linear
        operator.

        The network is checked for consistency: All dangling edges must either be
        in `out_edges`, `in_edges`, or `ignore_edges`.

        Parameters
        ----------
        out_edges: List of Edges
            The edges of the network to be used as the output edges.

        in_edges: List of Edges
            The edges of the network to be used as the input edges.

        nodes: None or List of Nodes
            Nodes of the network. If None, the nodes are obtained
            by finding all the nodes that belong to the graphs that include
            in_edges and out_edges.

        copy: bool, default True
            Whether to copy all the ``Nodes``/``Edges`` involved in the
            network.
        """
        if (len(in_edges) == 0 and len(out_edges) == 0
            and (nodes is None or len(nodes)==0)):
            raise ValueError("Since no edges were provided, it was not possible"
                             "to infer which nodes belong to the network."
                             "You may want to include a scalar node to represent"
                             "a matrix with shape")

        self.out_edges = list(out_edges)
        self.in_edges = list(in_edges)

        # Unlike in QuOperator we will keep track of the nodes instead of
        # dynamically searching for them when necessary. This is because
        # searching all nodes in a large graph can be quite expensive while
        # keeping track of them with network operations is straightforward.
        self.nodes = set(nodes) if nodes else tn.reachable(self.in_edges
                                                         + self.out_edges)

        self._check_edge_nodes_in_nodes()
        self._check_in_out_are_dangling()
        self._check_only_in_out_are_dangling()
        self._check_edges_unique()
        # I may need extra check such as for backends etc in the future

        # TODO: should this be a porperty obtained from dims?
        super().__init__(shape=(np.prod(self.dims[0]), np.prod(self.dims[1])))

        if copy:
            node_dict, edge_dict = tn.copy(self.nodes)
            self.nodes = set(node_dict[n] for n in self.nodes)
            self.in_edges = [edge_dict[e] for e in self.in_edges]
            self.out_edges = [edge_dict[e] for e in self.out_edges]

    @property
    def dims(self) -> List[List[int]]:
        out_space = [e.dimension for e in self.out_edges]
        in_space = [e.dimension for e in self.in_edges]
        return [out_space, in_space]

    def _check_in_out_are_dangling(self) -> None:
        for (i, e) in enumerate(self.out_edges):
          if not e.is_dangling():
            raise ValueError("Output edge {} is not dangling!".format(i))
        for (i, e) in enumerate(self.in_edges):
          if not e.is_dangling():
            raise ValueError("Input edge {} is not dangling!".format(i))

    def _check_only_in_out_are_dangling(self):
        known_edges = set(self.in_edges + self.out_edges)
        all_dangling_edges = get_subgraph_dangling(self.nodes)
        if known_edges != all_dangling_edges:
            unexpected_edges = all_dangling_edges.difference(known_edges)
            raise ValueError("The network includes unexpected dangling edges."
                            + str(unexpected_edges))

    def _check_edges_unique(self):
        """Check that in_edges and out_edges are unique."""
        if (len(set(self.in_edges)) != len(self.in_edges) or
            len(set(self.out_edges)) != len(self.out_edges)):
            raise ValueError("The edges included as in_edges and out_edges"
                             "are not unique.")

    def _check_edge_nodes_in_nodes(self):
        edges = self.in_edges + self.out_edges
        if not set(e.node1 for e in edges) <= self.nodes:
            raise ValueError("The nodes for in_edges and out_edges are not "
                             "included in the passed nodes.")

    def copy(self) -> "Network":
        """
        Returns
        -------
        Network
            Returns a shallow copy of the Network. That is, tensors stored in
            ``tensornetwork.Nodes`` is not copied.
        """
        node_dict, edge_dict = tn.copy(self.nodes)
        nodes = set(node_dict[n] for n in self.nodes)
        in_edges = [edge_dict[e] for e in self.in_edges]
        out_edges = [edge_dict[e] for e in self.out_edges]

        return Network._fast_constructor(out_edges, in_edges, nodes, self.shape)

    @classmethod
    def _fast_constructor(cls, out_edges, in_edges, nodes, shape):
        """Fast constructor for a Network. This is unsafe and should only be
        used if it is known wit absolute certainty that the input edges and
        nodes form a correct Network. For example, after a matmul operation
        with two valid networks.
        """
        out = cls.__new__(cls)
        out.in_edges = in_edges
        out.out_edges = out_edges
        out.nodes = nodes
        qutip.core.data.Data.__init__(out, shape)

        return out

    def _repr_svg_(self):
        return to_graphviz(self.nodes, engine='dot')._repr_svg_()

    def conj(self) -> "Network":
        """Returns the conjugate of the network.

        The output consists on the conjugation of the nodes.

        Returns
        -------
        Network
            A network that represents the conjugation of the input.
        """
        node_dict, edge_dict = tn.copy(self.nodes, conjugate=True)
        nodes = set(node_dict[n] for n in self.nodes)
        in_edges = [edge_dict[e] for e in self.in_edges]
        out_edges = [edge_dict[e] for e in self.out_edges]

        return Network._fast_constructor(out_edges, in_edges, nodes, self.shape)

    def transopose(self) -> "Network":
        """Returns the transpose of the network.

        The output consists on the transpose of ``in_edges`` and
        ``out_edges`` such that:

        ``new_in_edges, new_out_edges = in_edges, out_edges``

        Returns
        -------
        Network
            A network that represents the transpose of the input.
        """
        node_dict, edge_dict = tn.copy(self.nodes, conjugate=False)
        nodes = set(node_dict[n] for n in self.nodes)
        in_edges = [edge_dict[e] for e in self.out_edges]
        out_edges = [edge_dict[e] for e in self.in_edges]
        shape = (self.shape[1], self.shape[0])

        return Network._fast_constructor(out_edges, in_edges, nodes, shape)

    def adjoint(self) -> "Network":
        """Returns the adjoint of the network.

        The output consists on the conjugation of all nodes and the transpose
        of the ``in_edges`` and ``out_edges`` such that:

        ``new_in_edges, new_out_edges = in_edges, out_edges``

        Returns
        -------
        Network
            A network that represents the adjoint of the input.
        """
        node_dict, edge_dict = tn.copy(self.nodes, conjugate=True)
        nodes = set(node_dict[n] for n in self.nodes)
        in_edges = [edge_dict[e] for e in self.out_edges]
        out_edges = [edge_dict[e] for e in self.in_edges]
        shape = (self.shape[1], self.shape[0])

        return Network._fast_constructor(out_edges, in_edges, nodes, shape)

    def contract(
        self,
        contractor: Callable = greedy,
        final_edge_order: Optional[Sequence[Edge]] = None,
        ) -> "Network":
        """Return the contracted version of the tensor network.

        Parameters
        ----------
        contractor: Callable
            A function that performs the contraction. Defaults to
            ``tensornetwork.contractor.greedy``, which uses the greedy
            algorithm from `opt_einsum` to determine a contraction order.

        final_edge_order: iterable of tensornetwork.Edges

        Returns
        -------
        Network
            A contracted version of the network with a single node.

        See also
        --------
            tensornetwork.contractor: This module contains other functions that
            can be used instead of ``greedy``.

        """
        # TODO: I will use a different approach and just remove
        # identities on after a matmul operation.  

        # nodes_dict, dangling_edges_dict = eliminate_identities(self.nodes)
        nodes_dict, edges_dict = tn.copy(self.nodes)

        in_edges = [edges_dict[e] for e in self.in_edges]
        out_edges = [edges_dict[e] for e in self.out_edges]
        nodes = set(nodes_dict[n] for n in self.nodes if n in nodes_dict)

        if final_edge_order is not None:
            final_edge_order = [edges_dict[e] for e in final_edge_order]
            nodes = set(
                [contractor(nodes, output_edge_order=final_edge_order)])
        else:
            nodes = set([contractor(nodes, ignore_edge_order=True)])

        return Network._fast_constructor(out_edges, in_edges, nodes, self.shape)

    def to_array(self,
        contractor: Callable = greedy,
        ) -> Tensor:
        """Returns a 2D array that represents the contraction of the tensor
        network.

        The ordering for the axes of the final array is:
          `*out_edges, *in_edges`.

        Parameters
        ----------
        contractor: tensornetwork.contractor
            A function that performs the contraction. Defaults to
            `greedy`, which uses the greedy algorithm from `opt_einsum` to
            determine a contraction order.

        Returns
        -------
        tensor: numpy.ndarray
            The final tensor representing the operator.
        """
        final_edge_order = self.out_edges + self.in_edges
        network = self.contract(contractor, final_edge_order=final_edge_order)
        nodes = network.nodes
        if len(nodes) != 1:
          raise ValueError("Node count '{}' > 1 after contraction!".format(
              len(nodes)))
        array = list(nodes)[0].tensor

        return array.reshape(self.shape)

    @classmethod
    def from_2d_array(cls, array):
        """Create a network from a 2D, 1D or scalar array. This network will
        have a single node with out_edges referring to the first index of the
        array and in_edges referring to the second index of the array. If any of
        those has dimension 1, it is ignored when creating in_edges or
        out_edges.

        Parameters
        ----------
        array: ndarray or Data
            Array from which the single node of the network is created.

        Returns
        -------
        network: Network
            Instance of Network with a single node representing array.

        Examples
        --------
        >>> array = np.array((2, 2))
        >>> net = Network.from_2d_array(array)
        >>> net.dims
        [[2], [2]]

        One dimensional arrays are understoof as kets and cero dimensional ones
        as scalars.

        >>> array = np.array((2)) # ket
        >>> net = Network.from_2d_array(array)
        >>> net.dims
        [[2], []]

        If the array has one dimension being 1, it is reshaped before the
        Network instantiation.

        >>> array = np.array((2, 1)) # ket
        >>> net = Network.from_2d_array(array)
        >>> net.dims
        [[2], []]
        """

        if isinstance(array, qutip.data.Data):
            array = array.to_array()
        elif not isinstance(array, np.ndarray):
            raise ValueError("`array` is not instance of Data or np.array.")

        shape = array.shape
        if len(shape) > 2:
            raise ValueError("This method only works with 2D, 1D or scalar "
                             "arrays but the"
                             "input has " + str(len(array.shape)) + "dimensions")

        if len(shape)==1:
            node = tn.Node(array)
            return Network(node[:], [])

        if len(shape)==0:
            node = tn.Node(array)
            return Network([], [], [node])

        if array.shape[0]==1 and array.shape[1]!=1:
            array = array.reshape(array.shape[1])
            node = tn.Node(array)
            return Network([], node[:])

        elif array.shape[0]!=1 and array.shape[1]==1:
            array = array.reshape(array.shape[0])
            node = tn.Node(array)
            return Network(node[:], [])

        elif array.shape[0]==1 and array.shape[1]==1:
            array = array.reshape(())
            node = tn.Node(array)
            return Network([], [], nodes=[node])

        else:
            node = tn.Node(array)
            return Network(node[0:1], node[1:])

    def trace(self) -> "scalar":
        raise NotImplementedError()

    def norm(self) -> "scalar":
        raise NotImplementedError()

    def as_quoperator(self):
        # TODO: This can not be implemented unless tensornetwork includes the 
        # __init__ file for the quantum module.
        raise NotImplementedError()

    def partial_trace(
        self,
        subsystems_to_trace_out: Collection[int]) -> "Network":
        """The partial trace of the operator.

        Subsystems to trace out are supplied as indices, so that dangling edges
        are connected to each other as:
        `out_edges[i] ^ in_edges[i] for i in subsystems_to_trace_out`

        This does not modify the original network. The original ordering of the
        remaining subsystems is maintained.

        Parameters
        ----------
        subsystems_to_trace_out:
            Indices of subsystems to trace out.

        Returns
        -------
        network: Network
            Network representing the partial trace of the input.
        """
        raise NotImplementedError() # This is yet not Implemented
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

    def __matmul__(self, other: "Network") -> "Network":
        """The action of this network on another.

        Given `Network`s `A` and `B`, produces a new `Network` for `A @ B`,
        where `A @ B` means: "the action of A, as a linear operator, on B".

        Note:
        -----
            Under the hood, this produces copies of the tensor networks
            defining `A` and `B` and then connects the copies by hooking up the
            `in_edges` of `A.copy()` to the `out_edges` of `B.copy()`.
        """
        # Copy all nodes involved in the two operators.
        # We must do this separately for self and other, in case self and other
        # are defined via the same network components (e.g. if self === other).
        new_nodes_self, new_edges_self = tn.copy(self.nodes, False)
        new_nodes_other, new_edges_other = tn.copy(other.nodes, False)

        in_edges = [new_edges_self[e] for e in self.in_edges]
        out_edges = [new_edges_other[e] for e in other.out_edges]

        out_edges, in_edges = _match_edges_by_split(out_edges, in_edges)

        # connect edges to create network for the result
        for (e_in, e_out) in zip(in_edges, out_edges):
          _ = e_in ^ e_out

        in_edges = [new_edges_other[e] for e in other.in_edges]
        out_edges = [new_edges_self[e] for e in self.out_edges]
        nodes = set([new_nodes_self[n] for n in self.nodes] +
                    [new_nodes_other[n] for n in other.nodes])
        shape = (self.shape[0], other.shape[1])

        return Network._fast_constructor(out_edges, in_edges, nodes, shape)

    def tensor(self, other: "Network") -> "Network":
        """Tensor product with another operator.

        Given two operators `A` and `B`, produces a new operator `AB` representing
        `A` ⊗ `B`. The `out_edges` (`in_edges`) of `AB` is simply the
        concatenation of the `out_edges` (`in_edges`) of `A.copy()` with that of
        `B.copy()`:

        ``new_out_edges = [*out_edges_A_copy, *out_edges_B_copy]``

        ``new_in_edges = [*in_edges_A_copy, *in_edges_B_copy]``

        Parameters
        ----------
        other: Network
            The other network (`B`).

        Returns
        -------
        network: Network
            Network representing `A` ⊗ `B`.
        """
        a = self.copy()
        b = other.copy()

        in_edges = [*a.in_edges, *b.in_edges]
        out_edges = [*a.out_edges, *b.out_edges]
        nodes = a.nodes | b.nodes
        shape = (a.shape[0]*b.shape[0], a.shape[1]*b.shape[1])

        return Network._fast_constructor(out_edges, in_edges, nodes, shape)


def _match_edges_by_split(out_edges, in_edges):
    """Split the edges in out_edges and in_edges to allow matrix
    multiplication. This is done reshaping nodes by spliting in_edges and out
    edges.

    Parameters
    ----------
    out_edges:
        List of ``Edges``.

    in_edges:
        List of ``Edges``.

    Examples:
    ---------

    """
    # Shallow copy to allow popping from out_edges
    _out_edges = out_edges[:]
    _in_edges = in_edges[:]

    in_dims = [e.dimension for e in in_edges]
    out_dims = [e.dimension for e in out_edges]

    new_in_edges = []
    new_out_edges = []

    if len(_in_edges) == 0 and len(_out_edges)==0:
        return _out_edges, _in_edges

    if len(_in_edges) == 0 or len(_out_edges)==0:
        raise ValueError("Edges are not compatible. The dimensions of in_edges: " +
                         str(in_dims) + " whereas for out_edges: "+str(out_dims))

    if np.prod(in_dims) != np.product(out_dims):
        raise ValueError("Edges are not compatible. The dimensions of in_edges: " +
                         str(in_dims) + " whereas for out_edges: "+str(out_dims))

    e_in = in_edges.pop()
    e_out = out_edges.pop()

    while True:

        if e_in.dimension == e_out.dimension:
            new_in_edges.append(e_in)
            new_out_edges.append(e_out)

            if len(in_edges) == 0 and len(out_edges) == 0:
                break

            e_in = in_edges.pop()
            e_out = out_edges.pop()


        elif e_in.dimension > e_out.dimension:
            # IndexError will be caught and by try/except which will then
            # raise the appropriate error
            if e_in.dimension%e_out.dimension != 0:
                raise ValueError("Edges are not compatible. The dimensions of in_edges: " +
                                 str(in_dims) + " whereas for out_edges: "+str(out_dims))
            else:
                # new_shape=(e_out.dimension, e_in.dimension//e_out.dimension)
                # new_e_in, e_in = tn.split_edge(e_in, shape=new_shape)
                new_shape = (e_in.dimension//e_out.dimension, e_out.dimension)
                e_in, new_e_in = tn.split_edge(e_in, shape=new_shape)

                new_out_edges.append(e_out)
                new_in_edges.append(new_e_in)

                e_out = out_edges.pop()

        elif e_in.dimension < e_out.dimension:
            # IndexError will be caught and by try/except which will then
            # raise the appropriate error
            if e_out.dimension%e_in.dimension != 0:
                raise ValueError("Edges are not compatible. The dimensions of in_edges: " +
                                 str(in_dims) + " whereas for out_edges: "+str(out_dims))
            else:
                # new_shape=(e_in.dimension, e_out.dimension//e_in.dimension)
                # new_e_out, e_out = tn.split_edge(e_out, shape=new_shape)
                new_shape=(e_out.dimension//e_in.dimension, e_in.dimension)
                e_out, new_e_out = tn.split_edge(e_out, shape=new_shape)

                new_out_edges.append(new_e_out)
                new_in_edges.append(e_in)

                e_in = in_edges.pop()



    new_out_edges.reverse()
    new_in_edges.reverse()
    return new_out_edges, new_in_edges


def eliminate_identities(nodes: Collection[AbstractNode]) -> Tuple[dict, dict]:
    """Eliminates any connected CopyNodes that are identity matrices.

    This will modify the network represented by `nodes`.
    Only identities that are connected to other nodes are eliminated.

    Parameters
    ----------
    nodes:
        Collection of nodes to search.

    Returns
    -------
    nodes_dict:
        Dictionary mapping remaining Nodes to any replacements.

    dangling_edges_dict:
        Dictionary specifying all dangling-edge replacements.
    """
    nodes_dict = {}
    dangling_edges_dict = {}
    for n in nodes:
        if isinstance(
            n, CopyNode) and n.get_rank() == 2 and not (n[0].is_dangling() and
                                                        n[1].is_dangling()):
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
