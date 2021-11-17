import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from qutip.core import data
from qutip.core.data import dense
from qutip_tensornetwork.core.data import Network
from qutip_tensornetwork.testing import assert_network_close
from .conftest import random_node, random_complex_network
import tensornetwork as tn

def test_init_only_in_edge():
    node = random_node((2,))
    network = Network([], node[0:], copy=False)
    assert len(network.nodes) == 1
    assert network.nodes.pop() is node
    assert len(network.in_edges) == 1
    assert network.in_edges[0] is node[0]
    assert len(network.out_edges) == 0

def test_init_only_out_edge():
    node = random_node((2,))
    network = Network(node[0:], [], copy=False)
    assert len(network.nodes) == 1
    assert network.nodes.pop() is node
    assert len(network.out_edges) == 1
    assert network.out_edges[0] is node[0]
    assert len(network.in_edges) == 0

def test_init_both_in_and_out_edge():
    node = random_node((2, 2))
    network = Network(node[0:1], node[1:], copy=False)
    assert len(network.nodes) == 1
    assert network.nodes.pop() is node
    assert len(network.out_edges) == 1
    assert network.out_edges[0] is node[0]
    assert len(network.in_edges) == 1
    assert network.in_edges[0] is node[1]

def test_init_raise_if_edges_not_unique():
    """in_edges and out_edges must be unique."""
    node = random_node((2, 2))
    with pytest.raises(ValueError):
       network = Network([node[0]], [node[0]])

def test_non_dangling_in_or_out_raises():
    """Input edges for in_edges and out_edges must be dangling."""
    node = random_node((2,))
    node2 = random_node((2,))
    node[0] ^ node2[0]

    with pytest.raises(ValueError):
       network = Network([node[0]], [])

    with pytest.raises(ValueError):
       network = Network([], [node[0]])

def test_nodes_dont_include_edges_raises():
    """Included nodes do not have the passed in_edges or out_edges."""
    node = random_node((2,2))
    node2 = random_node((2,2))
    with pytest.raises(ValueError):
        network = Network(node[0:1], node[1:], nodes=[node2], copy=False)

def test_non_in_out_edges_dangling_raises():
    """Included nodes can not have dangling edges that are not in_edges or
    out_edges
    """
    node = random_node((2,2,2))
    with pytest.raises(ValueError):
        network = Network([node[0]], [node[1]], copy=False)

def test_scalar_nodes():
    """It is possible to include scalar nodes as long as they are not the only
    nodes.
    """
    array = np.random.random()
    node_scalar = tn.Node(array)
    node = random_node((2,2))
    network = Network([node[0]], [node[1]], nodes=[node, node_scalar],
                      copy=False)
    assert network.nodes <= set([node, node_scalar])

def test_scalar_network():
    """Test that networks that contains only a scalar node can be created.
    """
    node = random_node(())
    network = Network([], [], nodes=[node])
    np.testing.assert_allclose(node.tensor, network.nodes.pop().tensor)

def test_scalar_network_contraction():
    node = random_node(())
    network = Network([], [], nodes=[node])
    np.testing.assert_allclose(node.tensor, network.to_array())

def test_empty_rises():
    with pytest.raises(ValueError):
       network = Network([], [], nodes=[])

    with pytest.raises(ValueError):
       network = Network([], [], nodes=None)

# TODO: discuss this
# def test_nodes_is_shallow_copy():
    # array = np.random.random(2)
    # node = tn.Node(array)
    # network = Network([node[0]], [])
    # nodes = network.nodes
    # assert nodes is not network._nodes
    # assert nodes.pop() is network._nodes.pop()

def test_copy_in_init():
    node = random_node((2,2))
    network = Network(node[0:1], node[1:], copy=False)
    assert node is network.nodes.pop()
    network = Network(node[0:1], node[1:], copy=True)
    network_node = network.nodes.pop()
    assert network_node is not node
    assert network_node.tensor is node.tensor

def test_copy():
    node = random_node((2,2))
    network = Network(node[0:1], node[1:], copy=False)

    copied_network = network.copy()
    assert copied_network.in_edges[0] is not network.in_edges[0]
    assert copied_network.out_edges[0] is not network.out_edges[0]

    assert copied_network.nodes is not network.nodes
    node = network.nodes.pop()
    copied_node = copied_network.nodes.pop()
    assert node is not copied_node
    assert node.tensor is copied_node.tensor

def test_fast_constructor():
    node = random_node((2,2))
    network = Network(node[0:1], node[1:], copy=False)

    new_network = Network._fast_constructor(network.out_edges,
                                             network.in_edges,
                                             network.nodes,
                                             network.shape)

    assert network.in_edges is new_network.in_edges
    assert network.out_edges is new_network.out_edges
    assert network.nodes is new_network.nodes
    assert network.shape == new_network.shape

def test_shape():
    node = random_node((2,2))

    network = Network(node[0:], [])
    assert network.shape == (4, 1)

    network = Network([], node[0:])
    assert network.shape == (1, 4)

    network = Network([node[0]], [node[1]])
    assert network.shape == (2, 2)

# This are very simple tests that are not meant to be complete. For a complete
# set of tests for the mathematical operations see test_mathematics. Both shape
# and dims are tested as these are generated independently.
def test_conj():
    node = random_node((2,3))
    network = Network([node[0]], [node[1]])

    assert network.shape == network.conj().shape
    assert network.dims == network.conj().dims

    np.testing.assert_almost_equal(node.tensor.conj(),
                                   network.conj().nodes.pop().tensor)

def test_adjoint():
    node = random_node((2,3))
    network = Network([node[0]], [node[1]])

    assert network.shape[0] == network.adjoint().shape[1]
    assert network.shape[1] == network.adjoint().shape[0]
    assert network.dims[0] == network.adjoint().dims[1]
    assert network.dims[1] == network.adjoint().dims[0]

    np.testing.assert_almost_equal(node.tensor.conj(),
                                   network.adjoint().nodes.pop().tensor)

def test_transpose():
    node = random_node((2,3))
    network = Network([node[0]], [node[1]])

    assert network.shape[0] == network.transopose().shape[1]
    assert network.shape[1] == network.transopose().shape[0]
    assert network.dims[0] == network.transopose().dims[1]
    assert network.dims[1] == network.transopose().dims[0]

    np.testing.assert_almost_equal(node.tensor,
                                   network.transopose().nodes.pop().tensor)

class TestContract():
    node1 = random_node((2,))
    node2 = random_node((2, 2, 2))
    node1[0] ^ node2[2]

    network = Network([node2[0]], [node2[1]], [node1, node2])

    expect = (node1@node2).tensor
    def test_default(self):
        result = self.network.contract()
        result = result.nodes.pop().tensor

        np.testing.assert_allclose(result, self.expect)

    def test_edge_order(self):
        # We transpose the final result by changing the final_edge_order
        result = self.network.contract(final_edge_order=self.network.in_edges +
                                       self.network.out_edges)
        result = result.nodes.pop().tensor

        np.testing.assert_allclose(result, self.expect.T)

def test_to_array():
    node1 = random_node((2,))
    node2 = random_node((2, 2, 2))
    node1[0] ^ node2[2]

    network = Network(node2[0:2], [], [node1, node2])
    # We transpose the final result by changing the final_edge_order
    result = network.to_array()
    expect = (node1@node2).tensor.reshape((4,1))

    np.testing.assert_allclose(result, expect)

class TestMatmul:

    @pytest.mark.parametrize("dim_in, dim_out",
                            [([4], [2,2]),
                             ([2,2], [4]),
                             ([2,10], [2,5,2]),
                             ([2,10], [2,2,5]),
                             ([2], [2]),
                            ])
    def test_comatible(self, dim_in, dim_out):
        """Tests that matrix multiplication works with networks that have
        different dimensions. This test does not check for the correct tensor 
        structure but rather that the output is numerically correct once
        contracted."""
        right = random_node(dim_out)
        left = random_node(dim_in)

        right_net = Network(right[:], [])
        left_net = Network([], left[:])

        result = left_net@right_net

        # Desired
        left = left.tensor.reshape((1, np.prod(dim_out)))
        right = right.tensor.reshape((np.prod(dim_in), 1))
        desired = left@right

        np.testing.assert_allclose(result.to_array(), desired)




    @pytest.mark.parametrize("dim_in, dim_out",
                            [([4], [3]),
                             ([4], [2, 3]),
                             ([2, 3], [4]),
                             ([2, 10], [5, 2, 2]),
                             ([2, 10], [2, 2, 2]),
                             ([2, 5], [5, 2]),
                            ])
    def test_non_compatible_raises(self, dim_in, dim_out):
        """Tests that matrix multiplication between networks that are supposed
        to not be compatible raises the appropiate error.
        """
        left = random_node(dim_in)
        right = random_node(dim_out)

        right_net = Network(right[:], [])
        left_net = Network([], left[:])

        with pytest.raises(ValueError):
            result = left_net@right_net

    def test_numerically_correct(self):
        network1 = random_complex_network(5)
        network2 = network1.adjoint()

        expected = network2.to_array()@network1.to_array()
        np.testing.assert_allclose(expected, (network2@network1).to_array())

    def test_graph_structure(self):
        """The operation tested here can be respresented as a graph in the
        forllowing way:
        n1 - n3
        n2 - n4

        This test in particular checks that matmul for networks gives the correct
        graph.
        """
        n1 = random_node((3,))
        n2 = random_node((3,))
        n3 = random_node((3,))
        n4 = random_node((3,))

        network1 = Network([n1[0], n2[0]], [])
        network2 = Network([], [n3[0], n4[0]])
        result = network2@network1

        n1[0] ^ n3[0]
        n2[0] ^ n4[0]
        expexted_network = Network([], [], [n1, n2, n3, n4])

        assert_network_close(result, expexted_network)

def test_tensor():
    node1 = random_node((2,2))
    network1 = Network([node1[0]], [node1[1]])

    node2 = random_node((2,2))
    network2 = Network([node2[0]], [node2[1]])

    result = network1.tensor(network2)

    # Expected
    desired = Network([node1[0], node2[0]], [node1[1], node2[1]])

    assert_network_close(result, desired)

@pytest.mark.parametrize("shape, expected_dims",
                         [((2, 2), [[2], [2]]),
                         ((2, 1), [[2], []]),
                         ((1, 2), [[], [2]]),
                         ((2), [[2], []]),
                         ((), [[], []]),
                        ]
                        )
def test_from_2d_array(shape, expected_dims):
    array = np.random.random(shape)
    network = Network.from_2d_array(array)

    np.testing.assert_allclose(array, network.to_array().reshape(shape))
    assert network.dims == expected_dims
