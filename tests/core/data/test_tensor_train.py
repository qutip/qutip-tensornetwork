import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from qutip.core import data
from qutip.core.data import dense
from qutip_tensornetwork.core.data import Network, FiniteTT
from qutip_tensornetwork.testing import assert_network_close
from .conftest import random_node, random_complex_network, random_one_node_network
import tensornetwork as tn


def assert_nodes_name(tt):
    """Assert if nodes are not named correctly."""
    for i, node in enumerate(tt.train_nodes):
        assert node.name == f"node_{i}"


def assert_in_edges_name(tt):
    for i, edge in enumerate(tt.in_edges):
        assert tt.train_nodes[i]["in"] is edge


def assert_out_edges_name(tt):
    for i, edge in enumerate(tt.out_edges):
        assert tt.train_nodes[i]["out"] is edge


def assert_bond_edges_name(tt):
    for i, edge in enumerate(tt.bond_edges):
        assert tt.train_nodes[i]["rbond"] is edge
        assert tt.train_nodes[i + 1]["lbond"] is edge


class TestInit:
    @pytest.mark.parametrize(
        "in_shape, out_shape",
        [
            ((2, 2, 2), (2, 2, 2)),
            ((2, 2), (2, 2)),
            ((2,), (2,)),
            ((2, 2, 2), ()),
            ((2,), ()),
            ((), (2, 2, 2)),
            ((), (2,)),
        ],
    )
    def test_init_default_args(self, in_shape, out_shape):
        in_node = random_node(in_shape)
        out_node = random_node(out_shape)
        tt = FiniteTT(out_node[:], in_node[:])
        network = Network(out_node[:], in_node[:])

        assert len(tt.train_nodes) == max(len(in_shape), len(out_shape))
        assert len(tt.in_edges) == len(in_shape)
        assert len(tt.out_edges) == len(out_shape)
        assert len(tt.bond_edges) == max(len(in_shape) - 1, len(out_shape) - 1)
        assert set(tt.nodes) == set(tt.train_nodes)
        assert_in_edges_name(tt)
        assert_out_edges_name(tt)
        assert_bond_edges_name(tt)
        assert_nodes_name(tt)
        assert tt.bond_dimension == [e.dimension for e in tt.bond_edges]
        assert_almost_equal(network.to_array(), tt.to_array())

    def test_init_raise_if_edges_not_unique(self):
        """in_edges and out_edges must be unique."""
        node = random_node((2, 2))
        with pytest.raises(ValueError):
            tt = FiniteTT([node[0]], [node[0]])

    def test_non_dangling_in_or_out_raises(self):
        """Input edges for in_edges and out_edges must be dangling."""
        node = random_node((2,))
        node2 = random_node((2,))
        node[0] ^ node2[0]

        with pytest.raises(ValueError):
            tt = FiniteTT([node[0]], [])

        with pytest.raises(ValueError):
            tt = FiniteTT([], [node[0]])

    def test_nodes_dont_include_edges_raises(self):
        """Included nodes do not have the passed in_edges or out_edges."""
        node = random_node((2, 2))
        node2 = random_node((2, 2))
        with pytest.raises(ValueError):
            tt = FiniteTT(node[0:1], node[1:], nodes=[node2])

    def test_non_in_out_edges_dangling_raises(self):
        """Included nodes can not have dangling edges that are not in_edges or
        out_edges
        """
        node = random_node((2, 2, 2))
        with pytest.raises(ValueError):
            tt = FiniteTT([node[0]], [node[1]])

    def test_scalar_network(self):
        """Test that networks that contains only a scalar node can be created."""
        node = random_node(())
        tt = FiniteTT([], [], nodes=[node])
        assert len(tt.nodes) == 1
        assert len(tt.train_nodes) == 0
        assert len(tt.in_edges) == 0
        assert len(tt.out_edges) == 0
        assert len(tt.bond_edges) == 0
        np.testing.assert_allclose(node.tensor, tt.to_array())

    def test_empty_raises(self):
        with pytest.raises(ValueError) as e:
            network = FiniteTT([], [], nodes=[])

        with pytest.raises(ValueError) as e:
            network = FiniteTT([], [], nodes=None)


@pytest.mark.parametrize("n", [2, 3, 4])
class TestFrom_node_list:
    def test_ket(self, n):
        d = 3
        chi = 10
        list_tensors = [np.random.random((d, chi))]
        list_tensors += [np.random.random((d, chi, chi)) for _ in range(n - 2)]
        list_tensors += [np.random.random((d, chi))]

        tt = FiniteTT.from_node_list(list_tensors)

        assert len(tt.train_nodes) == n
        assert len(tt.in_edges) == 0
        assert len(tt.out_edges) == n
        assert len(tt.bond_edges) == n - 1
        assert set(tt.nodes) == set(tt.train_nodes)
        assert_in_edges_name(tt)
        assert_out_edges_name(tt)
        assert_bond_edges_name(tt)
        assert_nodes_name(tt)
        assert tt.bond_dimension == [e.dimension for e in tt.bond_edges]

        for node_actual, node_desired in zip(tt.train_nodes, list_tensors):
            assert (node_actual.tensor == node_desired).all()

    def test_op(self, n):
        d = 3
        chi = 10
        list_tensors = [np.random.random((d, d, chi))]
        list_tensors += [np.random.random((d, d, chi, chi)) for _ in range(n - 2)]
        list_tensors += [np.random.random((d, d, chi))]

        tt = FiniteTT.from_node_list(list_tensors)

        assert len(tt.train_nodes) == n
        assert len(tt.in_edges) == n
        assert len(tt.out_edges) == n
        assert len(tt.bond_edges) == n - 1
        assert set(tt.nodes) == set(tt.train_nodes)
        assert_in_edges_name(tt)
        assert_out_edges_name(tt)
        assert_bond_edges_name(tt)
        assert_nodes_name(tt)
        assert tt.bond_dimension == [e.dimension for e in tt.bond_edges]

        for node_actual, node_desired in zip(tt.train_nodes, list_tensors):
            assert (node_actual.tensor == node_desired).all()

    def test_node(self, n):
        d = 3
        chi = 10
        list_tensors = [np.random.random((d, d, chi))]
        list_tensors += [np.random.random((d, d, chi, chi)) for _ in range(n - 2)]
        list_tensors += [np.random.random((d, d, chi))]
        list_tensors = [tn.Node(tensor) for tensor in list_tensors]

        tt = FiniteTT.from_node_list(list_tensors)

        assert len(tt.train_nodes) == n
        assert len(tt.in_edges) == n
        assert len(tt.out_edges) == n
        assert len(tt.bond_edges) == n - 1
        assert set(tt.nodes) == set(tt.train_nodes)
        assert_in_edges_name(tt)
        assert_out_edges_name(tt)
        assert_bond_edges_name(tt)
        assert_nodes_name(tt)
        assert tt.bond_dimension == [e.dimension for e in tt.bond_edges]

        for node_actual, node_desired in zip(tt.train_nodes, list_tensors):
            assert (node_actual.tensor == node_desired.tensor).all()

    def test_incorrect_bond_dim_raises(self, n):
        d = 3
        chi = 10
        list_tensors = [np.random.random((d, d, chi + 1))]
        list_tensors += [np.random.random((d, d, chi, chi)) for _ in range(n - 2)]
        list_tensors += [np.random.random((d, d, chi))]
        list_tensors = [tn.Node(tensor) for tensor in list_tensors]

        with pytest.raises(ValueError):
            FiniteTT.from_node_list(list_tensors)

    def test_incorrect_shape_raises(self, n):
        d = 3
        chi = 10
        list_tensors = [np.random.random((d, d, chi, chi))]
        list_tensors += [np.random.random((d, d, chi, chi)) for _ in range(n - 2)]
        list_tensors += [np.random.random((d, d, chi))]
        list_tensors = [tn.Node(tensor) for tensor in list_tensors]

        with pytest.raises(ValueError):
            FiniteTT.from_node_list(list_tensors)

        if n > 2:
            list_tensors = [np.random.random((d, d, chi))]
            list_tensors += [
                np.random.random((d, d, chi, chi, 10)) for _ in range(n - 2)
            ]
            list_tensors += [np.random.random((d, d, chi))]
            list_tensors = [tn.Node(tensor) for tensor in list_tensors]

            with pytest.raises(ValueError):
                FiniteTT.from_node_list(list_tensors)


@pytest.mark.parametrize(
    "shape, expected_dims",
    [
        ((2, 2), [[2], [2]]),
        ((2, 1), [[2], []]),
        ((1, 2), [[], [2]]),
        ((2), [[2], []]),
        ((), [[], []]),
    ],
)
def test_from_2d_array(shape, expected_dims):
    array = np.random.random(shape)
    tt = FiniteTT.from_2d_array(array)

    assert isinstance(tt, FiniteTT)
    assert len(tt.train_nodes) == (1 if shape else 0)
    assert len(tt.bond_edges) == 0
    assert tt.dims == expected_dims
    assert_in_edges_name(tt)
    assert_out_edges_name(tt)
    assert_bond_edges_name(tt)
    assert_nodes_name(tt)
    assert tt.bond_dimension == [e.dimension for e in tt.bond_edges]
    np.testing.assert_allclose(array, tt.to_array().reshape(shape))
