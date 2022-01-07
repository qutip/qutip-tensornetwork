import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from qutip.core import data
from qutip.core.data import dense
from qutip_tensornetwork.core.data import Network, FiniteTT, network_to_tt
from qutip_tensornetwork.testing import assert_network_close
from .conftest import random_node, random_complex_network, random_one_node_network
import tensornetwork as tn


def assert_nodes_name(tt):
    """Assert if nodes are not named correctly."""
    for i, node in enumerate(tt.node_list):
        assert node.name == f"node_{i}"


def assert_in_edges_name(tt):
    for i, edge in enumerate(tt.in_edges):
        assert tt.node_list[i]["in"] is edge


def assert_out_edges_name(tt):
    for i, edge in enumerate(tt.out_edges):
        assert tt.node_list[i]["out"] is edge


def assert_bond_edges_name(tt):
    for i, edge in enumerate(tt.bond_edges):
        assert tt.node_list[i]["rbond"] is edge
        assert tt.node_list[i + 1]["lbond"] is edge

def random_mpo(n, d, bond_dimension):
    """Create a random mpo with n sites d dimension per site and
    bond_dimesnion."""
    if n>1:
        list_tensors = [np.random.random((d, d, bond_dimension))-1/2]
        list_tensors += [np.random.random((d, d, bond_dimension,
                                           bond_dimension)) -1/2 for _ in range(n-2)]
        list_tensors += [np.random.random((d, d, bond_dimension)) -1/2]
        mpo = FiniteTT.from_node_list(list_tensors)
    elif n==1:
        node = tn.Node(np.random.random((d, d)))
        mpo = FiniteTT(node[0:1], node[1:])
    return mpo

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

        assert len(tt.node_list) == max(len(in_shape), len(out_shape))
        assert len(tt.in_edges) == len(in_shape)
        assert len(tt.out_edges) == len(out_shape)
        assert len(tt.bond_edges) == max(len(in_shape) - 1, len(out_shape) - 1)
        assert set(tt.nodes) == set(tt.node_list)
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
        assert len(tt.node_list) == 0
        assert len(tt.in_edges) == 0
        assert len(tt.out_edges) == 0
        assert len(tt.bond_edges) == 0
        np.testing.assert_allclose(node.tensor, tt.to_array())

    def test_empty_rises(self):
        with pytest.raises(ValueError) as e:
            network = FiniteTT([], [], nodes=[])

        with pytest.raises(ValueError) as e:
            network = FiniteTT([], [], nodes=None)


@pytest.mark.parametrize("n", [2, 3, 4])
class Test_from_node_list:
    def test_ket(self, n):
        d = 3
        chi = 10
        list_tensors = [np.random.random((d, chi))]
        list_tensors += [np.random.random((d, chi, chi)) for _ in range(n - 2)]
        list_tensors += [np.random.random((d, chi))]

        tt = FiniteTT.from_node_list(list_tensors)

        assert len(tt.node_list) == n
        assert len(tt.in_edges) == 0
        assert len(tt.out_edges) == n
        assert len(tt.bond_edges) == n - 1
        assert set(tt.nodes) == set(tt.node_list)
        assert_in_edges_name(tt)
        assert_out_edges_name(tt)
        assert_bond_edges_name(tt)
        assert_nodes_name(tt)
        assert tt.bond_dimension == [e.dimension for e in tt.bond_edges]

        for node_actual, node_desired in zip(tt.node_list, list_tensors):
            assert (node_actual.tensor == node_desired).all()

    def test_op(self, n):
        d = 3
        chi = 10
        list_tensors = [np.random.random((d, d, chi))]
        list_tensors += [np.random.random((d, d, chi, chi)) for _ in range(n - 2)]
        list_tensors += [np.random.random((d, d, chi))]

        tt = FiniteTT.from_node_list(list_tensors)

        assert len(tt.node_list) == n
        assert len(tt.in_edges) == n
        assert len(tt.out_edges) == n
        assert len(tt.bond_edges) == n - 1
        assert set(tt.nodes) == set(tt.node_list)
        assert_in_edges_name(tt)
        assert_out_edges_name(tt)
        assert_bond_edges_name(tt)
        assert_nodes_name(tt)
        assert tt.bond_dimension == [e.dimension for e in tt.bond_edges]

        for node_actual, node_desired in zip(tt.node_list, list_tensors):
            assert (node_actual.tensor == node_desired).all()

    def test_node(self, n):
        d = 3
        chi = 10
        list_tensors = [np.random.random((d, d, chi))]
        list_tensors += [np.random.random((d, d, chi, chi)) for _ in range(n - 2)]
        list_tensors += [np.random.random((d, d, chi))]
        list_tensors = [tn.Node(tensor) for tensor in list_tensors]

        tt = FiniteTT.from_node_list(list_tensors)

        assert len(tt.node_list) == n
        assert len(tt.in_edges) == n
        assert len(tt.out_edges) == n
        assert len(tt.bond_edges) == n - 1
        assert set(tt.nodes) == set(tt.node_list)
        assert_in_edges_name(tt)
        assert_out_edges_name(tt)
        assert_bond_edges_name(tt)
        assert_nodes_name(tt)
        assert tt.bond_dimension == [e.dimension for e in tt.bond_edges]

        for node_actual, node_desired in zip(tt.node_list, list_tensors):
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
def tets_network_to_tt(in_shape, out_shape):
    in_node = random_node(in_shape)
    out_node = random_node(out_shape)
    network = Network(out_node[:], in_node[:])
    tt = network_to_tt(network)

    assert len(tt.node_list) == max(len(in_shape), len(out_shape))
    assert len(tt.in_edges) == len(in_shape)
    assert len(tt.out_edges) == len(out_shape)
    assert len(tt.bond_edges) == max(len(in_shape) - 1, len(out_shape) - 1)
    assert set(tt.nodes) == set(tt.node_list)
    assert_in_edges_name(tt)
    assert_out_edges_name(tt)
    assert_bond_edges_name(tt)
    assert_nodes_name(tt)
    assert tt.bond_dimension == [e.dimension for e in tt.bond_edges]
    assert_almost_equal(network.to_array(), tt.to_array())


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
    assert len(tt.node_list) == (1 if shape else 0)
    assert len(tt.bond_edges) == 0
    assert tt.dims == expected_dims
    assert_in_edges_name(tt)
    assert_out_edges_name(tt)
    assert_bond_edges_name(tt)
    assert_nodes_name(tt)
    assert tt.bond_dimension == [e.dimension for e in tt.bond_edges]
    np.testing.assert_allclose(array, tt.to_array().reshape(shape))

class TestTruncate():
    """These tests do not ensure that the operation is mathematically correct
    but rather that the resulting object is a tensor-train with proper
    nodes/edges."""


    @pytest.mark.parametrize("n", [5, 2, 1])
    def test_default(self, n):
        """We test that the default values return a tensor-train without any
        truncation. Note that the actual bond_dimension may still change."""
        # Create a random MPO
        d = 2
        bond_dimension = 100
        mpo = random_mpo(n, d, bond_dimension)

        copy = mpo.copy()

        error = mpo.truncate()
        assert len(error) == 0
        # The default values will truncate the final tensor train as the first
        # few bond dimensions are lager than they need to be.

        assert len(mpo.node_list) == n
        assert len(mpo.in_edges) == n
        assert len(mpo.out_edges) == n
        assert len(mpo.bond_edges) == n-1
        assert set(mpo.nodes) == set(mpo.node_list)
        assert_in_edges_name(mpo)
        assert_out_edges_name(mpo)
        assert_bond_edges_name(mpo)
        assert_nodes_name(mpo)
        expected_bond_dim = [4, 16, 64, 100, 100]
        assert mpo.bond_dimension == expected_bond_dim[:n-1]
        assert_almost_equal(copy.to_array(), mpo.to_array())


    @pytest.mark.parametrize("truncated_bond_dimension", [10, [1, 2, 3, 4] ], ids=["int", "list"])
    def test_truncate_bodn_dimension(self, truncated_bond_dimension):
        """We test that the argument bond_dimension truncates the tt train to
        return a tt with the specified bon_dimension."""
        n=5
        mpo = random_mpo(n, 2, 100)
        copy = mpo.copy()

        mpo.truncate(bond_dimension=truncated_bond_dimension)
        # The default values will truncate the final tensor train as the first
        # few bond dimensions are lager than they need to be.

        assert len(mpo.node_list) == n
        assert len(mpo.in_edges) == n
        assert len(mpo.out_edges) == n
        assert len(mpo.bond_edges) == n-1
        assert set(mpo.nodes) == set(mpo.node_list)
        assert_in_edges_name(mpo)
        assert_out_edges_name(mpo)
        assert_bond_edges_name(mpo)
        assert_nodes_name(mpo)
        expected_bond_dim = [4, 16, 64, 100]
        if isinstance(truncated_bond_dimension,int):
            truncated_bond_dimension = [truncated_bond_dimension]*len(expected_bond_dim)

        expected_bond_dim = [min(d, truncation) for d, truncation in zip(expected_bond_dim,
                                                        truncated_bond_dimension)]
        assert mpo.bond_dimension == expected_bond_dim[:n]