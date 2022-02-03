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


def random_mpo(n, d, bond_dimension):
    """Create a random mpo with n sites d dimension per site and
    bond_dimesnion."""
    if n > 1:
        list_tensors = [random_node((d, d, bond_dimension)) - 1 / 2]
        list_tensors += [
            random_node((d, d, bond_dimension, bond_dimension)) - 1 / 2
            for _ in range(n - 2)
        ]
        list_tensors += [random_node((d, d, bond_dimension)) - 1 / 2]
        mpo = FiniteTT.from_nodes(list_tensors)
    elif n == 1:
        node = tn.Node(random_node((d, d)))
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
class Test_node_list:
    def test_ket(self, n):
        d = 3
        chi = 10
        list_tensors = [random_node((d, chi))]
        list_tensors += [random_node((d, chi, chi)) for _ in range(n - 2)]
        list_tensors += [random_node((d, chi))]

        tt = FiniteTT.from_nodes(list_tensors)

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
            assert (node_actual.tensor == node_desired.tensor).all()

    def test_op(self, n):
        d = 3
        chi = 10
        list_tensors = [random_node((d, d, chi))]
        list_tensors += [random_node((d, d, chi, chi)) for _ in range(n - 2)]
        list_tensors += [random_node((d, d, chi))]

        tt = FiniteTT.from_nodes(list_tensors)

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

    def test_node(self, n):
        d = 3
        chi = 10
        list_tensors = [random_node((d, d, chi))]
        list_tensors += [random_node((d, d, chi, chi)) for _ in range(n - 2)]
        list_tensors += [random_node((d, d, chi))]
        list_tensors = [tn.Node(tensor) for tensor in list_tensors]

        tt = FiniteTT.from_nodes(list_tensors)

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
        list_tensors = [random_node((d, d, chi + 1))]
        list_tensors += [random_node((d, d, chi, chi)) for _ in range(n - 2)]
        list_tensors += [random_node((d, d, chi))]
        list_tensors = [tn.Node(tensor) for tensor in list_tensors]

        with pytest.raises(ValueError):
            FiniteTT.from_nodes(list_tensors)

    def test_incorrect_shape_raises(self, n):
        d = 3
        chi = 10
        list_tensors = [random_node((d, d, chi, chi))]
        list_tensors += [random_node((d, d, chi, chi)) for _ in range(n - 2)]
        list_tensors += [random_node((d, d, chi))]
        list_tensors = [tn.Node(tensor) for tensor in list_tensors]

        with pytest.raises(ValueError):
            FiniteTT.from_nodes(list_tensors)

        if n > 2:
            list_tensors = [random_node((d, d, chi))]
            list_tensors += [random_node((d, d, chi, chi, 10)) for _ in range(n - 2)]
            list_tensors += [random_node((d, d, chi))]
            list_tensors = [tn.Node(tensor) for tensor in list_tensors]

            with pytest.raises(ValueError):
                FiniteTT.from_nodes(list_tensors)


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


class TestTruncate:
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
        assert_no_truncation(error)
        # The default values will truncate the final tensor train as the first
        # few bond dimensions are larger than they need to be.

        assert len(mpo.train_nodes) == n
        assert len(mpo.in_edges) == n
        assert len(mpo.out_edges) == n
        assert len(mpo.bond_edges) == n - 1
        assert set(mpo.nodes) == set(mpo.train_nodes)
        assert_in_edges_name(mpo)
        assert_out_edges_name(mpo)
        assert_bond_edges_name(mpo)
        assert_nodes_name(mpo)
        expected_bond_dim = [4, 16, 64, 100, 100]
        assert mpo.bond_dimension == expected_bond_dim[: n - 1]
        assert_almost_equal(copy.to_array(), mpo.to_array())

    @pytest.mark.parametrize(
        "truncated_bond_dimension", [10, [1, 2, 3, 4]], ids=["int", "list"]
    )
    def test_truncate_bodn_dimension(self, truncated_bond_dimension):
        """We test that the argument bond_dimension truncates the tt train to
        return a tt with the specified bon_dimension."""
        n = 5
        mpo = random_mpo(n, 2, 100)
        copy = mpo.copy()

        error = mpo.truncate(bond_dimension=truncated_bond_dimension)
        # The default values will truncate the final tensor train as the first
        # few bond dimensions are lager than they need to be.

        assert isinstance(error, list)
        assert len(mpo.train_nodes) == n
        assert len(mpo.in_edges) == n
        assert len(mpo.out_edges) == n
        assert len(mpo.bond_edges) == n - 1
        assert set(mpo.nodes) == set(mpo.train_nodes)
        assert_in_edges_name(mpo)
        assert_out_edges_name(mpo)
        assert_bond_edges_name(mpo)
        assert_nodes_name(mpo)
        expected_bond_dim = [4, 16, 64, 100]
        if isinstance(truncated_bond_dimension, int):
            truncated_bond_dimension = [truncated_bond_dimension] * len(
                expected_bond_dim
            )

        expected_bond_dim = [
            min(d, truncation)
            for d, truncation in zip(expected_bond_dim, truncated_bond_dimension)
        ]
        assert mpo.bond_dimension == expected_bond_dim[:n]

    @pytest.mark.parametrize("n_bonds", [3, 5])
    def test_raises_wrong_len_argument(self, n_bonds):
        bond_dimension = [2, 2, 2, 2]
        mpo = random_mpo(n_bonds + 1, 2, 10)

        with pytest.raises(ValueError) as e:
            mpo.truncate(bond_dimension=bond_dimension)

        assert (
            "4 values where provided for `bond_dimension` but there"
            f" are {n_bonds} bond edges." in str(e)
        )

        max_truncation_err = [1.0, 1.0, 1.0, 1.0]
        with pytest.raises(ValueError) as e:
            mpo.truncate(max_truncation_err=max_truncation_err)

        assert (
            "4 values where provided for `max_truncation_err` but there"
            f" are {n_bonds} bond edges." in str(e)
        )

    def test_numerical_correctness(self):
        """This tests that the truncation is done as expected using a diagonal
        set of nodes as example."""

        def arange_diag(shape, start=None):
            """Returns a diagonal numpy array with shape ``shape`` that has in
            its diagonal values from ``np.arange``"""
            if start is None:
                start = min(shape)
            # We sort values from large to small as it is expected from the
            # svd decomposition that comes later.
            diag = np.arange(start, 0, -1)
            array = np.zeros(shape)
            np.fill_diagonal(array, diag)
            return array

        # We create a MPO with tensors that are "diagonal". By this we mean
        # that during the truncation process the nodes are reshaped into a
        # matrix that will be diagonal and hence the SVD decomposition will be
        # trivial (u and v are identity). We use values of chi so that truncate
        # without arguments does not change the nodes.
        d = 2
        chi = 10
        n = 4

        list_chi = [d ** (2 * i) for i in range(1, n)]
        list_tensors = [arange_diag((d * d, list_chi[0])).reshape((d, d, list_chi[0]))]

        list_tensors += [
            arange_diag((d * d * chi1, chi2)).reshape((d, d, chi1, chi2))
            for chi1, chi2 in zip(list_chi[:-1], list_chi[1:])
        ]
        list_tensors += [
            arange_diag((d * d, list_chi[-1])).reshape((d, d, list_chi[-1]))
        ]
        list_tensors = [tn.Node(tensor) for tensor in list_tensors]
        initial = FiniteTT.from_nodes(list_tensors)

        mpo = initial.copy()
        for node1, node2 in zip(mpo.train_nodes, initial.train_nodes):
            assert_almost_equal(node1.tensor, node2.tensor)

        # We check that no node is changed after truncate as no different bond
        # dimension was specified
        error = mpo.truncate()
        for i, (node1, node2) in enumerate(zip(mpo.train_nodes, initial.train_nodes)):
            assert_almost_equal(
                node1.tensor, node2.tensor, err_msg=f"Error in node {i}"
            )

        assert error == [[], [], []]

        # expected mpo and errors from a truncate(2) method call. The expected
        # mpo consist on the truncation of all but the first two values of
        # each node. The error will show as if less values were truncated but
        # this is because some are truncated during the contraction of v and
        # the next node.
        expected_error = [
            [2.0, 1.0],
            [12.0, 11.0, 8.0, 7.0, 4.0, 3.0],
            [48.0, 47.0, 32.0, 31.0, 16.0, 15.0],
        ]

        list_chi = [2] * (n - 1)  # Bond dimension 2 for all bonds
        start = [d ** (2 * i) for i in range(1, n)] + [4]
        list_tensors = [
            arange_diag((d * d, list_chi[0]), start[0]).reshape((d, d, list_chi[0]))
        ]

        list_tensors += [
            arange_diag((d * d * chi1, chi2), s).reshape((d, d, chi1, chi2))
            for chi1, chi2, s in zip(list_chi[:-1], list_chi[1:], start[1:])
        ]
        list_tensors += [
            arange_diag((d * d, list_chi[-1]), start[-1]).reshape((d, d, list_chi[-1]))
        ]

        list_tensors = [tn.Node(tensor) for tensor in list_tensors]
        expected_mpo = FiniteTT.from_nodes(list_tensors)

        error = mpo.truncate(2)
        for i, (node1, node2) in enumerate(
            zip(mpo.train_nodes, expected_mpo.train_nodes)
        ):
            assert_almost_equal(
                node1.tensor, node2.tensor, err_msg=f"Error in node {i}"
            )

        assert error == expected_error


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
def test_copy(in_shape, out_shape):
    in_node = random_node(in_shape)
    out_node = random_node(out_shape)
    tt = FiniteTT(out_node[:], in_node[:])
    copy = tt.copy()

    assert isinstance(copy, FiniteTT)
    assert len(copy.train_nodes) == max(len(in_shape), len(out_shape))
    assert len(copy.in_edges) == len(in_shape)
    assert len(copy.out_edges) == len(out_shape)
    assert len(copy.bond_edges) == max(len(in_shape) - 1, len(out_shape) - 1)
    assert set(copy.nodes) == set(copy.train_nodes)
    assert_in_edges_name(copy)
    assert_out_edges_name(copy)
    assert_bond_edges_name(copy)
    assert_nodes_name(copy)
    assert copy.bond_dimension == [e.dimension for e in copy.bond_edges]
    assert_almost_equal(copy.to_array(), tt.to_array())


def assert_no_truncation(truncate_values):
    for error in truncate_values:
        assert len(error) == 0
