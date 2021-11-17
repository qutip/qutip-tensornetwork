"""This module tests that the basic conversion and creation methods in qutip
work properly with networks."""
import numpy as np
import pytest
import qutip
from qutip_tensornetwork.core.data import Network
from qutip import data
from .conftest import random_complex_network, random_numpy_dense

_network = random_complex_network(2)

@pytest.mark.parametrize("copy", [True, False])
def test_create(copy):
    qobj = qutip.Qobj(_network, copy=copy)
    assert isinstance(qobj.data, Network)
    if copy:
        assert qobj.data is not _network
        np.testing.assert_allclose(qobj.full(), _network.to_array())
        #TODO: This does not test that both networks are copies directly but
        # testing that is not straightforward
    else:
        assert qobj.data is _network


# We only test for the dense to network because we expect the rest to work if 
# this one works.
@pytest.mark.parametrize("shape",
                         [(2,2), (2,1), (1,2), (1,1)],
                         ids=["operator", "ket", "bra", "scalar"])
@pytest.mark.parametrize("dtype",
                         ["network", Network],
                         ids=["string `network`", "data type"])
def test_convert_dense_to_network(shape, dtype):
    array = random_numpy_dense(shape)
    qobj = qutip.Qobj(array)
    qobj = qobj.to(dtype)
    assert isinstance(qobj.data, Network)
    np.testing.assert_allclose(qobj.full(), array)

def test_network_to_dense():
    qobj = qutip.Qobj(_network)
    qobj = qobj.to("dense")
    assert isinstance(qobj.data, qutip.data.Dense)
    np.testing.assert_allclose(qobj.full(), _network.to_array())
