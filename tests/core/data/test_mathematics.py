"""These are a complete set of tests that check that the specialisations for
the Network class are numerically equivalent to the Dense class. For tests
regarding the structure of the output network, we refer to the tests in
test_network. This tests only check for single node networks."""
import numpy as np
import tensornetwork as tn
import pytest

import qutip.tests.core.data.test_mathematics as testing

from qutip_tensornetwork.core.data import Network
from qutip_tensornetwork import data
from . import conftest

testing._ALL_CASES = {
    Network: lambda shape: [lambda: conftest.random_one_node_network(shape)],
}
testing._RANDOM = {
    Network: lambda shape: [lambda: conftest.random_one_node_network(shape)],
}

class TestMatmul(testing.TestMatmul):
    specialisations = [
        pytest.param(data.matmul_network, Network, Network, Network),
    ]
