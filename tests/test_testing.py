"""The tests in this file are still very basic and more care is needed to
ensure assert_network_close works properly. These will be included once the
function is changed to its final form where it will be able to handle networks
that have similar nodes."""

from qutip_tensornetwork import testing
from qutip_tensornetwork import Network
from .core.data.conftest import random_node
import pytest

def test_network_same():
    node1 = random_node((2,2))
    node2 = node1*3
    node1[0] ^ node2[0]
    network = Network([node1[1]], [node2[1]])
    network2 = network.copy()
    testing.assert_network_close(network, network2)

def test_network_different():
    node1 = random_node((2,2))
    # We multiply by three to ensure no nodes are equal
    node2 = node1 * 3
    node1[0] ^ node2[0]
    network = Network([node1[1]], [node2[1]])

    node1 = random_node((2,2))
    # Multiplication by a different number ensures that networks will be differet
    node2 = node1 * 5

    node1[0] ^ node2[0]
    network2 = Network([node1[1]], [node2[1]])
    with pytest.raises(AssertionError):
        testing.assert_network_close(network, network2)

