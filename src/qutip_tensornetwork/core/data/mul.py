import qutip
from .network import Network

__all__ = ["mul_network", "imul_network", "neg_network"]


def mul_network(network, value):
    """
    Perform the operation
        ``value * network``
    where ``network`` is an instance of ``Network`` and value is a ``scalar``
    value.
    """
    return network * value

def imul_network(network, value):
    """
    Perform the in-place operation
        ``value * network``
    where ``network`` is an instance of ``Network`` and value is a ``scalar``
    value.
    """
    network *= value
    return network

def neg_network(network):
    """
    Perform the operation
        ``-1 * network``
    where ``network`` is an instance of ``Network``.
    """
    return network * -1

qutip.data.mul.add_specialisations([(Network, Network, mul_network)])
qutip.data.imul.add_specialisations([(Network, Network, imul_network)])
qutip.data.neg.add_specialisations([(Network, Network, neg_network)])
