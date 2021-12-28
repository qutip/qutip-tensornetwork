import qutip
from qutip_tensornetwork.core.data import Network
import warnings

__all__ = ["transpose_network", "conj_network", "adjoint_network"]


def transpose_network(network):
    return network.transpose()


def conj_network(network):
    return network.conj()


def adjoint_network(network):
    return network.adjoint()


qutip.data.transpose.add_specialisations(
    [
        (Network, Network, transpose_network),
    ]
)

qutip.data.conj.add_specialisations(
    [
        (Network, Network, conj_network),
    ]
)
qutip.data.adjoint.add_specialisations(
    [
        (Network, Network, adjoint_network),
    ]
)
