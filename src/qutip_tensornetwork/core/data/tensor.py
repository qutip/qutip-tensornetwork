import qutip
from .network import Network


__all__ = ["tensor_network"]

def tensor_network(left, right):
    """Returns a network obtained from the tensor operation between left and
    right."""
    return left.tensor(right)

qutip.data.kron.add_specialisations([(Network, Network, Network, tensor_network)])
