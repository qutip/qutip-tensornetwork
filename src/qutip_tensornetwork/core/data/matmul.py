import tensornetwork as tn
import qutip
from .network import Network

__all__ = ["matmul_network"]


def matmul_network(left, right, scale=1, out=None):
    """
    Perform the operation
        ``out := scale * (left @ right) + out``
    where `left`, `right` and `out` are matrices.  `scale` is a complex scalar,
    defaulting to 1. If `out` is not given it is assumed to be 0.

    Notes
    -----
    out as parameter is not included yet.
    """
    # This part of the operation wont be added until we think about what to do
    # with addition.
    if out is not None:
        raise NotImplementedError(
            "The addition of Network data layer instances is not yet implemented"
        )

    if scale != 1:
        out._nodes = out._nodes | tn.Node(scale)

    return left @ right


qutip.data.matmul.add_specialisations([(Network, Network, Network, matmul_network)])
