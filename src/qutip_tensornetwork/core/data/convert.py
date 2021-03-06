import qutip
from .network import Network
import tensornetwork as tn


def _network_to_dense(network):
    return qutip.data.Dense(network.to_array(), copy=False)


# Register the data layer
qutip.data.to.add_conversions(
    [
        (Network, qutip.data.Dense, Network.from_2d_array),
        (qutip.data.Dense, Network, _network_to_dense),
    ]
)

# User friendly name for conversion with `to` or Qobj creation functions:
qutip.data.to.register_aliases(["network"], Network)
