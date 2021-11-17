import qutip
from .network import Network
import tensornetwork as tn


def from_quoperator(quoperator):
    return Network(quoperator.out_edges, quoperator.in_edges,
                   quoperator.ref_nodes, quoperator.ignore_edges)

# Conversion function
def _network_from_dense(dense):
    array = dense.to_array()
    if array.shape[0]==1 and array.shape[1]!=1:
        array = array.reshape(array.shape[1])
        node = tn.Node(array)
        return Network([], node[:])

    elif array.shape[0]!=1 and array.shape[1]==1:
        array = array.reshape(array.shape[0])
        node = tn.Node(array)
        return Network(node[:], [])

    elif array.shape[0]==1 and array.shape[1]==1:
        array = array.reshape(())
        node = tn.Node(array)
        return Network([], [], nodes=[node])

    else:
        node = tn.Node(array)
        return Network(node[0:1], node[1:])

def _network_to_dense(network):
    return qutip.data.Dense(network.to_array(), copy=False)


# Register the data layer
qutip.data.to.add_conversions(
    [
        (Network, qutip.data.Dense, _network_from_dense),
        (qutip.data.Dense, Network, _network_to_dense),
    ]
)

# User friendly name for conversion with `to` or Qobj creation functions:
qutip.data.to.register_aliases(["network"], Network)

# This will be used once init for quantum in TensroNetwork is included.
# Conversion functions from QuOperators to Network data class.
# def is_quoperator(quoperator):
    # return isinstance(quoperator, QuOperator)

# qutip.data.create.add_creators(
    # [
        # (is_quoperator, from_quoperator, 85),
    # ]
)
