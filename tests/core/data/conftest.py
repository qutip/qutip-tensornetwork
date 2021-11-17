import warnings

import numpy as np
import qutip
import tensornetwork as tn
from qutip_tensornetwork import Network
from qutip_tensornetwork.core.data.convert import _network_from_dense

def random_numpy_dense(shape):
    """Generate a random numpy dense matrix with the given shape."""
    out = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    return out

def random_node(shape):
    """Generate a random node with the given shape."""
    return tn.Node(random_numpy_dense(shape))

def random_one_node_network(shape):
    """Returns a network with a single node of given shape."""
    array = random_numpy_dense(shape)
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

def random_complex_network(dim):
    """Returns a network with the following nodes and edges:
            *
           / \
          * - *
          |   |

    where * is a node and the lines are edges. The nodes contain random data
    and the edges had dimension `dim`.

    This network is employed as example of relatively complex network.
    """
    top = random_node((dim, dim))
    bottom_left = random_node((dim, dim, dim))
    bottom_right = random_node((dim, dim, dim))

    bottom_left[0] ^ bottom_right[0]
    bottom_left[1] ^ top[0]
    bottom_right[1] ^ top[1]

    return Network([bottom_left[2],bottom_right[2]], [], nodes=[top,
                                                                bottom_left,
                                                                bottom_right])