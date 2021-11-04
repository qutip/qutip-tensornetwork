import tensornetwork as tn
import qutip
from .network import Network

__all__ = ["matmul_dataquoperator"]
def get_compatible_dims():
    return NotImplementedError()

# TODO: I realized that this function is not what I need as I need to reshape
# the in_edges and out_edges of the same network. This will be accomplised by
# the reshape method in Network and the compatible_dims function that will
# return a compatible dimensio with both left and right if any. Although the
# algorithms are going to be similar to this one.
def match_edges_by_split(out_edges, in_edges):
    # Shallow copy to allow popping from out_edges
    _out_edges = out_edges[:] _in_edges = in_edges[:]

    new_in_edges = []
    new_out_edges = []

    # What to do id one of the dimensions is 1?
    e_in = in_edges.pop()
    e_out = out_edges.pop()

    try:
        while True:

            if e_in.dimension == e_out.dimension:
                new_in_edges.append(e_in)
                new_out_edges.append(e_out)

                if len(in_edges) == 0 and len(out_edges) == 0:
                    break

                e_in = in_edges.pop()
                e_out = out_edges.pop()


            elif e_in.dimension > e_out.dimension:
                # IndexError will be caught and by try/except which will then
                # raise the appropriate error
                if e_in.dimension%e_out.dimension != 0:
                    raise IndexError()
                else:
                    new_shape=(e_out.dimension, e_in.dimension//e_out.dimension)
                    new_e_in, e_in = tn.split_edge(e_in, shape=new_shape)

                    new_out_edges.append(e_out)
                    new_in_edges.append(new_e_in)

                    e_out = out_edges.pop()

            elif e_in.dimension < e_out.dimension:
                # IndexError will be caught and by try/except which will then
                # raise the appropriate error
                if e_out.dimension%e_in.dimension != 0:
                    raise IndexError()
                else:
                    new_shape=(e_in.dimension, e_out.dimension//e_in.dimension)
                    new_e_out, e_out = tn.split_edge(e_out, shape=new_shape)

                    new_out_edges.append(new_e_out)
                    new_in_edges.append(e_in)

                    e_in = in_edges.pop()

    except IndexError:
        in_dims = []
        raise ValueError("Edges are not compatible. The dimensions of in_edges: " +
                         str(in_dims) + " whereas for out_edges: "+str())


    new_out_edges.reverse()
    new_in_edges.reverse()
    return new_out_edges, new_in_edges





def _check_shape(left, right, out):
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            "incompatible matrix shapes " + str(left.shape) + " and "
            + str(right.shape)
        )
    if (
        out is not None
        and out.shape[0] != left.shape[0]
        and out.shape[1] != right.shape[1]
    ):
        raise ValueError(
            "incompatible output shape, got "
            + str(out.shape)
            + " but needed "
            + str((left.shape[0], right.shape[1]))
        )

def matmul_network(left, right, scale=1, out=None):
    """
    Perform the operation
        ``out := scale * (left @ right) + out``
    where `left`, `right` and `out` are matrices.  `scale` is a complex scalar,
    defaulting to 1. If `out` is not given it is assumed to be 0.
    """
    if scale!=1:
        return NotImplementedError()
    dims = get_compatible_dims(right.dims[1], left.dims[0])
    right.reshape(dims)
    left.reshape(dims)

    # Copy all nodes involved in the two operators.
    # We must do this separately for self and other, in case self and other
    # are defined via the same network components (e.g. if self === other).
    nodes_dict1, edges_dict1 = copy(left.nodes, False)
    nodes_dict2, edges_dict2 = copy(right.nodes, False)


    # connect edges to create network for the result
    for (e1, e2) in zip(in_edges, out_edges):
      _ = edges_dict1[e1] ^ edges_dict2[e2]

    in_edges = [edges_dict2[e] for e in other.in_edges]
    out_edges = [edges_dict1[e] for e in self.out_edges]
    ref_nodes = ([n for _, n in nodes_dict1.items()] +
                 [n for _, n in nodes_dict2.items()])
    ignore_edges = ([edges_dict1[e] for e in self.ignore_edges] +
                    [edges_dict2[e] for e in other.ignore_edges])

    return Network(out_edges, in_edges, ref_nodes, ignore_edges)


qutip.data.matmul.add_specialisations([(Network, Network,
                                        Network, matmul_network)])
