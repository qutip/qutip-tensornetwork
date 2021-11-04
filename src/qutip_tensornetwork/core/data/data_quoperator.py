import numbers
from qutip_tensornetwork.quantum import QuOperator

import qutip
import numpy as np
import tensornetwork as tn
from tensornetwork.visualization.graphviz import to_graphviz

__all__ = ['copy_quoperator', 'DataQuOperator']

def _is_shape_compatible_with_dims(shape, dims):
    return shape[0]==np.prod(dims[0]) and shape[1]==np.prod(dims[1])

def copy_quoperator(quoperator, copy=True, conjugate=False):
    """Copies nodes of QuOperator and returns a new QuOperator with the copied nodes and new edges linked."""
    if copy==False:
        return quoperator

    nodes_dict, edge_dict = tn.copy(quoperator.nodes, conjugate=conjugate)
    out_edges = [edge_dict[e] for e in quoperator.out_edges]
    in_edges = [edge_dict[e] for e in quoperator.in_edges]
    ref_nodes = [nodes_dict[n] for n in quoperator.ref_nodes]
    ignore_edges = [edge_dict[e] for e in quoperator.ignore_edges]
    return QuOperator(out_edges, in_edges, ref_nodes, ignore_edges)

def _check_valid_shape(shape, error_msg):
    """Check that shape is a tuple of len 2 with positive integer numbers."""
    if not (
        isinstance(shape, tuple)
        and len(shape) == 2
        and isinstance(shape[0], numbers.Integral)
        and isinstance(shape[1], numbers.Integral)
        and shape[0] > 0
        and shape[1] > 0
    ):
        raise ValueError(error_msg)

class DataQuOperator(qutip.core.data.Data):
    def __init__(self, data, shape=None, copy=True, dtype=np.complex128, backend=None):
        if backend is not None:
            raise NotImplementedError()

        if isinstance(data, QuOperator):
            self._from_quoperator(data, shape=shape, dtype=dtype)

        elif isinstance(data, (np.ndarray, list)):
            self._from_array_like(data, shape=shape, dtype=dtype)

        else:
            raise TypeError("Not a valid data type. It must be either a list, a quoperator or one of the valid backends in TensorNetwork.")

    def _from_array_like(self, data, shape, dtype):
        data = np.asarray(data, dtype=dtype)

        # inherit shape
        if shape is None:
            shape = data.shape

            if len(shape) == 0:
                shape = (1, 1)

            # Promote to a ket by default if passed 1D data.
            if len(shape) == 1:
                shape = (shape[0], 1)

        _check_valid_shape(shape, "Shape must be a 2-tuple of positive ints, but is " + repr(shape))
        data = data.reshape(shape)
        quoperator = QuOperator.from_tensor(data, [0], [1])

        self._quoperator = quoperator
        super().__init__(shape=shape)

    def _from_quoperator(self, quoperator, shape, dtype):
        shape_quop = (np.prod(quoperator.out_space, dtype=int), np.prod(quoperator.in_space, dtype=int))

        # shape_data should never be wrong but I added this for sanity check
        _check_valid_shape(shape_quop, """The input QuOperator's shape must be a tuple of
            len 2 with positive integer numbers.""" + repr(shape_quop))

        if shape is not None:
            _check_valid_shape(shape, "Shape must be a 2-tuple of positive ints, but is " + repr(shape))

        if shape is None:
            shape = shape_quop

        # Due to ambiguity on how to reshape the nodes in an arbitrary tensor network, we will not allow 
        # reshaping with shape argument if input is a quoperator.
        elif (shape_quop[0] != shape[0]) or (shape_quop[1] != shape[1]):
            raise ValueError("Shape mismatch for QuOperator shape" + repr(shape_data)
                             + "and input argument shape, " + repr(shape))

        # We only check one node when we should check multiple nodes.
        if dtype is not None and dtype != quoperator.nodes.pop().dtype:
            raise NotImplementedError("The provided dtype is different from the quoperators dtype."
                                     + "We do not support changing dtype.")

        self._quoperator = quoperator
        super().__init__(shape=shape)

    @property
    def shape(self):
        return (np.prod(self.dims[0], dtype=int), np.prod(self.dims[1], dtype=int))

    @property
    def dims(self):
        return [self._quoperator.out_space, self._quoperator.in_space]

    def conj(self):
        return DataQuOperator(copy_quoperator(self._quoperator, conjugate=True), copy=False)

    def adjoint(self):
        # Adjoint method already creates a copy
        return DataQuOperator(self._quoperator.adjoint(), copy=False)

    def to_array(self):
        return self._quoperator.eval().reshape(self.shape)

    def transpose(self):
        return NotImplementedError()

    def trace(self):
        return self._quoperator.trace().eval()

    def copy(self):
        return DataQuOperator(self._quoperator, copy=True)

    def _repr_svg_(self):
        return to_graphviz(self._quoperator.nodes)._repr_svg_()
