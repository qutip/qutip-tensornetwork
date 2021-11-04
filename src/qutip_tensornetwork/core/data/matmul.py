import qutip
from .data_quoperator import DataQuOperator

__all__ = ["matmul_dataquoperator"]


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

def check_dims(dims_in, dims_out):
    if dims_in != dims_out:
        raise ValueError(
            "incompatible dimensions, got "
            + str(dims_in)
            + " and "
            + str(dims_out)
        )

def matmul_dataquoperator(left, right, scale=1, out=None):
    """
    Perform the operation
        ``out := scale * (left @ right) + out``
    where `left`, `right` and `out` are matrices.  `scale` is a complex scalar,
    defaulting to 1. If `out` is not given it is assumed to be 0.
    """
    return DataQuOperator(left._quoperator @ right._quoperator, dtype=None)


qutip.data.matmul.add_specialisations([(DataQuOperator, DataQuOperator,
                                        DataQuOperator, matmul_dataquoperator)])
