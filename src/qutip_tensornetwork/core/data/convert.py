import qutip
from .network import Network
from .tensor_network.quantum import QuOperator

__all__ = []

def is_quoperator(quoperator):
    return isinstance(quoperator, QuOperator)

def from_quoperator(quoperator):
    return NotImplementedError()

# Conversion function
def _quop_from_dense(dense):
    return DataQuOperator(dense.to_array())

def _quop_to_dense(quoperator):
    return qutip.data.Dense(quoperator.to_array(), copy=False)


# Register the data layer
qutip.data.to.add_conversions(
    [
        (DataQuOperator, qutip.data.Dense, _quop_from_dense),
        (qutip.data.Dense, DataQuOperator, _quop_to_dense),
    ]
)

# User friendly name for conversion with `to` or Qobj creation functions:
qutip.data.to.register_aliases(["quoperator", "tensor_network"], DataQuOperator)

qutip.data.create.add_creators(
    [
        (is_quoperator, DataQuOperator, 85),
    ]
)
