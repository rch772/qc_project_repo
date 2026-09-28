from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram

from src.optimization.qubo import QUBOResult


def qubo_to_ising(qubo: QUBOResult) -> tuple[SparsePauliOp, float]:
    """Return (cost_operator, ising_offset) such that, for a computational
    basis state |x>, <x| cost_operator |x> + ising_offset == qubo.energy(x)."""
    qp = QuadraticProgram()
    for i in range(qubo.num_variables):
        qp.binary_var(name=f"x{i}")

    linear = {f"x{i}": coeff for i, coeff in qubo.linear.items()}
    quadratic = {
        (f"x{i}", f"x{j}"): coeff for (i, j), coeff in qubo.quadratic.items()
    }
    qp.minimize(linear=linear, quadratic=quadratic, constant=qubo.constant)

    qubit_op, ising_offset = qp.to_ising()
    return qubit_op, float(ising_offset)
