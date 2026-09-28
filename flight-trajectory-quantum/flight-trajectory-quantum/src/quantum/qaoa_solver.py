from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from scipy.optimize import minimize

from src.optimization.qubo import QUBOResult
from src.quantum.ising import qubo_to_ising
from src.utils.config import QAOAConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

BYTES_PER_AMPLITUDE = 16  # complex128
# Gate names qiskit-aer's simulator executes natively; once a decomposed
# circuit contains only these, further decomposition is unnecessary.
_BASIC_GATE_NAMES = {
    "u", "u1", "u2", "u3", "cx", "cz", "rz", "rx", "ry", "id", "x", "y", "z",
    "h", "s", "sdg", "t", "tdg", "swap", "rzz", "rxx", "ryy", "p", "measure", "barrier",
}


class QuantumSimulatorError(Exception):
    """Raised when a requested quantum simulation is not feasible on this
    machine (or otherwise cannot be carried out), instead of letting the
    underlying simulator crash or exhaust memory."""


def _check_qubit_feasibility(num_qubits: int, max_qubits: int) -> None:
    if num_qubits <= max_qubits:
        return
    required_mb = (2 ** num_qubits) * BYTES_PER_AMPLITUDE / (1024 ** 2)
    raise QuantumSimulatorError(
        f"This QUBO needs {num_qubits} qubits, which exceeds the configured "
        f"statevector-simulation safety limit of {max_qubits} qubits "
        f"(QAOAConfig.max_qubits). Simply holding the statevector would need "
        f"approximately {required_mb:,.0f} MB. This is a genuine simulator "
        f"scaling limit, not a bug -- reduce the network size / edge "
        f"density (fewer qubits), or raise max_qubits if more memory is "
        f"available. See docs/experimental_methodology.md."
    )


def _fully_decompose(circuit: QuantumCircuit, max_iterations: int = 15) -> QuantumCircuit:
    """Unfold QAOAAnsatz's high-level blocks (its own custom 'QAOA'
    instruction, nested cost/mixer sub-circuits) down to gates Aer can
    execute directly, without invoking the transpiler's synthesis/
    optimization passes at all."""
    current = circuit
    for _ in range(max_iterations):
        names = {instr.operation.name for instr in current.data}
        if names <= _BASIC_GATE_NAMES:
            return current
        current = current.decompose()
    raise QuantumSimulatorError(
        f"Could not fully decompose the QAOA ansatz to primitive gates "
        f"within {max_iterations} decompose() passes; remaining "
        f"instructions: {sorted(names - _BASIC_GATE_NAMES)}."
    )


@dataclass
class QAOAResult:
    qubo: QUBOResult
    num_qubits: int
    reps: int
    shots: int
    optimizer: str
    optimal_params: np.ndarray
    trained_ising_energy: float          # exact <H_C> at the trained parameters
    trained_qubo_energy: float           # trained_ising_energy + ising_offset == expected QUBO energy
    num_optimizer_iterations: int
    optimizer_converged: bool
    # raw measurement counts, keyed by VARIABLE-ORDER bitstrings "x0 x1 ... x_{n-1}"
    # (i.e. counts["101"] means x0=1, x1=0, x2=1) -- already corrected from
    # Qiskit's little-endian qubit ordering so downstream code never has to
    # think about bit order again.
    counts: dict[str, int]
    circuit_build_seconds: float
    training_seconds: float
    sampling_seconds: float


def _bind_variable_order(raw_bitstring: str) -> str:
    """Qiskit's classical register string is little-endian: the leftmost
    character is the highest qubit index. Since variable x_i is mapped to
    qubit i (see src/quantum/ising.py), reversing the string turns it into
    "x0 x1 ... x_{n-1}" order."""
    return raw_bitstring[::-1]


def solve_qaoa(qubo: QUBOResult, config: QAOAConfig) -> QAOAResult:
    t0 = time.perf_counter()
    cost_operator, ising_offset = qubo_to_ising(qubo)
    num_qubits = cost_operator.num_qubits
    if num_qubits != qubo.num_variables:
        raise RuntimeError(
            f"Ising conversion produced {num_qubits} qubits but QUBO has "
            f"{qubo.num_variables} variables -- variable/qubit mapping is broken."
        )
    _check_qubit_feasibility(num_qubits, config.max_qubits)

    ansatz = QAOAAnsatz(cost_operator=cost_operator, reps=config.reps)
    isa_ansatz = _fully_decompose(ansatz)
    isa_cost_operator = cost_operator  # decompose-only never permutes qubits, so no layout remap needed
    t1 = time.perf_counter()

    estimator = EstimatorV2()
    n_calls = 0

    def expectation(params: np.ndarray) -> float:
        nonlocal n_calls
        n_calls += 1
        job = estimator.run([(isa_ansatz, isa_cost_operator, [params])])
        return float(job.result()[0].data.evs[0])

    rng = np.random.default_rng(config.seed)
    x0 = rng.uniform(0.0, np.pi, ansatz.num_parameters)

    logger.info(
        f"Training QAOA: {num_qubits} qubits, reps={config.reps} "
        f"({ansatz.num_parameters} parameters), optimizer={config.optimizer}, "
        f"max_iter={config.max_iter}"
    )
    opt_result = minimize(
        expectation, x0, method=config.optimizer, options={"maxiter": config.max_iter}
    )
    t2 = time.perf_counter()

    logger.info(
        f"QAOA training finished: {n_calls} evaluations, "
        f"trained ising energy={opt_result.fun:.4f}, converged={bool(opt_result.success)}"
    )

    final_circ = isa_ansatz.assign_parameters(opt_result.x)
    final_circ.measure_all()
    sampler = SamplerV2(seed=config.seed)
    job = sampler.run([final_circ], shots=config.shots)
    raw_counts = job.result()[0].data.meas.get_counts()
    counts = {_bind_variable_order(k): v for k, v in raw_counts.items()}
    t3 = time.perf_counter()

    return QAOAResult(
        qubo=qubo,
        num_qubits=num_qubits,
        reps=config.reps,
        shots=config.shots,
        optimizer=config.optimizer,
        optimal_params=np.asarray(opt_result.x),
        trained_ising_energy=float(opt_result.fun),
        trained_qubo_energy=float(opt_result.fun) + ising_offset,
        num_optimizer_iterations=n_calls,
        optimizer_converged=bool(opt_result.success),
        counts=counts,
        circuit_build_seconds=t1 - t0,
        training_seconds=t2 - t1,
        sampling_seconds=t3 - t2,
    )
