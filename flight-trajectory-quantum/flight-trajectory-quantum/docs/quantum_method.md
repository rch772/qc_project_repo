# Quantum Method: QAOA

Implementation: `src/quantum/ising.py` (QUBO -> Ising) and
`src/quantum/qaoa_solver.py` (the QAOA circuit, training, and sampling).

## Why QAOA

QAOA (the Quantum Approximate Optimization Algorithm) was chosen over
alternatives (VQE with a generic ansatz, quantum annealing) because: (1) it
is designed specifically for QUBO/Ising-type combinatorial objectives,
which is exactly what `build_qubo` produces; (2) Qiskit ships a ready-made,
well-documented `QAOAAnsatz` circuit-library primitive that constructs the
correct cost/mixer alternation directly from a cost Hamiltonian, which was
verified to exist and work in the pinned Qiskit 2.5.2 / qiskit-aer 0.17.2
environment (see `docs/experimental_methodology.md` for what "verified" 
means in practice here); (3) it needs no quantum annealing hardware or
cloud access -- a local Aer simulator is sufficient for the scales this
project targets.

## Qubits and the cost Hamiltonian

Each QUBO variable `x_e` maps to one qubit, via the standard
substitution `x_e -> (I - Z_e) / 2` (`x_e = 0` <-> eigenvalue `+1` of `Z_e`,
`x_e = 1` <-> eigenvalue `-1`). Substituting this into

```
H(x) = constant + sum_i linear[i] x_i + sum_{i<j} quadratic[i,j] x_i x_j
```

produces a diagonal Ising Hamiltonian `H_C = sum_i h_i Z_i + sum_{i<j} J_{ij} Z_i Z_j + offset`,
whose eigenvalues (in the computational basis) are exactly the QUBO
energies of the corresponding bitstrings. `src/quantum/ising.py` does not
do this substitution by hand -- it hands the QUBO's linear/quadratic/constant
coefficients to `qiskit_optimization.QuadraticProgram.to_ising()`, which
performs the substitution and returns `(SparsePauliOp, offset)`. This was
confirmed empirically (not just assumed) to preserve the variable <-> qubit
index mapping (`x_i` <-> qubit `i`) by building a small 3-variable example
by hand, converting it, and checking that the resulting Pauli-string
positions matched the expected variables term by term before this was
relied on anywhere else in the project.

## The QAOA ansatz

For `reps = p` layers, Qiskit's `QAOAAnsatz(cost_operator=H_C, reps=p)`
builds:

```
|psi(gamma, beta)> = [ exp(-i*beta_p*H_M) exp(-i*gamma_p*H_C) ] ... [ exp(-i*beta_1*H_M) exp(-i*gamma_1*H_C) ] |+>^n
```

starting from the uniform superposition `|+>^n` (all `n` qubits in
`(|0>+|1>)/sqrt(2)`), where `H_M = sum_i X_i` is the standard transverse-field
mixer. This gives `2p` trainable real parameters (`gamma_1..gamma_p,
beta_1..beta_p`).

**Lowering the ansatz to executable gates.** `QAOAAnsatz`'s circuit is built
from high-level, nested custom instructions (a top-level "QAOA" block, cost
and mixer sub-blocks), which the Aer simulator cannot execute directly (it
raises `unknown instruction: QAOA` if you try). This project lowers the
ansatz by repeatedly calling `.decompose()` until only primitive gates
remain (`src/quantum/qaoa_solver.py::_fully_decompose`), rather than running
it through Qiskit's preset transpiler pass manager targeting an
`AerSimulator` backend -- see `docs/experimental_methodology.md` for why
(a genuine transpiler bug was found at higher qubit counts, and basis-gate
translation for a specific hardware target is unnecessary for a local
simulation anyway).

## Training (the classical half of QAOA)

The `(gamma, beta)` parameters are trained by minimizing the **exact**
(noiseless) expectation value `<psi(gamma,beta)| H_C |psi(gamma,beta)>`,
computed with `qiskit_aer.primitives.EstimatorV2` (statevector-exact, no
shot noise), using `scipy.optimize.minimize` (default `COBYLA`) as the
classical outer loop. Initial parameters are drawn from
`Uniform(0, pi)` using `QAOAConfig.seed`, so training is reproducible given
the same seed.

**Why train on the exact expectation instead of a shot-sampled one?** This
project uses a simulator specifically to isolate the algorithm's behavior
from hardware/shot noise (a deliberate scope decision, consistent with the
"use a simulator for now" instruction) -- the final answer a real device
would give is still evaluated honestly via shot-based sampling, just not
during training.

## Measurement and decoding

Once trained, the final circuit is bound to the optimizer's best parameters,
measured, and sampled `shots` times via `qiskit_aer.primitives.SamplerV2`
(this step **does** include shot noise -- it's meant to represent what a
real run would return). Qiskit's classical bit-string output is
little-endian (the leftmost character is the *highest*-indexed qubit), so
every raw measurement string is reversed once, immediately
(`_bind_variable_order`), before it is used anywhere else in the project --
downstream code only ever sees "x0 x1 ... x_{n-1}" order and never has to
reason about bit order again. This mapping (and its reversal) was verified
against a hand-computed 3-qubit example (see the development notes referenced
from `src/quantum/qaoa_solver.py`'s docstring) before being relied on.

Each measured bitstring selects a set of directed edges. `src/quantum/decoder.py`
attempts to reconstruct a path from that edge set (walking from `origin`,
following the unique outgoing edge if there is one, rejecting the bitstring
explicitly if any node has more than one outgoing/incoming selected edge, if
the walk doesn't reach `destination`, or if selected edges are left over
after it does -- see `docs/qubo_derivation.md` for the disjoint-cycle case
this last check exists for). **A measurement is never assumed valid just
because it was the most probable one** -- QAOA at shallow depth on a
penalty-heavy landscape frequently returns invalid (infeasible) bitstrings
as its single most likely outcome; this project reports that plainly
instead of silently taking the top measurement at face value (see
`tests/test_decoder.py` and the "no valid path" branch of `src/main.py`
and the web dashboard).

## Reported metrics

- **Quantum cost**: the QUBO energy of the lowest-cost *valid* decoded path
  (not necessarily the most probable measurement).
- **Success probability**: the total measured probability mass across
  every distinct measured bitstring that decoded to *some* valid path (not
  just the best one) -- `src/quantum/decoder.py::success_probability`.
- **Approximation ratio / optimality gap**: `src/optimization/metrics.py`,
  computed against the brute-force-verified optimum when available, or the
  classical Dijkstra result otherwise, with explicit zero-cost handling
  (see that module's docstring).

None of these numbers are adjusted, floored, or "rounded up" to look
better -- when QAOA fails to find any valid path, the quantum cost and both
metrics are reported as unavailable (`None` / N/A), not substituted with a
placeholder.
