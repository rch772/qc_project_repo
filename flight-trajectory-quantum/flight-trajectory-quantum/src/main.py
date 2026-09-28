from __future__ import annotations

import argparse
import csv
import datetime as dt
import platform
from pathlib import Path

import qiskit
import qiskit_aer

from src.classical.brute_force import brute_force_solution
from src.classical.dijkstra import classical_shortest_path
from src.data.scenario_generator import generate_scenario
from src.optimization.metrics import approximation_ratio, optimality_gap
from src.optimization.qubo import build_qubo
from src.quantum.decoder import best_valid_solution, decode_all_counts, success_probability
from src.quantum.qaoa_solver import QuantumSimulatorError, solve_qaoa
from src.utils.config import ObjectiveWeights, QAOAConfig, ScenarioConfig
from src.utils.logging_config import get_logger
from src.visualization.plot_network import plot_route_comparison

logger = get_logger(__name__)

# Beyond this many nodes, exhaustive brute force is skipped: the number of
# simple paths grows combinatorially and stops being tractable well before
# 15-20 nodes on a moderately connected graph.
BRUTE_FORCE_NODE_LIMIT = 10


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the full classical-vs-quantum flight trajectory optimization pipeline.")
    p.add_argument(
        "--num-nodes", type=int, default=5,
        help="Default is deliberately small: statevector QAOA training time roughly "
        "doubles with every extra qubit (see docs/experimental_methodology.md), and "
        "qubits = directed edges here. 5-6 nodes trains in well under a minute; "
        "8+ nodes (with default connectivity) can take several minutes.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--origin", type=int, default=None)
    p.add_argument("--destination", type=int, default=None)
    p.add_argument("--alpha", type=float, default=1.0, help="Weight on distance")
    p.add_argument("--beta", type=float, default=1.0, help="Weight on fuel cost")
    p.add_argument("--gamma", type=float, default=1.0, help="Weight on time cost")
    p.add_argument("--delta", type=float, default=1.0, help="Weight on weather/airspace penalty")
    p.add_argument("--reps", type=int, default=3, help="QAOA circuit depth p")
    p.add_argument("--optimizer", type=str, default="COBYLA")
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--penalty-lambda", type=float, default=None, help="Override the auto-derived QUBO penalty coefficient")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-save", action="store_true", help="Don't append a row to results/benchmark_results.csv")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    scenario_config = ScenarioConfig(
        num_nodes=args.num_nodes, seed=args.seed, origin=args.origin, destination=args.destination
    )
    network = generate_scenario(scenario_config)
    origin = network.metadata["origin"]
    destination = network.metadata["destination"]
    weights = ObjectiveWeights(alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta)

    print("=" * 60)
    print("QUANTUM FLIGHT TRAJECTORY OPTIMIZATION")
    print("=" * 60)
    print(f"Scenario: {network.num_nodes} nodes, {network.num_edges} directed edges, seed={args.seed}")
    print(f"Origin: {network.nodes[origin].name} (id={origin})")
    print(f"Destination: {network.nodes[destination].name} (id={destination})")
    print(
        f"Objective weights: alpha={weights.alpha} beta={weights.beta} gamma={weights.gamma} "
        f"delta={weights.delta} (normalize={weights.normalize_components})"
    )
    print()

    bf_result = None
    if network.num_nodes <= BRUTE_FORCE_NODE_LIMIT:
        bf_result, examined = brute_force_solution(network, origin, destination, weights)
        print(f"Brute-force ground truth ({examined} simple paths examined):")
        print(f"  Route: {' -> '.join(network.nodes[n].name for n in bf_result.path)}")
        print(f"  Cost:  {bf_result.total_cost:.4f}")
        print()
    else:
        print(
            f"Brute-force ground truth SKIPPED (network has {network.num_nodes} > "
            f"{BRUTE_FORCE_NODE_LIMIT} nodes; exhaustive path enumeration is intractable "
            f"at this size). Classical Dijkstra result is used as the optimality reference instead."
        )
        print()

    classical_result = classical_shortest_path(network, origin, destination, weights)
    print("Classical (Dijkstra) solution:")
    print(f"  Route: {' -> '.join(network.nodes[n].name for n in classical_result.path)}")
    print(f"  Cost:  {classical_result.total_cost:.4f}")
    print()

    if bf_result is not None:
        matches = abs(classical_result.total_cost - bf_result.total_cost) < 1e-6
        print(f"Cross-validation (brute force == classical): {'PASS' if matches else 'FAIL'}")
        print()
        optimal_cost = bf_result.total_cost
    else:
        optimal_cost = classical_result.total_cost

    qubo = build_qubo(network, origin, destination, weights, penalty_lambda=args.penalty_lambda)
    print(f"QUBO formulation: {qubo.num_variables} binary variables (qubits), penalty_lambda={qubo.penalty_lambda:.4f}")
    print()

    qaoa_config = QAOAConfig(
        reps=args.reps, optimizer=args.optimizer, max_iter=args.max_iter,
        shots=args.shots, seed=args.seed, penalty_lambda=args.penalty_lambda,
    )
    print(
        f"Running QAOA: reps={qaoa_config.reps}, optimizer={qaoa_config.optimizer}, "
        f"max_iter={qaoa_config.max_iter}, shots={qaoa_config.shots}, "
        f"max_qubits={qaoa_config.max_qubits} ..."
    )
    try:
        qaoa_result = solve_qaoa(qubo, qaoa_config)
    except QuantumSimulatorError as exc:
        print(f"  SKIPPED: {exc}")
        print()
        print("-" * 60)
        print("COMPARISON")
        print("-" * 60)
        print(f"Qubits required:         {qubo.num_variables} (exceeds max_qubits={qaoa_config.max_qubits})")
        print(f"Classical optimal cost:  {optimal_cost:.4f}")
        print("Quantum cost:            N/A (simulation infeasible on this machine at this qubit count)")
        print("=" * 60)
        if not args.no_plot:
            plot_path = Path("results/benchmarks") / f"routes_n{network.num_nodes}_seed{args.seed}.png"
            plot_route_comparison(network, origin, destination, classical_result.path, None, plot_path)
            print(f"\nSaved route comparison plot to {plot_path}")
        return 0

    print(
        f"  Optimizer evaluations: {qaoa_result.num_optimizer_iterations}, "
        f"converged={qaoa_result.optimizer_converged}"
    )
    print(
        f"  Circuit build: {qaoa_result.circuit_build_seconds:.2f}s | "
        f"Training: {qaoa_result.training_seconds:.2f}s | "
        f"Sampling: {qaoa_result.sampling_seconds:.2f}s"
    )
    print()

    decoded = decode_all_counts(qubo, qaoa_result.counts, qaoa_result.shots)
    best = best_valid_solution(decoded)
    succ_p = success_probability(decoded)

    print("Quantum (QAOA) solution:")
    if best is not None:
        print(f"  Route: {' -> '.join(network.nodes[n].name for n in best.path)}")
        print(f"  Cost:  {best.qubo_energy:.4f}  (measured probability of this exact bitstring: {best.probability:.4f})")
    else:
        print("  No measured bitstring decoded to a valid origin -> destination path.")
        if decoded:
            top = decoded[0]
            print(f"  Most probable measurement (p={top.probability:.4f}) was invalid: {top.validation_error}")
    print(f"  Success probability (any valid path, summed): {succ_p:.4%}")
    print()

    print("-" * 60)
    print("COMPARISON")
    print("-" * 60)
    print(f"Qubits used:            {qubo.num_variables}")
    print(f"Shots:                  {qaoa_config.shots}")
    print(f"Classical optimal cost: {optimal_cost:.4f}")
    if best is not None:
        gap = optimality_gap(best.qubo_energy, optimal_cost)
        ratio = approximation_ratio(best.qubo_energy, optimal_cost)
        print(f"Quantum cost:            {best.qubo_energy:.4f}")
        print(f"Optimality gap:          {gap:.4%}")
        print(f"Approximation ratio:     {ratio:.4f}")
    else:
        gap = ratio = None
        print("Quantum cost:            N/A (no valid path found)")
        print("Optimality gap:          N/A")
        print("Approximation ratio:     N/A")
    print("=" * 60)
    print("NOTE: this is a small noiseless-simulator benchmark, not a demonstration")
    print("of quantum advantage. See docs/experimental_methodology.md for limitations.")
    print("=" * 60)

    if not args.no_plot:
        plot_path = Path("results/benchmarks") / f"routes_n{network.num_nodes}_seed{args.seed}.png"
        plot_route_comparison(
            network, origin, destination, classical_result.path,
            best.path if best else None, plot_path,
        )
        print(f"\nSaved route comparison plot to {plot_path}")

    if not args.no_save:
        row_path = _append_result_row(
            args, network, weights, qubo, qaoa_result, classical_result,
            bf_result, best, succ_p, optimal_cost, gap, ratio,
        )
        print(f"Appended result row to {row_path}")

    return 0


def _append_result_row(args, network, weights, qubo, qaoa_result, classical_result,
                        bf_result, best, succ_p, optimal_cost, gap, ratio) -> Path:
    results_path = Path("results/benchmark_results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not results_path.exists()

    row = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "num_nodes": network.num_nodes,
        "num_edges": network.num_edges,
        "num_qubits": qubo.num_variables,
        "seed": args.seed,
        "alpha": weights.alpha,
        "beta": weights.beta,
        "gamma": weights.gamma,
        "delta": weights.delta,
        "reps": qaoa_result.reps,
        "optimizer": qaoa_result.optimizer,
        "shots": qaoa_result.shots,
        "penalty_lambda": qubo.penalty_lambda,
        "brute_force_ran": bf_result is not None,
        "classical_cost": classical_result.total_cost,
        "optimal_cost_reference": optimal_cost,
        "quantum_cost": best.qubo_energy if best else "",
        "quantum_success_probability": succ_p,
        "optimality_gap": gap if gap is not None else "",
        "approximation_ratio": ratio if ratio is not None else "",
        "training_seconds": qaoa_result.training_seconds,
        "sampling_seconds": qaoa_result.sampling_seconds,
        "optimizer_converged": qaoa_result.optimizer_converged,
        "qiskit_version": qiskit.__version__,
        "qiskit_aer_version": qiskit_aer.__version__,
        "python_version": platform.python_version(),
    }

    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    return results_path


if __name__ == "__main__":
    raise SystemExit(main())
