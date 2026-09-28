from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import os

from flask import Flask, Response, jsonify, render_template, request

from src.classical.brute_force import brute_force_solution
from src.classical.dijkstra import classical_shortest_path
from src.data.scenario_generator import generate_scenario
from src.main import BRUTE_FORCE_NODE_LIMIT
from src.models.network import DisconnectedNetworkError, FlightNetwork, InvalidNodeError, InvalidPathError
from src.optimization.metrics import approximation_ratio, optimality_gap
from src.optimization.qubo import build_qubo
from src.quantum.decoder import best_valid_solution, decode_all_counts, success_probability
from src.quantum.qaoa_solver import QuantumSimulatorError, solve_qaoa
from src.utils.config import ObjectiveWeights, QAOAConfig, ScenarioConfig
from src.utils.logging_config import get_logger
from src.visualization.plot_network import plot_network_only, plot_route_comparison

logger = get_logger(__name__)


@dataclass
class _AppState:
    network: FlightNetwork | None = None
    origin: int | None = None
    destination: int | None = None
    last_route_plot: bytes | None = None


STATE = _AppState()


def _network_from_payload(payload) -> FlightNetwork | None:
    """Load a request-scoped network so serverless requests need no sticky worker."""
    if payload is None:
        return None
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("scenario must be a serialized flight network object")
    return FlightNetwork.from_dict(payload)


def _route_names(network: FlightNetwork, path) -> list[str]:
    return [network.nodes[n].name for n in path]


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/scenario")
    def api_scenario():
        body = request.get_json(force=True) or {}
        try:
            config = ScenarioConfig(
                num_nodes=int(body.get("num_nodes", 5)),
                seed=int(body.get("seed", 42)),
                k_nearest=int(body.get("k_nearest", 2)),
                extra_edge_fraction=float(body.get("extra_edge_fraction", 0.1)),
                restricted_fraction=float(body.get("restricted_fraction", 0.08)),
                weather_fraction=float(body.get("weather_fraction", 0.15)),
                area_size=float(body.get("area_size", 500.0)),
                base_fuel_rate=float(body.get("base_fuel_rate", 3.0)),
                base_speed=float(body.get("base_speed", 8.0)),
                origin=body.get("origin"),
                destination=body.get("destination"),
            )
        except Exception as exc:  # pydantic ValidationError or bad int()/float() cast
            return jsonify({"error": f"Invalid scenario parameters: {exc}"}), 400

        network = generate_scenario(config)
        origin = network.metadata["origin"]
        destination = network.metadata["destination"]

        STATE.network = network
        STATE.origin = origin
        STATE.destination = destination
        STATE.last_route_plot = None

        estimated_qubits = network.num_edges  # one binary variable per directed edge
        return jsonify({
            "num_nodes": network.num_nodes,
            "num_edges": network.num_edges,
            "estimated_qubits": estimated_qubits,
            "origin": origin,
            "destination": destination,
            "origin_name": network.nodes[origin].name,
            "destination_name": network.nodes[destination].name,
            "node_names": {str(n.id): n.name for n in network.nodes.values()},
            "brute_force_feasible": network.num_nodes <= BRUTE_FORCE_NODE_LIMIT,
            "scenario": network.to_dict(),
        })

    @app.get("/api/plot/network")
    def api_plot_network():
        try:
            network = _network_from_payload(request.args.get("scenario")) or STATE.network
        except (TypeError, ValueError, KeyError) as exc:
            return jsonify({"error": f"Invalid scenario payload: {exc}"}), 400
        if network is None:
            return jsonify({"error": "No scenario generated yet."}), 400
        origin = int(request.args.get("origin", network.metadata.get("origin", STATE.origin or 0)))
        destination = int(request.args.get("destination", network.metadata.get("destination", STATE.destination or network.num_nodes - 1)))
        try:
            png = plot_network_only(network, origin, destination)
        except (KeyError, ValueError) as exc:
            return jsonify({"error": f"Invalid plot endpoints: {exc}"}), 400
        return Response(png, mimetype="image/png")

    @app.get("/api/plot/routes")
    def api_plot_routes():
        scenario_payload = request.args.get("scenario")
        if scenario_payload is not None:
            try:
                network = _network_from_payload(scenario_payload)
                classical_path = json.loads(request.args.get("classical_path", "null"))
                quantum_path = json.loads(request.args.get("quantum_path", "null"))
                if network is None or not isinstance(classical_path, list):
                    raise ValueError("scenario and classical_path are required")
                origin = int(network.metadata["origin"])
                destination = int(network.metadata["destination"])
                network.validate_path(classical_path)
                if quantum_path:
                    network.validate_path(quantum_path)
                png = plot_route_comparison(network, origin, destination, classical_path, quantum_path)
            except (TypeError, ValueError, KeyError, json.JSONDecodeError, InvalidPathError) as exc:
                return jsonify({"error": f"Invalid route plot payload: {exc}"}), 400
            return Response(png, mimetype="image/png")
        if STATE.last_route_plot is None:
            return jsonify({"error": "No results yet -- run the optimization first."}), 400
        return Response(STATE.last_route_plot, mimetype="image/png")

    @app.post("/api/run")
    def api_run():
        body = request.get_json(force=True) or {}
        try:
            network = _network_from_payload(body.get("scenario")) if "scenario" in body else STATE.network
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return jsonify({"error": f"Invalid scenario payload: {exc}"}), 400
        if network is None:
            return jsonify({"error": "Generate a scenario first."}), 400
        try:
            origin = int(network.metadata["origin"])
            destination = int(network.metadata["destination"])
        except (KeyError, TypeError, ValueError):
            origin, destination = STATE.origin, STATE.destination
        if origin is None or destination is None:
            return jsonify({"error": "Scenario is missing origin/destination metadata."}), 400
        # Keep the old local flow available; Vercel requests carry their own network.
        STATE.network, STATE.origin, STATE.destination = network, origin, destination
        w = body.get("weights", {})
        try:
            weights = ObjectiveWeights(
                alpha=float(w.get("alpha", 1.0)),
                beta=float(w.get("beta", 1.0)),
                gamma=float(w.get("gamma", 1.0)),
                delta=float(w.get("delta", 1.0)),
                normalize_components=bool(w.get("normalize_components", True)),
            )
        except Exception as exc:
            return jsonify({"error": f"Invalid objective weights: {exc}"}), 400

        q = body.get("qaoa", {})
        try:
            qaoa_config = QAOAConfig(
                reps=int(q.get("reps", 3)),
                optimizer=str(q.get("optimizer", "COBYLA")),
                max_iter=int(q.get("max_iter", 150)),
                shots=int(q.get("shots", 4096)),
                seed=int(q.get("seed", 42)),
                max_qubits=int(q.get("max_qubits", 24)),
                penalty_lambda=q.get("penalty_lambda"),
            )
        except Exception as exc:
            return jsonify({"error": f"Invalid QAOA settings: {exc}"}), 400

        result: dict = {}

        try:
            bf_result = None
            if network.num_nodes <= BRUTE_FORCE_NODE_LIMIT:
                bf_result, examined = brute_force_solution(network, origin, destination, weights)
                result["brute_force"] = {
                    "ran": True,
                    "route": _route_names(network, bf_result.path),
                    "cost": bf_result.total_cost,
                    "paths_examined": examined,
                }
            else:
                result["brute_force"] = {
                    "ran": False,
                    "reason": f"Skipped: {network.num_nodes} nodes exceeds the "
                              f"brute-force limit of {BRUTE_FORCE_NODE_LIMIT}.",
                }

            classical_result = classical_shortest_path(network, origin, destination, weights)
            result["classical"] = {
                "route": _route_names(network, classical_result.path),
                "path": list(classical_result.path),
                "cost": classical_result.total_cost,
            }

            if bf_result is not None:
                result["cross_validation"] = (
                    "PASS" if abs(classical_result.total_cost - bf_result.total_cost) < 1e-6 else "FAIL"
                )
                optimal_cost = bf_result.total_cost
            else:
                result["cross_validation"] = None
                optimal_cost = classical_result.total_cost
        except (InvalidNodeError, DisconnectedNetworkError) as exc:
            return jsonify({"error": str(exc)}), 400

        qubo = build_qubo(network, origin, destination, weights, penalty_lambda=qaoa_config.penalty_lambda)
        result["qubo"] = {"num_qubits": qubo.num_variables, "penalty_lambda": qubo.penalty_lambda}

        best = None
        serverless = os.environ.get("VERCEL") == "1"
        serverless_limits_exceeded = serverless and (
            qubo.num_variables > 16
            or qaoa_config.reps > 3
            or qaoa_config.max_iter > 150
            or qaoa_config.shots > 4096
        )
        if serverless_limits_exceeded:
            result["quantum"] = {
                "status": "serverless_limit",
                "error_message": (
                    "Vercel demo limit: QAOA supports at most 16 qubits, p=3, "
                    "150 optimizer iterations, and 4,096 shots per request. "
                    "Run larger experiments locally."
                ),
            }
        else:
            try:
                qaoa_result = solve_qaoa(qubo, qaoa_config)
            except QuantumSimulatorError as exc:
                result["quantum"] = {"status": "infeasible", "error_message": str(exc)}
            else:
                decoded = decode_all_counts(qubo, qaoa_result.counts, qaoa_result.shots)
                best = best_valid_solution(decoded)
                succ_p = success_probability(decoded)
                quantum_info = {
                    "num_optimizer_iterations": qaoa_result.num_optimizer_iterations,
                    "converged": qaoa_result.optimizer_converged,
                    "success_probability": succ_p,
                    "timing": {
                        "circuit_build_seconds": qaoa_result.circuit_build_seconds,
                        "training_seconds": qaoa_result.training_seconds,
                        "sampling_seconds": qaoa_result.sampling_seconds,
                    },
                }
                if best is not None:
                    quantum_info.update({
                        "status": "ok",
                        "route": _route_names(network, best.path),
                        "path": list(best.path),
                        "cost": best.qubo_energy,
                        "probability": best.probability,
                    })
                else:
                    top = decoded[0] if decoded else None
                    quantum_info.update({
                        "status": "no_valid_path",
                        "route": None,
                        "path": None,
                        "cost": None,
                        "top_measurement_probability": top.probability if top else None,
                        "top_measurement_error": top.validation_error if top else None,
                    })
                result["quantum"] = quantum_info

        comparison = {"qubits": qubo.num_variables, "shots": qaoa_config.shots, "optimal_cost": optimal_cost}
        if best is not None:
            comparison["quantum_cost"] = best.qubo_energy
            comparison["optimality_gap"] = optimality_gap(best.qubo_energy, optimal_cost)
            comparison["approximation_ratio"] = approximation_ratio(best.qubo_energy, optimal_cost)
        else:
            comparison["quantum_cost"] = None
            comparison["optimality_gap"] = None
            comparison["approximation_ratio"] = None
        result["comparison"] = comparison

        STATE.last_route_plot = plot_route_comparison(
            network, origin, destination, classical_result.path, best.path if best else None
        )

        return jsonify(result)

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the flight trajectory optimization web dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app()
    print(f"Dashboard running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
