from __future__ import annotations

import argparse
from pathlib import Path

from src.data.scenario_generator import generate_scenario
from src.utils.config import ScenarioConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

TIERS: dict[str, list[int]] = {
    "small": [5, 6, 7, 8],
    "medium": [10, 12, 15],
    "large": [18, 20, 25],
}


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "scenarios"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic flight-network benchmark scenarios.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--num-nodes", type=int, default=None, help="Generate a single scenario with this many nodes")
    parser.add_argument(
        "--tier",
        choices=["small", "medium", "large", "all"],
        default=None,
        help="Generate every scenario size in the named benchmark tier",
    )
    parser.add_argument("--area-size", type=float, default=500.0)
    parser.add_argument("--k-nearest", type=int, default=3)
    parser.add_argument("--extra-edge-fraction", type=float, default=0.15)
    parser.add_argument("--restricted-fraction", type=float, default=0.08)
    parser.add_argument("--weather-fraction", type=float, default=0.15)
    parser.add_argument("--base-fuel-rate", type=float, default=3.0)
    parser.add_argument("--base-speed", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> list[Path]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.num_nodes is None and args.tier is None:
        sizes = TIERS["small"]
    elif args.tier == "all":
        sizes = [n for tier_sizes in TIERS.values() for n in tier_sizes]
    elif args.tier is not None:
        sizes = TIERS[args.tier]
    else:
        sizes = [args.num_nodes]

    output_dir = args.output_dir or _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for num_nodes in sizes:
        config = ScenarioConfig(
            num_nodes=num_nodes,
            seed=args.seed,
            area_size=args.area_size,
            k_nearest=args.k_nearest,
            extra_edge_fraction=args.extra_edge_fraction,
            restricted_fraction=args.restricted_fraction,
            weather_fraction=args.weather_fraction,
            base_fuel_rate=args.base_fuel_rate,
            base_speed=args.base_speed,
        )
        network = generate_scenario(config)
        out_path = output_dir / f"scenario_n{num_nodes}_seed{args.seed}.json"
        network.save_json(out_path)
        logger.info(f"Wrote {out_path} ({network.num_nodes} nodes, {network.num_edges} edges)")
        written.append(out_path)

    return written


if __name__ == "__main__":
    main()
