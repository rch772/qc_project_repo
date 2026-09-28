"""Flight-network visualizations: the background network (nodes + edges +
origin/destination) and, on top of it, the classical vs quantum route
comparison. Built from actual pipeline results only -- never synthetic/
placeholder data.

Every plotting function returns PNG bytes (in addition to optionally
writing to disk), so the same code path serves the CLI demo
(src/main.py, which saves to results/benchmarks/) and the web dashboard
(src/webapp/app.py, which streams the bytes directly in an HTTP response)
without duplicating any drawing logic.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

# Matplotlib must use a writable, ephemeral cache and a headless backend on
# Vercel. The images themselves are still rendered to memory (BytesIO).
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.network import FlightNetwork


def _draw_background(ax, network: FlightNetwork, origin: int, destination: int) -> None:
    """All edges (faint, direction-deduplicated), all node markers with
    labels, and the origin/destination highlight markers. Shared by every
    plot in this module so the two views stay visually consistent."""
    drawn_pairs: set[tuple[int, int]] = set()
    for (u, v) in network.edges:
        pair = (min(u, v), max(u, v))
        if pair in drawn_pairs:
            continue
        drawn_pairs.add(pair)
        nu, nv = network.nodes[u], network.nodes[v]
        ax.plot([nu.x, nv.x], [nu.y, nv.y], color="#cccccc", linewidth=1.0, zorder=1)

    xs_all = [n.x for n in network.nodes.values()]
    ys_all = [n.y for n in network.nodes.values()]
    ax.scatter(xs_all, ys_all, s=140, color="#555555", zorder=3)
    for node in network.nodes.values():
        ax.annotate(node.name, (node.x, node.y), textcoords="offset points", xytext=(6, 6), fontsize=9)

    o, d = network.nodes[origin], network.nodes[destination]
    ax.scatter([o.x], [o.y], s=260, color="#2ca02c", zorder=4, label="Origin", edgecolors="black")
    ax.scatter([d.x], [d.y], s=260, color="#d62728", marker="s", zorder=4, label="Destination", edgecolors="black")


def _finish_and_emit(fig, ax, title: str, save_path: str | Path | None) -> bytes:
    ax.set_title(title)
    ax.set_xlabel("x (distance units)")
    ax.set_ylabel("y (distance units)")
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    data = buf.getvalue()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(data)

    plt.close(fig)
    return data


def plot_network_only(
    network: FlightNetwork,
    origin: int,
    destination: int,
    save_path: str | Path | None = None,
    title: str = "Flight Network",
) -> bytes:
    """The scenario as generated, before any optimization has run."""
    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_background(ax, network, origin, destination)
    return _finish_and_emit(fig, ax, title, save_path)


def plot_route_comparison(
    network: FlightNetwork,
    origin: int,
    destination: int,
    classical_path: list[int] | tuple[int, ...],
    quantum_path: list[int] | tuple[int, ...] | None,
    save_path: str | Path | None = None,
    title: str = "Flight Trajectory Optimization: Classical vs Quantum",
) -> bytes:
    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_background(ax, network, origin, destination)

    def _draw_route(path, color, label, linewidth, offset):
        if not path:
            return
        xs = [network.nodes[n].x + offset for n in path]
        ys = [network.nodes[n].y + offset for n in path]
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=0.85, zorder=2, label=label)

    _draw_route(classical_path, "#1f77b4", f"Classical route ({len(classical_path) - 1} legs)", 3.0, offset=0.0)
    if quantum_path:
        _draw_route(quantum_path, "#ff7f0e", f"Quantum (QAOA) route ({len(quantum_path) - 1} legs)", 2.2, offset=1.5)
    else:
        ax.plot([], [], color="#ff7f0e", label="Quantum (QAOA): no valid path decoded")

    return _finish_and_emit(fig, ax, title, save_path)
