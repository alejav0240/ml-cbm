"""Generate a dependency-free, human-readable diagram of the DEMUCA ML flow."""

from __future__ import annotations

from html import escape
from pathlib import Path


NODES = [
    ("source", "Session media + reviewed annotations"),
    ("extract", "Normalize audio/video; extract frames + features"),
    ("sync", "Segment on shared clock; validate A/V drift"),
    ("pose", "Optional: MediaPipe pose landmarks"),
    ("join", "Align modalities to reviewed time windows"),
    ("gate", "Approval + audit + label eligibility gates"),
    ("split", "Patient-level train / validation / test split"),
    ("model", "Feature-based multilabel model + majority baseline comparison"),
    ("tracking", "MLflow: metrics, status, safe artifacts"),
]

EDGES = [(node[0], NODES[index + 1][0]) for index, node in enumerate(NODES[:-1])]


def _mermaid() -> str:
    lines = ["flowchart TD"]
    for node_id, label in NODES:
        lines.append(f'    {node_id}["{label}"]')
    lines.extend(f"    {source} --> {target}" for source, target in EDGES)
    lines.append("    classDef gate fill:#fff4cc,stroke:#a66b00,color:#222")
    lines.append("    class gate gate")
    return "\n".join(lines) + "\n"


def _svg() -> str:
    width = 920
    node_width = 700
    node_height = 58
    top = 26
    gap = 34
    height = top * 2 + len(NODES) * node_height + (len(NODES) - 1) * gap
    x = (width - node_width) // 2
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="title description">',
        '<title id="title">DEMUCA ML pipeline</title>',
        '<desc id="description">Media extraction, synchronization, pose, review gates, '
        'patient split, evaluation, and MLflow tracking.</desc>',
        '<style>text{font-family:system-ui,sans-serif}.node{fill:#f5f8fc;stroke:#35618f;'
        'stroke-width:2}.gate{fill:#fff4cc;stroke:#a66b00}.label{font-size:17px;'
        'fill:#15283c}.arrow{stroke:#596b7b;stroke-width:2;fill:none}</style>',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="460" y="22" text-anchor="middle" font-size="20" font-weight="700" '
        'fill="#15283c">DEMUCA ML pipeline</text>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="8" '
        'refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#596b7b"/>'
        '</marker></defs>',
    ]
    positions: dict[str, tuple[int, int]] = {}
    for index, (node_id, label) in enumerate(NODES):
        y = top + 18 + index * (node_height + gap)
        positions[node_id] = (x, y)
        klass = "node gate" if node_id == "gate" else "node"
        parts.append(
            f'<rect class="{klass}" x="{x}" y="{y}" width="{node_width}" '
            f'height="{node_height}" rx="12"/>'
        )
        parts.append(
            f'<text class="label" x="{width // 2}" y="{y + 36}" '
            f'text-anchor="middle">{escape(label)}</text>'
        )
    for source, target in EDGES:
        sx, sy = positions[source]
        tx, ty = positions[target]
        parts.append(
            f'<path class="arrow" marker-end="url(#arrow)" '
            f'd="M{sx + node_width // 2},{sy + node_height} '
            f'L{tx + node_width // 2},{ty - 6}"/>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_pipeline_graph(output_dir: Path) -> list[Path]:
    """Write SVG (directly viewable) and Mermaid source as run artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "pipeline_graph.svg"
    mermaid_path = output_dir / "pipeline_graph.mmd"
    svg_path.write_text(_svg(), encoding="utf-8")
    mermaid_path.write_text(_mermaid(), encoding="utf-8")
    return [svg_path, mermaid_path]
