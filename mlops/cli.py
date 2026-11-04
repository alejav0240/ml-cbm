"""CLI del ciclo local de registro y promoción MLOps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .deriva import DEFAULT_FEATURE_COLUMNS, analizar_deriva, escribir_reporte_deriva
from .registro import ErrorRegistroModelo, promover_candidato, registrar_candidato


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register")
    register.add_argument("--artifacts-dir", type=Path, required=True)
    register.add_argument("--registry", type=Path, required=True)
    register.add_argument("--version")
    promote = subparsers.add_parser("promote")
    promote.add_argument("--registry", type=Path, required=True)
    promote.add_argument("--version", required=True)
    promote.add_argument("--min-micro-f1", type=float, default=0.0)
    drift = subparsers.add_parser("drift", help="compare reference and candidate feature CSVs")
    drift.add_argument("--reference", type=Path, required=True)
    drift.add_argument("--candidate", type=Path, required=True)
    drift.add_argument("--out", type=Path, required=True)
    drift.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURE_COLUMNS))
    drift.add_argument("--threshold", type=float, default=0.2)
    drift.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "register":
            result = registrar_candidato(args.artifacts_dir, args.registry, version=args.version)
        elif args.command == "promote":
            result = promover_candidato(args.registry, args.version, min_micro_f1=args.min_micro_f1)
        else:
            result = analizar_deriva(
                args.reference,
                args.candidate,
                feature_columns=tuple(args.features),
                threshold=args.threshold,
            )
            escribir_reporte_deriva(args.out, result)
    except (ErrorRegistroModelo, OSError, ValueError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.command == "drift" and args.fail_on_drift and result["drift_detected"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
