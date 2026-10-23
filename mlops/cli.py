"""CLI del ciclo local de registro y promoción MLOps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    args = parser.parse_args()
    try:
        if args.command == "register":
            result = registrar_candidato(args.artifacts_dir, args.registry, version=args.version)
        else:
            result = promover_candidato(args.registry, args.version, min_micro_f1=args.min_micro_f1)
    except ErrorRegistroModelo as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
