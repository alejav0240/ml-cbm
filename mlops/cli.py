"""CLI del ciclo local de registro y promoción MLOps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .deriva import DEFAULT_FEATURE_COLUMNS, analizar_deriva, escribir_reporte_deriva
from .mlflow_registry import registrar_version_mlflow
from .registro import (
    ErrorRegistroModelo,
    desplegar_candidato,
    promover_candidato,
    registrar_candidato,
    revertir_despliegue,
)


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
    deploy = subparsers.add_parser("deploy", help="activate a promoted model atomically")
    deploy.add_argument("--registry", type=Path, required=True)
    deploy.add_argument("--version", required=True)
    deploy.add_argument("--out-dir", type=Path, required=True)
    rollback = subparsers.add_parser("rollback", help="restore the previous deployment")
    rollback.add_argument("--registry", type=Path, required=True)
    rollback.add_argument("--out-dir", type=Path, required=True)
    rollback.add_argument("--version")
    mlflow_register = subparsers.add_parser("register-mlflow", help="mirror a promoted version into MLflow Model Registry")
    mlflow_register.add_argument("--registry", type=Path, required=True)
    mlflow_register.add_argument("--version", required=True)
    mlflow_register.add_argument("--registered-model", required=True)
    mlflow_register.add_argument("--tracking-uri")
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
        elif args.command == "deploy":
            result = desplegar_candidato(args.registry, args.version, args.out_dir)
        elif args.command == "rollback":
            result = revertir_despliegue(args.registry, args.out_dir, args.version)
        elif args.command == "register-mlflow":
            result = registrar_version_mlflow(
                args.registry,
                args.version,
                args.registered_model,
                tracking_uri=args.tracking_uri,
            )
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
