"""Compatibility wrapper for the numbered ML worker."""
from pipeline_09_worker.entrypoints.orchestrator_service import *  # noqa: F401,F403
from pipeline_09_worker.entrypoints.orchestrator_service import main
if __name__ == "__main__":
    raise SystemExit(main())
