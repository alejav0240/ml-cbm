"""Compatibility wrapper for the numbered ML worker."""
from pipeline_09_worker.pipeline.media_pipeline import *  # noqa: F401,F403
from pipeline_09_worker.pipeline.media_pipeline import main
if __name__ == "__main__":
    raise SystemExit(main())
