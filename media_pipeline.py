"""Compatibility wrapper for the numbered ML worker."""

import importlib
import sys

_impl = importlib.import_module("pipeline_09_worker.pipeline.media_pipeline")
if __name__ == "__main__":
    raise SystemExit(_impl.main())
sys.modules[__name__] = _impl
