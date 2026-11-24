"""Compatibility wrapper for the numbered pipeline."""

import importlib
import sys

_impl = importlib.import_module("pipeline_03_multimedia.pipeline_graph")
if __name__ == "__main__":
    raise SystemExit(_impl.main())
sys.modules[__name__] = _impl
