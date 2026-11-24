"""Compatibility wrapper for the numbered MLOps pipeline."""
import importlib
import sys
_impl = importlib.import_module("pipeline_08_mlops.registro")
if __name__ == "__main__":
    raise SystemExit(_impl.main())
sys.modules[__name__] = _impl
