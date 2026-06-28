"""Ensure the auto-censor extension's runtime deps are present.

NudeNet inference needs onnxruntime; Forge core ships opencv + numpy + Pillow but
NOT onnxruntime. Install it if missing. (Skipped when Forge runs with
--skip-prepare-environment; in that case install onnxruntime into the venv manually.)
"""
import importlib.util

import launch  # provided by the Forge/A1111 launcher

if importlib.util.find_spec("onnxruntime") is None:
    # Plain CPU build is the safe default; a CUDA build (onnxruntime-gpu) can be
    # installed manually for GPU acceleration if the CUDA/cuDNN stack matches.
    launch.run_pip("install onnxruntime", "onnxruntime (auto-censor)")
