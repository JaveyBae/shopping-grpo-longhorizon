"""Opt-in startup hooks for Shopping GRPO Ray workers.

Ray applies ``runtime_env.env_vars`` before launching a worker process.  Loading
the veRL compatibility shim from Python's standard ``sitecustomize`` hook keeps
CUDA untouched until after Ray has assigned that worker's visible GPU.
"""

import os


if os.environ.get("SHOPPING_GRPO_INSTALL_VERL_COMPAT") == "1":
    from shopping_grpo.training.grpo.compat import install_torch_padding_fallback

    install_torch_padding_fallback()
