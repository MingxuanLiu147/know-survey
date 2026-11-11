"""Device selection shared by multimodal training, editing and evaluation."""

import logging


def resolve_device(cfg) -> str:
    """Require the requested CUDA device unless CPU fallback is explicit."""
    import torch

    device = str(cfg.get("device", "cuda"))
    require_cuda = bool(cfg.get("require_cuda", device.startswith("cuda")))
    if device.startswith("cuda") and not torch.cuda.is_available():
        message = (
            "CUDA is unavailable. Check your PyTorch installation and visible GPUs, "
            "or select device=cpu require_cuda=false for CPU execution."
        )
        if require_cuda:
            raise RuntimeError(message)
        logging.getLogger(__name__).warning(message)
        return "cpu"
    return device
