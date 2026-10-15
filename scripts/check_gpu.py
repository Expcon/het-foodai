"""Check that installed torch + torchvision can access RTX GPU, with no changes to system."""
from __future__ import annotations
try:
    import torch
    import torchvision
except ImportError as exc:
    raise SystemExit(f"PyTorch/torchvision not installed in this environment: {exc}")
print(f"torch={torch.__version__}, torchvision={torchvision.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device={torch.cuda.get_device_name(0)}")
else:
    print("GPU not available yet: do not run costly real training until fixed.")
