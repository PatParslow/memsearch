"""Let chromadb's onnxruntime embedding model use the GPU if it can.

onnxruntime looks for cudnn64_9.dll on PATH to load CUDAExecutionProvider;
without it, it silently falls back to CPU (much slower) and prints a
scary-looking provider-load-failure block on every run. If `pip install
nvidia-cudnn-cu13` (or cu12) has been run, this puts its bundled DLLs on
PATH for this process. No-ops quietly if that package isn't installed --
same CPU fallback as before, just without doing anything smarter.
"""

import os


def enable_gpu() -> None:
    try:
        import nvidia.cudnn  # type: ignore[import-not-found]
    except ImportError:
        return

    # nvidia-cudnn-cuXX ships as a PEP 420 namespace package (no __init__.py,
    # so __file__ is None) -- use __path__ instead.
    try:
        cudnn_bin = os.path.join(list(nvidia.cudnn.__path__)[0], "bin")
    except (IndexError, AttributeError):
        return

    if os.path.isdir(cudnn_bin):
        os.environ["PATH"] = cudnn_bin + os.pathsep + os.environ.get("PATH", "")
