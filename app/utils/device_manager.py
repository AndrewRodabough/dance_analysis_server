"""
Device manager for PyTorch/MMPose across NVIDIA, AMD, and CPU.
"""
import os
import logging

logger = logging.getLogger(__name__)


def get_device():
    """
    Get the appropriate device for PyTorch/MMPose inference.
    
    Returns:
        str: Device string ('cuda:0', 'cpu', etc.)
    """
    # Check environment variable set by docker-compose profile
    device_type = os.environ.get("DEVICE_TYPE", "").lower()
    
    if device_type == "cpu":
        logger.info("Using CPU (from DEVICE_TYPE env var)")
        return "cpu"
    
    # For both 'cuda' and 'rocm', try to use GPU
    if device_type in ("cuda", "rocm"):
        try:
            import torch
            if torch.cuda.is_available():
                device_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                device = f"cuda:{device_id}"
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
                logger.info(f"Using GPU: {gpu_name} (device: {device}, type: {device_type})")
                return device
            else:
                logger.warning(f"DEVICE_TYPE={device_type} but no GPU available, falling back to CPU")
                return "cpu"
        except ImportError:
            logger.warning("PyTorch not available, using CPU")
            return "cpu"
    
    # Auto-detect if DEVICE_TYPE not set
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda:0"
            logger.info(f"Auto-detected GPU: {torch.cuda.get_device_name(0)}")
            return device
    except ImportError:
        pass
    
    logger.info("Using CPU (default)")
    return "cpu"


def log_device_info():
    """Log detailed device information."""
    device = get_device()
    logger.info(f"Selected device: {device}")
    
    try:
        import torch
        if device.startswith("cuda"):
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA version: {torch.version.cuda}")
                logger.info(f"GPU count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        logger.warning("PyTorch not installed")
