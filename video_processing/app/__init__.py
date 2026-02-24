"""Video processing app package."""

# Make tasks import optional for offline analysis workflow
# tasks.py requires rq/redis which aren't needed for direct keypoint analysis
try:
    from . import tasks
    __all__ = ["tasks"]
except ImportError:
    # Running in minimal environment without job queue dependencies
    __all__ = []
