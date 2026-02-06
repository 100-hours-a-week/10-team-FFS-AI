import time
from functools import wraps
from inspect import iscoroutinefunction

from prometheus_client import Histogram

PIPELINE_STAGE_DURATION = Histogram(
    name="outfit_pipeline_stage_duration_seconds",
    documentation="Duration of each outfit pipeline stage",
    labelnames=["stage"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

PIPELINE_TOTAL_DURATION = Histogram(
    name="outfit_pipeline_total_duration_seconds",
    documentation="Total duration of outfit recommendation pipeline",
    buckets=[1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0],
)


def measure_time(stage: str):
    def decorator(func):
        if iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    PIPELINE_STAGE_DURATION.labels(stage=stage).observe(
                        time.perf_counter() - start
                    )
            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    PIPELINE_STAGE_DURATION.labels(stage=stage).observe(
                        time.perf_counter() - start
                    )

            return sync_wrapper
    return decorator
