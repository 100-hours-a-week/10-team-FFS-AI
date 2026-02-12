import inspect
import logging
import time
from functools import wraps
from inspect import iscoroutinefunction

from prometheus_client import Counter, Histogram

logger = logging.getLogger("performance")

OUTFIT_PIPELINE_ERRORS = Counter(
    name="outfit_pipeline_errors_total",
    documentation="Total number of errors in outfit pipeline",
    labelnames=["stage", "error_type"],
)

OUTFIT_PIPELINE_STAGE_DURATION = Histogram(
    name="outfit_pipeline_stage_duration_seconds",
    documentation="Duration of each outfit pipeline stage",
    labelnames=["stage", "status"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)


OUTFIT_PIPELINE_TOTAL_DURATION = Histogram(
    name="outfit_pipeline_total_duration_seconds",
    documentation="Total duration of outfit recommendation pipeline",
    labelnames=["status"],
    buckets=[1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0],
)

BATCH_STATUS_REQUESTS = Counter(
    name="batch_status_requests_total",
    documentation="Total batch status check requests (Polling)",
    labelnames=["batch_id"],
)

REDIS_QUERIES = Counter(
    name="redis_queries_total",
    documentation="Total Redis queries",
    labelnames=["operation"],
)

CLOSET_STAGE_DURATION = Histogram(
    name="closet_pipeline_stage_duration_seconds",
    documentation="Duration of closet analysis stages",
    labelnames=["stage", "status"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)


def measure_time(stage: str, metric: Histogram = OUTFIT_PIPELINE_STAGE_DURATION):
    def decorator(func):
        def _get_trace_id(*args, **kwargs):
            if "trace_id" in kwargs:
                return kwargs["trace_id"]

            try:
                sig = inspect.signature(func)
                bound_args = sig.bind_partial(*args, **kwargs)
                return bound_args.arguments.get("trace_id", "no_trace_id")
            except Exception:
                return "no_trace_id"

        if iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                trace_id = _get_trace_id(*args, **kwargs)
                start = time.perf_counter()
                status = "success"
                error_type = "none"

                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    status = "error"
                    error_type = type(e).__name__
                    OUTFIT_PIPELINE_ERRORS.labels(stage=stage, error_type=error_type).inc()
                    raise e
                finally:
                    duration = time.perf_counter() - start

                    if "stage" in metric._labelnames:
                        metric.labels(stage=stage, status=status).observe(duration)
                    else:
                        metric.labels(status=status).observe(duration)

                    logging.info(
                        f"[PERF] stage={stage} duration={duration:.4f}s "
                        f"trace_id={trace_id} success={status == 'success'}"
                    )

            return async_wrapper

        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                trace_id = _get_trace_id(*args, **kwargs)
                start = time.perf_counter()
                status = "success"
                error_type = "none"

                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    status = "error"
                    error_type = type(e).__name__
                    OUTFIT_PIPELINE_ERRORS.labels(stage=stage, error_type=error_type).inc()
                    raise e
                finally:
                    duration = time.perf_counter() - start

                    if "stage" in metric._labelnames:
                        metric.labels(stage=stage, status=status).observe(duration)
                    else:
                        metric.labels(status=status).observe(duration)

                    logging.info(
                        f"[PERF] stage={stage} duration={duration:.4f}s "
                        f"trace_id={trace_id} success={status == 'success'}"
                    )

            return sync_wrapper

    return decorator
