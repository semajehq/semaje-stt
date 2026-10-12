"""Prometheus metrics for the STT service (GET /metrics)."""
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest

CONTENT_TYPE = CONTENT_TYPE_LATEST
REGISTRY = CollectorRegistry()

HTTP_REQUESTS = Counter(
    "semaje_stt_http_requests_total", "HTTP requests by route and status.",
    ["path", "method", "status"], registry=REGISTRY,
)
HTTP_LATENCY = Histogram(
    "semaje_stt_http_request_seconds", "HTTP request latency.", ["path"], registry=REGISTRY,
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 900),
)
AUDIO_SECONDS = Counter(
    "semaje_stt_audio_seconds_total", "Seconds of audio transcribed.", ["engine", "model"], registry=REGISTRY,
)
REALTIME_FACTOR = Histogram(
    "semaje_stt_realtime_factor", "Processing time divided by audio duration (lower is faster).",
    ["engine", "model"], registry=REGISTRY, buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 4, 8),
)
FAILURES = Counter(
    "semaje_stt_failures_total", "Failed transcriptions by error code.", ["code"], registry=REGISTRY,
)
REJECTED = Counter(
    "semaje_stt_rejected_total", "Requests rejected because the service was at capacity.", registry=REGISTRY,
)

# Only label known routes so random 404 paths can't blow up label cardinality.
KNOWN_PATHS = {
    "/transcribe", "/languages", "/models", "/models/download", "/models/activate",
    "/health", "/health/ready", "/health/live", "/metrics",
}


def observe_http(path: str, method: str, status: int, seconds: float) -> None:
    label = path if path in KNOWN_PATHS else ("/models/{id}" if path.startswith("/models/") else "other")
    HTTP_REQUESTS.labels(label, method, str(status)).inc()
    HTTP_LATENCY.labels(label).observe(seconds)


def observe_transcription(engine: str, model: str, audio_seconds: float, processing_seconds: float) -> None:
    AUDIO_SECONDS.labels(engine, model).inc(max(audio_seconds, 0))
    if audio_seconds > 0:
        REALTIME_FACTOR.labels(engine, model).observe(processing_seconds / audio_seconds)


def render() -> bytes:
    return generate_latest(REGISTRY)
