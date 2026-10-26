"""Estimated cost per call, from Meta's published pay-as-you-go prices (dev.meta.ai/docs/pricing-rate-limits,
read 2026-10-26). These are ESTIMATES for the model_calls table, not invoices: re-check the page before a
batch run and compare against the billing dashboard.
"""
PRICES_READ_ON = "2026-10-26"

SPARK_PER_M = {"muse-spark-1.3": {"input": 1.25, "cached_input": 0.15, "output": 4.25}}
SAM_PER_1000_FRAMES = 0.20
SAM_MIN_FRAMES = 50
TRANSCRIBE_PER_HOUR = 0.18


def spark_cost(model: str, usage: dict | None) -> float | None:
    """Chat Completions usage: prompt_tokens / completion_tokens (documented field names)."""
    price = SPARK_PER_M.get(model)
    if not price or not isinstance(usage, dict):
        return None
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return None
    return round((prompt * price["input"] + completion * price["output"]) / 1_000_000, 6)


def sam_video_cost(frames_sent: int) -> float:
    """Billed per frame of the video we sent, minimum 50 frames per request."""
    return round(max(SAM_MIN_FRAMES, frames_sent) * SAM_PER_1000_FRAMES / 1000, 6)


def transcribe_cost(audio_duration_ms: int | None) -> float | None:
    """Metered in whole seconds of audio processed."""
    if not isinstance(audio_duration_ms, int):
        return None
    return round((audio_duration_ms // 1000) * TRANSCRIBE_PER_HOUR / 3600, 6)
