"""Interval overlap helpers independent of model and transaction loading."""


def temporal_interval_iou(true_start, true_end, predicted_start, predicted_end):
    """Return duration IoU of two time spans, or 0 for absent/zero-duration spans.

    Boundaries must be comparable datetime-like objects or None. A reversed
    interval is invalid. Touching endpoints have zero intersection duration.
    """
    boundaries = (true_start, true_end, predicted_start, predicted_end)
    if any(value is None for value in boundaries):
        return 0.0
    if true_end < true_start or predicted_end < predicted_start:
        raise ValueError("Interval end must not precede its start.")
    intersection = max(
        0.0,
        (min(true_end, predicted_end) - max(true_start, predicted_start)).total_seconds(),
    )
    union = (max(true_end, predicted_end) - min(true_start, predicted_start)).total_seconds()
    return intersection / union if union > 0 else 0.0
