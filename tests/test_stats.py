from chinchilla_llm_bench.stats import (
    PeakTracker,
    RateTracker,
    RunSummary,
    fmt_mean_std,
    fmt_ms_mean_std,
    mean_std,
    peak_rate,
)


def test_mean_std_empty():
    assert mean_std([]) == (0.0, 0.0)


def test_mean_std_single():
    assert mean_std([5.0]) == (5.0, 0.0)


def test_mean_std_known_values():
    m, s = mean_std([2.0, 4.0, 6.0])
    assert m == 4.0
    assert abs(s - 2.0) < 1e-9  # sample std of [2,4,6]


def test_fmt_mean_std():
    assert fmt_mean_std(4448.64, 1603.51) == "4448.64 ± 1603.51"


def test_fmt_ms_mean_std():
    assert fmt_ms_mean_std(0.14427, 0.01948) == "144.27 ± 19.48"


def test_run_summary_duration_label_carries_rounded_seconds():
    assert RunSummary(59.999, 0, 0, 0).duration_label == "1m 0.00s"
    assert RunSummary(3599.999, 0, 0, 0).duration_label == "1h 0m 0.00s"
    assert RunSummary(-1, 0, 0, 0).duration_label == "0.00s"


def test_peak_tracker_tracks_peak_and_rate():
    tracker = PeakTracker(window=1.0)
    now = 0.0
    for i in range(10):
        tracker.add(now)
        now += 0.1
    assert abs(tracker.peak - 10.0) < 1e-6
    assert abs(tracker.rate(now) - 10.0) < 1e-6


def test_peak_tracker_empty():
    tracker = PeakTracker()
    assert tracker.rate(1.0) == 0.0


def test_rate_tracker_counts_trailing_window():
    tracker = RateTracker(window=1.0)
    now = 0.0
    for _ in range(5):
        tracker.add(now)
        now += 0.1
    assert abs(tracker.rate(now) - 5.0) < 1e-9
    # after a 2s gap the old events have aged out of the 1s window
    assert tracker.rate(now + 2.0) == 0.0


def test_rate_tracker_empty():
    tracker = RateTracker()
    assert tracker.rate(5.0) == 0.0


def test_peak_rate_sliding_window():
    # 100 tokens spread over 10s = mean 10/s, but a burst of 50 in 0.5s
    # inside the 1s window -> peak 50
    times = [i * 0.01 for i in range(50)] + [2.0 + i * 0.1 for i in range(50)]
    assert abs(peak_rate(times) - 50.0) < 1e-9


def test_peak_rate_uniform_stream():
    # 20 tokens at 10/s over 1.9s: any 1s window holds at most 10 tokens
    times = [i * 0.1 for i in range(20)]
    assert abs(peak_rate(times) - 10.0) < 1e-9


def test_peak_rate_short_burst_uses_actual_span():
    # whole stream shorter than the window -> tokens / actual span
    times = [0.0, 0.05, 0.10, 0.15]
    assert abs(peak_rate(times) - 4.0 / 0.15) < 1e-9


def test_peak_rate_empty_and_single():
    assert peak_rate([]) == 0.0
    assert abs(peak_rate([0.5]) - 1.0) < 1e-9  # 1 token / 1s window
