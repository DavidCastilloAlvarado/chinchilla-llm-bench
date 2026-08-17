from chinchilla_llm_bench.stats import (
    PeakTracker,
    RateTracker,
    fmt_mean_std,
    fmt_ms_mean_std,
    mean_std,
    peak_rate,
    window_rate,
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


def test_peak_rate_flat_stream():
    # 10 tokens at 0.1s apart -> 10 tok/s
    times = [i * 0.1 for i in range(10)]
    assert abs(peak_rate(times, window=0.5) - 10.0) < 1e-6


def test_peak_rate_burst_beats_flat():
    times = [0.0, 0.05, 0.1, 10.0, 10.1]
    # burst of 3 tokens in 0.1s -> 20 tok/s peak
    assert abs(peak_rate(times, window=0.5) - 20.0) < 1e-6


def test_peak_rate_too_few_samples():
    assert peak_rate([]) == 0.0
    assert peak_rate([0.1]) == 0.0


def test_window_rate_counts_arrivals_in_range():
    arrivals = [(t, 1) for t in (0.0, 0.1, 0.2, 5.0)]
    assert abs(window_rate(arrivals, 0.0, 0.3) - 10.0) < 1e-9  # 3 tok / 0.3s
    assert window_rate(arrivals, 1.0, 2.0) == 0.0
    assert window_rate(arrivals, 0.5, 0.5) == 0.0


def test_window_rate_weighted_entries():
    # pp-style: a burst of 500 prompt tokens counted at the ttfr moment
    arrivals = [(0.66, 500), (0.7, 1), (2.0, 500)]
    assert abs(window_rate(arrivals, 0.0, 0.75) - 501 / 0.75) < 1e-9
    assert abs(window_rate(arrivals, 0.66, 1.0) - 501 / 0.34) < 1e-6


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
