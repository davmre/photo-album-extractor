from __future__ import annotations

from datetime import datetime

import pytest

from core import date_utils


def test_recover_datetime_interval():
    """Test the function with various inputs"""

    test_cases = [
        # (description, parsed_datetime, expected_start, expected_end)
        (
            "Year only",
            datetime(1998, 1, 1, 0, 0, 0),
            datetime(1998, 1, 1, 0, 0, 0),
            datetime(1998, 12, 31, 23, 59, 59, 999999),
        ),
        (
            "Year and month",
            datetime(1998, 6, 1, 0, 0, 0),
            datetime(1998, 6, 1, 0, 0, 0),
            datetime(1998, 6, 30, 23, 59, 59, 999999),
        ),
        (
            "Full date",
            datetime(1998, 6, 15, 0, 0, 0),
            datetime(1998, 6, 15, 0, 0, 0),
            datetime(1998, 6, 15, 23, 59, 59, 999999),
        ),
        (
            "Date and hour",
            datetime(1998, 6, 15, 14, 0, 0),
            datetime(1998, 6, 15, 14, 0, 0),
            datetime(1998, 6, 15, 14, 59, 59, 999999),
        ),
        (
            "Date, hour, and minute",
            datetime(1998, 6, 15, 14, 30, 0),
            datetime(1998, 6, 15, 14, 30, 0),
            datetime(1998, 6, 15, 14, 30, 59, 999999),
        ),
        (
            "Fully specified",
            datetime(1998, 6, 15, 14, 30, 45),
            datetime(1998, 6, 15, 14, 30, 45),
            datetime(1998, 6, 15, 14, 30, 45, 999999),
        ),
    ]

    for description, parsed_dt, expected_start, expected_end in test_cases:
        start, end = date_utils.recover_datetime_interval(parsed_dt)
        print(f"{description}:")
        print(f"  Input:    {parsed_dt}")
        print(f"  Interval: {start} to {end}")
        print(f"  Expected: {expected_start} to {expected_end}")
        print(f"  Correct:  {start == expected_start and end == expected_end}")
        print()
        assert start == expected_start
        assert end == expected_end


def test_interpolate_dates_basic():
    """Test basic interpolation with simple cases"""

    # Case 1: Simple two-point interpolation with no constraints
    intervals = [None, None]
    result = date_utils.interpolate_dates(intervals)
    assert len(result) == 2
    assert result[0] <= result[1]

    # Case 2: Single constrained point
    start_1998 = datetime(1998, 1, 1)
    end_1998 = datetime(1998, 12, 31, 23, 59, 59, 999999)
    intervals = [(start_1998, end_1998)]
    result = date_utils.interpolate_dates(intervals)
    assert len(result) == 1
    assert start_1998 <= result[0] <= end_1998


def test_interpolate_dates_ordering():
    """Test that output dates maintain album order"""

    # Case: Mixed constrained and unconstrained photos
    start_1998 = datetime(1998, 1, 1)
    end_1998 = datetime(1998, 12, 31, 23, 59, 59, 999999)
    start_2000 = datetime(2000, 1, 1)
    end_2000 = datetime(2000, 12, 31, 23, 59, 59, 999999)

    intervals = [
        (start_1998, end_1998),  # Photo 1: sometime in 1998
        None,  # Photo 2: no constraint
        None,  # Photo 3: no constraint
        (start_2000, end_2000),  # Photo 4: sometime in 2000
    ]

    result = date_utils.interpolate_dates(intervals)
    assert len(result) == 4

    # Check ordering
    for i in range(len(result) - 1):
        assert result[i] <= result[i + 1]

    # Check constraints
    assert start_1998 <= result[0] <= end_1998
    assert start_2000 <= result[3] <= end_2000


def test_interpolate_dates_precision_preference():
    """Test preference for interval beginnings"""

    # Case: Single year constraint - should prefer Jan 1
    start_1998 = datetime(1998, 1, 1)
    end_1998 = datetime(1998, 12, 31, 23, 59, 59, 999999)
    intervals = [(start_1998, end_1998)]

    result = date_utils.interpolate_dates(intervals)
    assert result[0] == start_1998  # Should choose beginning of interval


def test_interpolate_dates_tight_constraints():
    """Test handling of tightly constrained sequences"""

    # Case: Overlapping intervals that barely work
    jan_1998 = datetime(1998, 1, 15)
    feb_1998 = datetime(1998, 2, 15)
    mar_1998 = datetime(1998, 3, 15)

    intervals = [
        (jan_1998, feb_1998),  # Photo 1: Jan 15 - Feb 15
        (jan_1998, mar_1998),  # Photo 2: Jan 15 - Mar 15 (overlaps)
        (feb_1998, mar_1998),  # Photo 3: Feb 15 - Mar 15
    ]

    result = date_utils.interpolate_dates(intervals)
    assert len(result) == 3

    # Check ordering
    assert result[0] <= result[1] <= result[2]

    # Check constraints
    assert jan_1998 <= result[0] <= feb_1998
    assert jan_1998 <= result[1] <= mar_1998
    assert feb_1998 <= result[2] <= mar_1998


def test_interpolate_dates_impossible_constraints():
    """Test detection of impossible constraint combinations"""

    # Case: Impossible ordering - later photo has earlier constraint
    early_date = datetime(1998, 1, 1)
    late_date = datetime(2000, 1, 1)

    intervals = [
        (late_date, late_date),  # Photo 1: must be in 2000
        (early_date, early_date),  # Photo 2: must be in 1998 (impossible!)
    ]

    with pytest.raises(ValueError, match="impossible.*constraint"):
        date_utils.interpolate_dates(intervals)


def test_interpolate_dates_long_sequence():
    """Test interpolation with many unconstrained photos between anchors"""

    start_1998 = datetime(1998, 1, 1)
    start_2000 = datetime(2000, 1, 1)

    # 1998 anchor, 5 unconstrained photos, 2000 anchor
    intervals = [
        (start_1998, start_1998),
        None,
        None,
        None,
        None,
        None,
        (start_2000, start_2000),
    ]

    result = date_utils.interpolate_dates(intervals)
    assert len(result) == 7

    # Check ordering
    for i in range(len(result) - 1):
        assert result[i] <= result[i + 1]

    # Check constraints
    assert result[0] == start_1998
    assert result[6] == start_2000

    # Check that middle photos are interpolated between anchors
    for i in range(1, 6):
        assert start_1998 <= result[i] <= start_2000


def test_interpolate_dates_edge_cases():
    """Test edge cases"""

    # Empty list
    assert date_utils.interpolate_dates([]) == []

    # All None
    intervals = [None, None, None]
    result = date_utils.interpolate_dates(intervals)
    assert len(result) == 3
    for i in range(len(result) - 1):
        assert result[i] <= result[i + 1]


def test_interpolate_dates_unique_dates():
    intervals = [
        (datetime(1993, 1, 1, 0, 0), datetime(1993, 12, 31, 23, 59, 59, 999999)),
        None,
        (datetime(1993, 1, 1, 0, 0), datetime(1993, 12, 31, 23, 59, 59, 999999)),
        None,
        None,
        None,
        None,
    ]
    results = date_utils.interpolate_dates(intervals)
    unique_results = set(results)
    assert len(results) == len(unique_results)


def test_interpolate_dates_backwards_propagation():
    # Date for an unspecified first element should be informed by its successor.
    start_jan_1998 = datetime(1998, 1, 1)
    end_jan_1998 = datetime(1998, 1, 31)

    intervals = [None, (start_jan_1998, end_jan_1998)]
    result = date_utils.interpolate_dates(intervals)
    assert len(result) == 2

    target_intervals = [
        (datetime(1997, 12, 31), start_jan_1998),
        (start_jan_1998, end_jan_1998),
    ]

    for d, (start, end) in zip(result, target_intervals):
        assert d >= start
        assert d <= end


def test_interpolate_dates_segmented_basic():
    """Test basic segmented interpolation"""

    # Case: Simple impossible constraint that should be segmented
    early_date = datetime(1998, 1, 1)
    late_date = datetime(2000, 1, 1)

    intervals = [
        (late_date, late_date),  # Photo 1: 2000 (impossible with next)
        (early_date, early_date),  # Photo 2: 1998 (creates conflict)
        None,  # Photo 3: unconstrained
        (late_date, late_date),  # Photo 4: 2000 again
    ]

    # Regular interpolation should fail
    with pytest.raises(ValueError):
        date_utils.interpolate_dates(intervals)

    # Segmented interpolation should work
    segments = date_utils.interpolate_dates_segmented(intervals)
    result = [d for segment in segments for d in segment]
    assert len(segments) == 2
    assert len(result) == 4

    # Check that constraints are satisfied
    assert result[0] == late_date  # First segment: photo 1
    assert result[1] == early_date  # Second segment: photos 2-4
    assert result[2] >= result[1]
    assert result[3] == late_date


def test_interpolate_dates_segmented_clusters():
    """Test segmentation with topic clusters"""

    # Case: Two coherent clusters separated by impossible constraint
    jan_1992 = datetime(1992, 1, 1)
    jul_1992 = datetime(1992, 7, 1)
    jan_1991 = datetime(1991, 1, 1)  # Flashback
    aug_1992 = datetime(1992, 8, 1)

    intervals = [
        (jan_1992, jan_1992),  # Cluster 1: Jan 1992
        (jan_1992, jan_1992),  # Cluster 1: Jan 1992
        (jul_1992, jul_1992),  # Cluster 1: Jul 1992
        (jan_1991, jan_1991),  # Outlier: Jan 1991 (impossible!)
        (aug_1992, aug_1992),  # Cluster 2: Aug 1992
        (aug_1992, aug_1992),  # Cluster 2: Aug 1992
    ]

    segments = date_utils.interpolate_dates_segmented(intervals)
    result = [d for segment in segments for d in segment]
    assert len(result) == 6

    # Check segment 1 (photos 0-2): Jan-Jul 1992
    assert result[0] == jan_1992
    assert result[1] > jan_1992  # Should be slightly later to ensure uniqueness
    assert result[2] == jul_1992
    assert result[0] <= result[1] <= result[2]

    # Check outlier (photo 3): Jan 1991
    assert result[3] == jan_1991

    # Check segment 2 (photos 4-5): Aug 1992
    assert result[4] == aug_1992
    assert result[5] > aug_1992  # Should be slightly later to ensure uniqueness
    assert result[4] <= result[5]

    # Check that all dates are unique
    assert len(result) == len(set(result))


def test_interpolate_dates_segmented_fallback():
    """Test that segmented interpolation works on cases where regular interpolation works"""

    # Case: Simple case that regular interpolation handles fine
    start_1998 = datetime(1998, 1, 1)
    end_1998 = datetime(1998, 12, 31, 23, 59, 59, 999999)
    start_2000 = datetime(2000, 1, 1)
    end_2000 = datetime(2000, 12, 31, 23, 59, 59, 999999)

    intervals = [(start_1998, end_1998), None, None, (start_2000, end_2000)]

    # Both should work and give same result
    regular_result = date_utils.interpolate_dates(intervals)
    segmented_result = date_utils.interpolate_dates_segmented(intervals)[0]

    assert len(regular_result) == len(segmented_result) == 4
    assert regular_result == segmented_result


def test_two_digit_years():
    cases = [
        ("7/99", datetime(1999, 7, 1)),
        ("7/00", datetime(2000, 7, 1)),
        ("7/01", datetime(2001, 7, 1)),
        ("07/01", datetime(2001, 7, 1)),
        ("12/01", datetime(2001, 12, 1)),
        ("12/2000", datetime(2000, 12, 1)),
        ("12/25/97", datetime(1997, 12, 25)),
        ("1999/12/01", datetime(1999, 12, 1)),
        ("2000/4/01", datetime(2000, 4, 1)),
        ("'99 12 2", datetime(1999, 12, 2)),
        ("'01 8 7", datetime(2001, 8, 7)),
        ("8 7 '01", datetime(2001, 8, 7)),
        ("Aug 9 05", datetime(2005, 8, 9)),
        ("Aug 05", datetime(2005, 8, 1)),
    ]
    for in_str, expected in cases:
        result = date_utils.parse_flexible_date_as_datetime(in_str)
        assert result == expected
