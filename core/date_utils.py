from __future__ import annotations

import calendar
from collections.abc import Sequence
from datetime import datetime, timedelta

from dateutil import parser as dateutil_parser

SPECIAL_PERIODS = {
    "christmas eve": ("dec 24", "dec 24"),
    "christmas day": ("dec 25", "dec 25"),
    "christmas": ("dec 25", "dec 25"),
    "halloween": ("oct 31", "oct 31"),
    "winter": ("january", "march"),
    "spring": ("march", "june"),
    "summer": ("june", "september"),
    "fall": ("september", "december"),
}


def parse_flexible_date_as_datetime(user_input: str) -> datetime | None:
    user_input = user_input.strip()
    try:
        return dateutil_parser.parse(user_input, default=datetime(1900, 1, 1, 0, 0, 0))
    except (ValueError, TypeError, dateutil_parser.ParserError):
        return None


def parse_flexible_date(user_input: str) -> str | None:
    """Parse flexible date input and return standardized format."""
    parsed_dt = parse_flexible_date_as_datetime(user_input)
    if parsed_dt is None:
        return None
    return parsed_dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_flexible_date_as_interval(
    user_input: str,
) -> tuple[datetime, datetime] | None:
    user_input = user_input.strip().lower()
    start_text = user_input
    end_text = user_input
    for k, (v1, v2) in SPECIAL_PERIODS.items():
        start_text = start_text.replace(k, v1)
        end_text = end_text.replace(k, v2)
    try:
        interval_start = dateutil_parser.parse(
            start_text, default=datetime(1900, 1, 1, 0, 0)
        )
        interval_end = dateutil_parser.parse(
            end_text, default=datetime(1900, 12, 28, 23, 59)
        )
    except (ValueError, TypeError, dateutil_parser.ParserError):
        return None
    if interval_start.day == 1 and interval_end.day == 28:
        # Get the actual end of this month
        interval_end = interval_end.replace(
            day=calendar.monthrange(interval_end.year, interval_end.month)[1]
        )
    return (interval_start, interval_end)


def recover_datetime_interval(
    parsed_dt: datetime, default_dt=datetime(1900, 1, 1, 0, 0, 0)
) -> tuple[datetime, datetime]:
    """
    Heuristically recover the interval of datetimes that could have produced
    the given parsed datetime, based on which components match the default values.

    Args:
        parsed_dt: The datetime that was parsed from user input
        default_dt: The default datetime used during parsing

    Returns:
        tuple: (start_datetime, end_datetime) representing the interval
    """

    # Start with the parsed datetime as our lower bound
    start_dt = parsed_dt

    # We'll build the upper bound by replacing default components with their max values
    year = parsed_dt.year
    month = parsed_dt.month
    day = parsed_dt.day
    hour = parsed_dt.hour
    minute = parsed_dt.minute
    second = parsed_dt.second
    microsecond = parsed_dt.microsecond

    # Check each component from least to most significant
    # If a component matches the default, it (and all less significant components)
    # should be expanded to their maximum values

    if microsecond == default_dt.microsecond:
        microsecond = 999999

    if second == default_dt.second:
        second = 59
        microsecond = 999999

    if minute == default_dt.minute:
        minute = 59
        second = 59
        microsecond = 999999

    if hour == default_dt.hour:
        hour = 23
        minute = 59
        second = 59
        microsecond = 999999

    if day == default_dt.day:
        # Get the last day of the month
        day = calendar.monthrange(year, month)[1]
        hour = 23
        minute = 59
        second = 59
        microsecond = 999999

    if month == default_dt.month:
        month = 12
        day = 31  # December always has 31 days
        hour = 23
        minute = 59
        second = 59
        microsecond = 999999

    # Year is always specified if we got a valid parse, so we don't check it

    end_dt = datetime(year, month, day, hour, minute, second, microsecond)

    return start_dt, end_dt


def interpolate_dates(
    date_intervals: Sequence[tuple[datetime, datetime] | None],
) -> list[datetime]:
    """
    Interpolate plausible dates for photos given interval constraints.

    Args:
        date_intervals: List where each element is either:
            - None: no date constraint for this photo
            - (start, end): datetime interval constraint for this photo

    Returns:
        List of datetime objects, one per photo, maintaining album order
        and satisfying all interval constraints.

    Raises:
        ValueError: If constraints are impossible to satisfy
    """
    if not date_intervals:
        return []

    n = len(date_intervals)

    # Step 1: Forward pass - compute earliest feasible date for each photo
    earliest: list[datetime] = [datetime(1900, 1, 1)] * n

    for i in range(n):
        interval = date_intervals[i]
        if interval is not None:
            # Constrained photo: earliest is max of interval start and previous photo's date
            interval_start = interval[0]
            if i == 0:
                earliest[i] = interval_start
            else:
                earliest[i] = max(interval_start, earliest[i - 1])
        else:
            # Unconstrained photo: earliest is same as previous photo (or default)
            if i == 0:
                # No previous photo and no constraint - use a reasonable default
                earliest[i] = datetime(1900, 1, 1)
            else:
                earliest[i] = earliest[i - 1]

    # Step 2: Backward pass - compute latest feasible date for each photo
    latest: list[datetime] = [datetime(2100, 12, 31)] * n

    for i in range(n - 1, -1, -1):
        interval = date_intervals[i]
        if interval is not None:
            # Constrained photo: latest is min of interval end and next photo's date
            interval_end = interval[1]
            if i == n - 1:
                latest[i] = interval_end
            else:
                latest[i] = min(interval_end, latest[i + 1])
        else:
            # Unconstrained photo: latest is same as next photo (or default)
            if i == n - 1:
                # No next photo and no constraint - use a reasonable default
                latest[i] = datetime(2100, 12, 31)
            else:
                latest[i] = latest[i + 1]

    # Step 3: Check feasibility
    for i in range(n):
        if earliest[i] > latest[i]:
            raise ValueError(
                f"impossible constraint at photo {i}: no valid date satisfies ordering requirements."
            )

    # Step 4: Assign actual dates
    result: list[datetime] = [datetime(1900, 1, 1)] * n

    for i in range(n):
        if date_intervals[i] is not None:
            # Constrained photo: prefer interval start (earliest feasible date)
            result[i] = earliest[i]
        else:
            # Unconstrained photo: interpolate within feasible range
            # Find the surrounding constrained photos to interpolate between
            left_anchor = None
            right_anchor = None
            left_idx = -1
            right_idx = n

            # Find left anchor
            for j in range(i - 1, -1, -1):
                if date_intervals[j] is not None:
                    left_anchor = earliest[j]
                    left_idx = j
                    break

            # Find right anchor
            for j in range(i + 1, n):
                if date_intervals[j] is not None:
                    right_anchor = earliest[j]  # Use earliest, not latest
                    right_idx = j
                    break

            if left_anchor is not None and right_anchor is not None:
                # Interpolate between anchors
                total_gaps = right_idx - left_idx
                current_gap = i - left_idx

                # Linear interpolation
                time_diff = right_anchor - left_anchor
                interpolated_offset = time_diff * current_gap / total_gaps
                result[i] = left_anchor + interpolated_offset

            elif left_anchor is not None:
                # Only left anchor - increment by small amount
                increment_seconds = (i - left_idx) * 60  # 1 minute per photo
                result[i] = left_anchor + timedelta(seconds=increment_seconds)

            elif right_anchor is not None:
                # Only right anchor - decrement by small amount
                decrement_seconds = (right_idx - i) * 60  # 1 minute per photo
                result[i] = right_anchor - timedelta(seconds=decrement_seconds)

            else:
                # No anchors - space photos evenly in a reasonable range
                base_date = datetime(2000, 1, 1)  # Arbitrary reasonable default
                increment_seconds = i * 3600  # 1 hour per photo
                result[i] = base_date + timedelta(seconds=increment_seconds)

    return result
