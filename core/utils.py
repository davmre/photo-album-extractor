from __future__ import annotations

from datetime import datetime

from dateutil import parser as dateutil_parser


def parse_flexible_date(user_input: str) -> str | None:
    """Parse flexible date input and return standardized format."""
    user_input = user_input.strip()
    if not user_input:
        return ""

    try:
        # Parse with dateutil - it's very flexible
        parsed_dt = dateutil_parser.parse(
            user_input, default=datetime(1900, 1, 1, 0, 0, 0)
        )
        # Return in standard format: YYYY-MM-DD HH:MM:SS
        return parsed_dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, dateutil_parser.ParserError):
        # If parsing fails, return None to indicate error
        return None
