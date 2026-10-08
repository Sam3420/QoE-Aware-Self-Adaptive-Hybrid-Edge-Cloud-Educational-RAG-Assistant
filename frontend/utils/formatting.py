"""Small display formatters."""
from datetime import datetime, timezone


def short_id(value: str | None, n: int = 8) -> str:
    return value[:n] if value else "—"


def ms(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"{value / 1000:.2f} s" if value >= 1000 else f"{value:.0f} ms"


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def when(value: datetime | None) -> str:
    """Render a stored UTC timestamp in local time."""
    if value is None:
        return "—"
    if hasattr(value, "to_pydatetime"):  # pandas Timestamp (from a DataFrame column)
        value = value.to_pydatetime()
    if value.tzinfo is None:  # SQLite returns naive datetimes; the app stores UTC
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S")
