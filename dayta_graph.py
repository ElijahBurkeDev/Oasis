"""dayta_graph.py - draw saved daytaPoints as an inline, colour text chart.

Pure functions only: no file or network access (tools.load_recent_dayta reads the
files), so the drawing logic can be tested without LM Studio.

Layout:
  * x axis = real local time. Every local calendar day gets an equal-width column
    group and a report sits at its time of day inside that group, so a morning
    report is always drawn left of an evening one. Reports are placed by their own
    timestamps, so one saved UTC day that straddles local midnight still draws
    correctly.
  * y axis = mood 1-10, one text row per value, labelled in the mood colour.
  * hollow circle = first report of the day, filled circle = latest report, joined
    by a line that is green when the mood rose and red when it fell.
  * dotted line = least-squares line of best fit through every report drawn in the
    window. It runs from the first to the last report (never extrapolated) and is
    only drawn once there are enough reports on enough days (see GraphConfig).
  * Colour is optional: with it off, the shapes alone still carry the meaning.

render_mood_chart() returns the text plus ChartStats: plain facts about the chart
(average, latest, trend direction and slope) so a caller, such as the chat model,
can comment on it without doing any arithmetic. build_mood_chart() is the
text-only shortcut.
"""
from __future__ import annotations

import os
import shutil
import sys
import textwrap
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

RESET, DIM, BOLD, TODAY = "\x1b[0m", "\x1b[2m", "\x1b[1m", "\x1b[1;4m"

# One 256-colour code per mood value 1..10: red -> orange -> yellow -> green.
MOOD_COLORS = (196, 202, 208, 214, 220, 190, 154, 118, 82, 46)
RISE_COLOR, FALL_COLOR = 34, 160
FIT_COLOR = 39  # line of best fit: blue, so it is never mistaken for a mood colour or a rise/fall line

# Change these if your console font shows boxes instead of circles.
GLYPH_FIRST, GLYPH_LAST, GLYPH_BOTH = "○", "●", "◉"
GLYPH_FIT = "·"  # one per column; plain Latin-1, so it survives even legacy console code pages

LEFT = 4  # width of the "10 ┤" y-axis prefix
_WEEKDAYS = ("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


@dataclass(frozen=True)
class GraphConfig:
    max_days: int = 30    # default window cap, in calendar days
    min_days: int = 7     # default window is never narrower than this
    min_slot: int = 4     # narrowest column group per day, in characters
    max_slot: int = 10    # widest column group per day
    # --- line of best fit ---
    fit_min_points: int = 3          # fewest distinct reports needed to draw the line
    fit_min_days: int = 2            # ...spread over at least this many calendar days
    fit_flat_per_week: float = 0.3   # a slope smaller than this many mood points per week counts as "flat"
    fit_confident_days: int = 5      # fewer days than this and the trend is flagged as tentative


@dataclass(frozen=True)
class _Report:
    when: datetime  # local time, timezone-aware
    mood: int


@dataclass(frozen=True)
class ChartStats:
    """Facts about what the chart shows. Deterministic, so nobody has to redo the arithmetic."""
    window: str                     # e.g. "Nov 3 - Nov 15"
    logged_days: int
    average: float                  # mean of each day's latest mood
    lowest: int
    highest: int
    latest: int                     # latest mood of the newest logged day
    trend: str                      # "rising", "falling", "flat", or "unknown" (no line of best fit)
    change_per_week: float | None   # slope of the line of best fit, mood points per week
    reports_fitted: int             # how many reports the line was fitted to (0 when no line)
    confident: bool                 # False when the trend rests on only a few days

    def as_dict(self) -> dict[str, Any]:
        """Compact, JSON-safe form for a tool result or prompt."""
        data: dict[str, Any] = {
            "window": self.window,
            "logged_days": self.logged_days,
            "average_mood": round(self.average, 1),
            "lowest_mood": self.lowest,
            "highest_mood": self.highest,
            "most_recent_day_mood": self.latest,
            "trend": self.trend,
        }
        if self.change_per_week is not None:
            if self.confident:
                data["trend_points_per_week"] = _signed(self.change_per_week)
                data["trend_confidence"] = "reasonable"
            else:  # a weekly rate extrapolated from a few days is misleading, so give direction only
                data["trend_confidence"] = "low (only a few days of data; direction only)"
        else:
            data["trend_note"] = "not enough data for a line of best fit yet"
        return data


@dataclass(frozen=True)
class MoodChart:
    text: str                   # printable chart, "" when there is nothing to draw
    stats: ChartStats | None    # None when there is no chart


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------
def terminal_width(default: int = 80) -> int:
    return shutil.get_terminal_size((default, 24)).columns


def _enable_windows_ansi() -> bool:
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))  # VT processing
    except Exception:
        return False


def color_supported() -> bool:
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    if not getattr(sys.stdout, "isatty", lambda: False)():
        return False
    return _enable_windows_ansi() if os.name == "nt" else True


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def _local(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone()  # stored as UTC, drawn in local time


def _pairs(points: list[dict[str, Any]]) -> list[tuple[_Report, _Report]]:
    """(first report, latest report) per daytaPoint, oldest day first. Bad points are skipped."""
    pairs: list[tuple[_Report, _Report]] = []
    for point in points:
        try:
            first = _Report(_local(point["initialTimestamp"]), int(point["initialReportedMoodValue"]))
            last = _Report(_local(point["currentTimestamp"]), int(point["currentReportedMoodValue"]))
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= first.mood <= 10 and 1 <= last.mood <= 10:
            pairs.append((first, last))
    pairs.sort(key=lambda pair: pair[0].when)
    return pairs


def _clock_fraction(when: datetime) -> float:
    """How far through its local day a time is: 0.0 at midnight, just under 1.0 at 23:59:59."""
    return (when.hour * 3600 + when.minute * 60 + when.second) / 86_400


def _best_fit(points: list[tuple[float, int]]) -> tuple[float, float] | None:
    """Least-squares line through (x in days, mood) points -> (slope, intercept); None if undefined."""
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    spread = sum((x - mean_x) ** 2 for x, _ in points)
    if spread < 1e-9:  # every report at the same moment: no slope to speak of
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / spread
    return slope, mean_y - slope * mean_x


def _signed(value: float) -> float:
    """Round to 1 dp and avoid a negative zero, so a flat trend never prints as -0.0."""
    return round(value, 1) + 0.0


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _fg(code: int, bold: bool = False) -> str:
    return f"\x1b[{'1;' if bold else ''}38;5;{code}m"


def _paint(text: str, style: str, color: bool) -> str:
    return f"{style}{text}{RESET}" if color and style else text


def _wrap(items: list[tuple[int, str]], width: int) -> list[str]:
    """Join (visible_length, text) items with 3 spaces, starting a new line when one would overflow."""
    lines: list[str] = []
    current, length = "", 0
    for visible, text in items:
        if current and length + 3 + visible > width:
            lines.append(current)
            current, length = "", 0
        current = f"{current}   {text}" if current else text
        length = length + 3 + visible if length else visible
    if current:
        lines.append(current)
    return lines


def build_mood_chart(
    points: list[dict[str, Any]],
    *,
    days: int | None = None,
    width: int | None = None,
    color: bool | None = None,
    today: date | None = None,
    cfg: GraphConfig = GraphConfig(),
) -> str:
    """Text-only shortcut for render_mood_chart(): the chart, or "" when there is nothing to draw."""
    return render_mood_chart(points, days=days, width=width, color=color, today=today, cfg=cfg).text


def render_mood_chart(
    points: list[dict[str, Any]],
    *,
    days: int | None = None,
    width: int | None = None,
    color: bool | None = None,
    today: date | None = None,
    cfg: GraphConfig = GraphConfig(),
) -> MoodChart:
    """Return the chart as printable text plus facts about it. Text is "" when there is nothing to draw.

    days  - how many recent calendar days to show; None picks a window that fits the data.
            Always capped by what fits the terminal width at cfg.min_slot per day.
    width - total characters available; None uses the terminal.
    color - ANSI colour on/off; None auto-detects.
    today - local date to end the chart on; None uses the clock.
    """
    pairs = _pairs(points)
    if not pairs:
        return MoodChart("", None)
    color = color_supported() if color is None else color
    width = max(30, (terminal_width() - 1) if width is None else width)
    today = datetime.now().astimezone().date() if today is None else today
    usable = width - LEFT

    # --- window: which calendar days, and how wide each one is -------------------------
    fit = max(1, usable // cfg.min_slot)
    report_days = [report.when.date() for pair in pairs for report in pair]
    oldest, newest = min(report_days), max(report_days)
    end = max(today, newest)
    if days is None:
        span = min(max(cfg.min_days, (end - oldest).days + 1), cfg.max_days, fit)
    else:
        span = max(1, min(days, fit))
    if newest < end - timedelta(days=span - 1):  # nothing recent: end on the newest report instead
        end = newest
    start = end - timedelta(days=span - 1)
    slot = max(cfg.min_slot, min(cfg.max_slot, usable // span))
    chart_width = span * slot

    # --- cell grid: row 0 is mood 10, row 9 is mood 1 ----------------------------------
    grid = [[(" ", "") for _ in range(chart_width)] for _ in range(10)]

    def put(row: int, col: int, char: str, style: str = "") -> None:
        grid[row][col] = (char, style)

    def place(report: _Report) -> tuple[int, int] | None:
        offset = (report.when.date() - start).days
        if not 0 <= offset < span:
            return None
        col = offset * slot + min(slot - 1, int(_clock_fraction(report.when) * slot))
        return col, 10 - report.mood

    for index in range(1, span):  # faint midnight separators
        for row in range(10):
            put(row, index * slot, "┊", DIM)

    # --- line of best fit: every distinct report drawn in the window --------------------
    fit_points: list[tuple[float, int, int]] = []  # (days since window start, mood, column)
    seen: set[tuple[datetime, int]] = set()
    for pair in pairs:
        for report in pair:
            spot = place(report)
            if spot is None or (report.when, report.mood) in seen:
                continue  # off-window, or a day with one report (first and latest are the same report)
            seen.add((report.when, report.mood))
            fit_points.append(((report.when.date() - start).days + _clock_fraction(report.when), report.mood, spot[0]))
    fit_days = len({int(x) for x, _, _ in fit_points})
    line = None
    if len(fit_points) >= cfg.fit_min_points and fit_days >= cfg.fit_min_days:
        line = _best_fit([(x, mood) for x, mood, _ in fit_points])
    fit_style = _fg(FIT_COLOR)

    if line is not None:  # drawn before connectors and markers, so those always stay on top
        slope, intercept = line
        previous: int | None = None
        for col in range(min(c for _, _, c in fit_points), max(c for _, _, c in fit_points) + 1):
            x = col // slot + (col % slot + 0.5) / slot  # centre of this column, in days since window start
            row = round(10 - (intercept + slope * x))
            if not 0 <= row <= 9:  # the line has left the 1-10 scale here
                previous = None
                continue
            put(row, col, GLYPH_FIT, fit_style)
            if previous is not None and abs(row - previous) > 1:  # steep: fill the gap so the line stays joined
                for gap in range(min(row, previous) + 1, max(row, previous)):
                    put(gap, col, GLYPH_FIT, fit_style)
            previous = row

    placed = [(pair, place(pair[0]), place(pair[1])) for pair in pairs]

    for (first, last), a, b in placed:  # connectors first, so markers always stay on top
        if a is None or b is None or a == b:
            continue
        (c1, r1), (c2, r2) = a, b
        style = _fg(RISE_COLOR) if last.mood > first.mood else _fg(FALL_COLOR) if last.mood < first.mood else DIM
        if r1 == r2:
            for col in range(min(c1, c2), max(c1, c2) + 1):
                put(r1, col, "─", style)
        elif c1 == c2:
            for row in range(min(r1, r2), max(r1, r2) + 1):
                put(row, c1, "│", style)
        else:
            for col in range(min(c1, c2), max(c1, c2) + 1):
                put(r1, col, "─", style)
            for row in range(min(r1, r2), max(r1, r2) + 1):
                put(row, c2, "│", style)
            rising = r2 < r1
            corner = ("┘" if rising else "┐") if c1 < c2 else ("└" if rising else "┌")
            put(r1, c2, corner, style)

    for (first, last), a, b in placed:
        if a is not None and a != b:
            put(a[1], a[0], GLYPH_FIRST, _fg(MOOD_COLORS[first.mood - 1], bold=True))
        if b is not None:
            same_spot = a == b and first.when != last.when
            put(b[1], b[0], GLYPH_BOTH if same_spot else GLYPH_LAST, _fg(MOOD_COLORS[last.mood - 1], bold=True))

    # --- facts about the window ---------------------------------------------------------
    first_label = f"{_MONTHS[start.month - 1]} {start.day}"
    last_label = f"{_MONTHS[end.month - 1]} {end.day}"
    span_label = first_label if start == end else f"{first_label} - {last_label}"

    shown = [last.mood for first, last in pairs if start <= first.when.date() <= end or start <= last.when.date() <= end]
    hidden = sum(1 for _, last in pairs if last.when.date() < start)

    stats: ChartStats | None = None
    if shown:
        trend, per_week = "unknown", None
        if line is not None:
            per_week = line[0] * 7
            trend = "flat" if abs(per_week) < cfg.fit_flat_per_week else "rising" if per_week > 0 else "falling"
        stats = ChartStats(
            window=span_label, logged_days=len(shown), average=sum(shown) / len(shown),
            lowest=min(shown), highest=max(shown), latest=shown[-1],
            trend=trend, change_per_week=per_week,
            reports_fitted=len(fit_points) if line is not None else 0,
            confident=line is not None and fit_days >= cfg.fit_confident_days,
        )

    # --- assemble text ------------------------------------------------------------------
    offset_text = datetime.now().astimezone().strftime("%z")
    zone = f"UTC{offset_text[:3]}:{offset_text[3:]}" if offset_text else "local time"
    lines = [_paint("Mood history", BOLD, color) + _paint(f"  {span_label}  (local time, {zone})", DIM, color)]

    for row in range(10):
        mood = 10 - row
        prefix = _paint(f"{mood:>2} ", _fg(MOOD_COLORS[mood - 1], bold=True), color) + _paint("┤", DIM, color)
        lines.append(prefix + "".join(_paint(char, style, color) for char, style in grid[row]))

    axis = "".join("┴" if col and col % slot == 0 else "─" for col in range(chart_width))
    lines.append("   " + _paint("└" + axis, DIM, color))

    weekday_row, day_row, month_row = [], [], []
    for index in range(span):
        day = start + timedelta(days=index)
        style = TODAY if day == today else DIM if day.weekday() >= 5 else ""
        weekday_row.append(_paint(_WEEKDAYS[day.weekday()].center(slot), style, color))
        day_row.append(_paint(str(day.day).center(slot), style, color))
        month = _MONTHS[day.month - 1] if index == 0 or day.day == 1 else ""
        month_row.append(_paint(month.ljust(slot), BOLD, color))
    for row in (weekday_row, day_row, month_row):
        lines.append(" " * LEFT + "".join(row))

    summary = (f"{len(shown)} logged day(s), avg {sum(shown) / len(shown):.1f}/10, range {min(shown)}-{max(shown)}"
               if shown else "no logged days in this window")
    if stats is not None and stats.change_per_week is not None:
        rate = f" ({_signed(stats.change_per_week):+.1f}/week)" if stats.confident else " (early days)"
        summary += f"; trend {stats.trend}{rate}"
    if hidden:
        summary += f"; {hidden} older day(s) not shown"
    if days is not None and days > span:
        summary += f"; showing {span} of {days} requested days (widen the terminal for more)"
    lines.append("")
    lines += [_paint(part, DIM, color) for part in textwrap.wrap(summary, width)]

    legend = [(len(text), text) for text in (
        f"{GLYPH_FIRST} first report", f"{GLYPH_LAST} latest report", f"{GLYPH_BOTH} same spot")]
    if line is not None:
        fit_text = f"{GLYPH_FIT} best fit"
        legend.append((len(fit_text), _paint(fit_text, fit_style, color)))
    if color:
        legend += [(len("green: rose"), _paint("green: rose", _fg(RISE_COLOR), True)),
                   (len("red: fell"), _paint("red: fell", _fg(FALL_COLOR), True))]
    else:
        legend.append((len("line joins a day's reports"), "line joins a day's reports"))
    lines += _wrap(legend, width)
    note = "Each column = one day, midnight to midnight; circles sit at their report time."
    lines += [_paint(part, DIM, color) for part in textwrap.wrap(note, width)]
    return MoodChart("\n".join(lines), stats)
