from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DAYTA_DIRECTORY = HERE / "dayta"
_DAYTA_FILENAME = re.compile(r"^dayta-(\d+)\.json$")


def _validated_existing_dayta(file_path: Path, epoch_day: int) -> dict[str, Any]:
    """Load and validate an existing daytaPoint without modifying it."""
    try:
        with file_path.open("r", encoding="utf-8") as existing_file:
            root = json.load(existing_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Existing dayta file is invalid JSON: {file_path}") from error
    except OSError as error:
        raise OSError(f"Could not read existing dayta file: {file_path}") from error

    if not isinstance(root, dict):
        raise ValueError("Existing dayta file must contain a JSON object")
    point = root.get("daytaPoint")
    if not isinstance(point, dict):
        raise ValueError("Existing dayta file must contain a 'daytaPoint' object")
    if point.get("date") != epoch_day:
        raise ValueError(
            f"Existing daytaPoint date {point.get('date')!r} does not match filename day {epoch_day}"
        )

    required = {
        "initialTimestamp": str,
        "currentTimestamp": str,
        "initialReportedMoodValue": int,
        "currentReportedMoodValue": int,
        "initialAssessment": str,
        "currentAssessment": str,
        "initialBecauseQuote": str,
        "currentBecauseQuote": str,
    }
    for key, expected_type in required.items():
        value = point.get(key)
        if expected_type is int:
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10:
                raise ValueError(f"Existing {key} must be an integer between 1 and 10")
        elif not isinstance(value, expected_type):
            raise ValueError(f"Existing {key} must be a {expected_type.__name__}")

    for key in ("initialTimestamp", "currentTimestamp"):
        try:
            parsed = datetime.fromisoformat(point[key])
        except ValueError as error:
            raise ValueError(f"Existing {key} is not a valid ISO 8601 timestamp") from error
        if parsed.tzinfo is None:
            raise ValueError(f"Existing {key} must include a timezone offset")

    return point


def load_recent_dayta(max_entries: int = 14, max_age_days: int = 90) -> list[dict[str, Any]]:
    """Return up to max_entries valid daytaPoints, newest first.

    Corrupt, mismatched or unreadable files are skipped, never raised: recall is a
    convenience and must not be able to break a chat turn. Temp files written by
    save_data (".dayta-<day>-*.tmp") never match the filename pattern.
    """
    try:
        names = os.listdir(DAYTA_DIRECTORY)
    except OSError:
        return []
    today = int(datetime.now(timezone.utc).timestamp() // 86_400)
    days = sorted((int(m.group(1)) for n in names if (m := _DAYTA_FILENAME.match(n))), reverse=True)

    points: list[dict[str, Any]] = []
    for day in days:
        if day > today:
            continue  # future-dated (clock change / stray file)
        if today - day > max_age_days:
            break  # sorted newest-first, so everything after this is older still
        try:
            points.append(_validated_existing_dayta(DAYTA_DIRECTORY / f"dayta-{day}.json", day))
        except (OSError, ValueError):
            continue
        if len(points) >= max_entries:
            break
    return points


def save_data(
    reported_mood_value: int,
    assessment: str = "",
    because_quote: str = "",
) -> Path:
    """Create today's daytaPoint or update only its current fields."""
    if isinstance(reported_mood_value, bool) or not isinstance(reported_mood_value, int):
        raise TypeError("reported_mood_value must be an integer")
    if not 1 <= reported_mood_value <= 10:
        raise ValueError("reported_mood_value must be between 1 and 10")
    if not isinstance(assessment, str):
        raise TypeError("assessment must be a string")
    if not isinstance(because_quote, str):
        raise TypeError("because_quote must be a string")

    now = datetime.now(timezone.utc)
    now_text = now.isoformat()
    epoch_day = int(now.timestamp() // 86_400)

    DAYTA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    file_path = DAYTA_DIRECTORY / f"dayta-{epoch_day}.json"

    if file_path.exists():
        existing = _validated_existing_dayta(file_path, epoch_day)
        point: dict[str, Any] = {
            "date": epoch_day,
            "initialTimestamp": existing["initialTimestamp"],
            "currentTimestamp": now_text,
            "initialReportedMoodValue": existing["initialReportedMoodValue"],
            "currentReportedMoodValue": reported_mood_value,
            "initialAssessment": existing["initialAssessment"],
            "currentAssessment": assessment,
            "initialBecauseQuote": existing["initialBecauseQuote"],
            "currentBecauseQuote": because_quote,
        }
    else:
        point = {
            "date": epoch_day,
            "initialTimestamp": now_text,
            "currentTimestamp": now_text,
            "initialReportedMoodValue": reported_mood_value,
            "currentReportedMoodValue": reported_mood_value,
            "initialAssessment": assessment,
            "currentAssessment": assessment,
            "initialBecauseQuote": because_quote,
            "currentBecauseQuote": because_quote,
        }

    value: dict[str, Any] = {"daytaPoint": point}
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=DAYTA_DIRECTORY,
            prefix=f".{file_path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            json.dump(value, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)

        os.replace(temporary_path, file_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return file_path
