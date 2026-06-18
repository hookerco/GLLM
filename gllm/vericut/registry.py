from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkEnvelope:
    """Axis limits for a setup, expressed in the setup's expected units."""

    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    z_min: float | None = None
    z_max: float | None = None

    def bounds_for(self, axis: str) -> tuple[float | None, float | None]:
        axis_name = axis.lower()
        return (
            getattr(self, f"{axis_name}_min"),
            getattr(self, f"{axis_name}_max"),
        )


@dataclass(frozen=True)
class VericutSetup:
    """A local Vericut setup that generated G-code may target."""

    id: str
    description: str
    vericut_bat: Path
    project_template: Path
    library_search_paths: tuple[Path, ...]
    allowed_tools: frozenset[str]
    required_modes: frozenset[str]
    expected_units: str | None = None
    feed_rate_min: float | None = None
    feed_rate_max: float | None = None
    spindle_speed_min: float | None = None
    spindle_speed_max: float | None = None
    work_envelope: WorkEnvelope | None = None
    safe_z_min: float | None = None


class VericutSetupRegistry:
    def __init__(self, setups: tuple[VericutSetup, ...]):
        self._setups = {setup.id: setup for setup in setups}
        if len(self._setups) != len(setups):
            raise ValueError("Vericut setup registry contains duplicate setup IDs")

    def get(self, setup_id: str) -> VericutSetup:
        try:
            return self._setups[setup_id]
        except KeyError as exc:
            known = ", ".join(sorted(self._setups)) or "<none>"
            raise KeyError(f"Unknown Vericut setup {setup_id!r}. Known setups: {known}") from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._setups))


def load_setup_registry(path: str | Path) -> VericutSetupRegistry:
    registry_path = Path(path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = payload.get("setups")
    if not isinstance(entries, list):
        raise ValueError("Vericut setup registry must contain a 'setups' list")

    return VericutSetupRegistry(tuple(_setup_from_entry(entry) for entry in entries))


def validate_setup_paths(setup: VericutSetup) -> list[str]:
    missing: list[str] = []
    if not setup.vericut_bat.exists():
        missing.append(f"vericut_bat: {setup.vericut_bat}")
    if not setup.project_template.exists():
        missing.append(f"project_template: {setup.project_template}")
    for path in setup.library_search_paths:
        if not path.exists():
            missing.append(f"library_search_path: {path}")
    return missing


def normalize_tool(value: str | int) -> str:
    text = str(value).strip().upper()
    if text.startswith("T"):
        text = text[1:].strip()
    return f"T{text}"


def normalize_mode(value: str) -> str:
    return str(value).replace(" ", "").strip().upper()


def _setup_from_entry(entry: dict[str, Any]) -> VericutSetup:
    try:
        setup_id = str(entry["id"]).strip()
        description = str(entry.get("description", "")).strip()
        vericut_bat = _expand_path(entry["vericut_bat"])
        project_template = _expand_path(entry["project_template"])
    except KeyError as exc:
        raise ValueError(f"Vericut setup is missing required field: {exc.args[0]}") from exc

    if not setup_id:
        raise ValueError("Vericut setup ID cannot be blank")

    library_search_paths = tuple(
        _expand_path(value) for value in entry.get("library_search_paths", [])
    )
    allowed_tools = frozenset(normalize_tool(value) for value in entry.get("allowed_tools", []))
    required_modes = frozenset(normalize_mode(value) for value in entry.get("required_modes", []))
    expected_units = _normalize_units(entry.get("expected_units"))
    feed_rate_min = _optional_float(entry.get("feed_rate_min"))
    feed_rate_max = _optional_float(entry.get("feed_rate_max"))
    spindle_speed_min = _optional_float(entry.get("spindle_speed_min"))
    spindle_speed_max = _optional_float(entry.get("spindle_speed_max"))
    _validate_min_max("feed_rate", feed_rate_min, feed_rate_max)
    _validate_min_max("spindle_speed", spindle_speed_min, spindle_speed_max)
    work_envelope = _work_envelope_from_entry(entry.get("work_envelope"))
    safe_z_min = entry.get("safe_z_min")

    return VericutSetup(
        id=setup_id,
        description=description,
        vericut_bat=vericut_bat,
        project_template=project_template,
        library_search_paths=library_search_paths,
        allowed_tools=allowed_tools,
        required_modes=required_modes,
        expected_units=expected_units,
        feed_rate_min=feed_rate_min,
        feed_rate_max=feed_rate_max,
        spindle_speed_min=spindle_speed_min,
        spindle_speed_max=spindle_speed_max,
        work_envelope=work_envelope,
        safe_z_min=float(safe_z_min) if safe_z_min is not None else None,
    )


def _expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser()


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _normalize_units(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {
        "in": "inch",
        "inch": "inch",
        "inches": "inch",
        "imperial": "inch",
        "mm": "metric",
        "metric": "metric",
        "millimeter": "metric",
        "millimeters": "metric",
    }
    try:
        return aliases[text]
    except KeyError as exc:
        raise ValueError(
            "Vericut setup expected_units must be 'inch' or 'metric'"
        ) from exc


def _validate_min_max(name: str, minimum: float | None, maximum: float | None) -> None:
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"Vericut setup {name}_min cannot exceed {name}_max")


def _work_envelope_from_entry(value: Any) -> WorkEnvelope | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Vericut setup work_envelope must be an object")

    envelope = WorkEnvelope(
        x_min=_optional_float(value.get("x_min")),
        x_max=_optional_float(value.get("x_max")),
        y_min=_optional_float(value.get("y_min")),
        y_max=_optional_float(value.get("y_max")),
        z_min=_optional_float(value.get("z_min")),
        z_max=_optional_float(value.get("z_max")),
    )
    _validate_min_max("work_envelope.x", envelope.x_min, envelope.x_max)
    _validate_min_max("work_envelope.y", envelope.y_min, envelope.y_max)
    _validate_min_max("work_envelope.z", envelope.z_min, envelope.z_max)
    return envelope
