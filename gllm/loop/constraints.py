from __future__ import annotations


def _format_optional_range(minimum: float | None, maximum: float | None) -> str:
    lower = "-inf" if minimum is None else f"{minimum:g}"
    upper = "inf" if maximum is None else f"{maximum:g}"
    return f"[{lower}, {upper}]"


def setup_constraints(setup) -> tuple[str, ...]:
    """Human-readable machine constraints fed into generate/repair/improve prompts."""
    constraints: list[str] = []
    if setup.allowed_tools:
        constraints.append(f"allowed_tools: {', '.join(sorted(setup.allowed_tools))}")
    if setup.required_modes:
        constraints.append(f"required_modes: {', '.join(sorted(setup.required_modes))}")
    if setup.expected_units:
        constraints.append(f"expected_units: {setup.expected_units}")
    if setup.feed_rate_min is not None or setup.feed_rate_max is not None:
        constraints.append(
            f"feed_rate_range: {_format_optional_range(setup.feed_rate_min, setup.feed_rate_max)}"
        )
    if setup.spindle_speed_min is not None or setup.spindle_speed_max is not None:
        constraints.append(
            "spindle_speed_range: "
            f"{_format_optional_range(setup.spindle_speed_min, setup.spindle_speed_max)}"
        )
    if setup.work_envelope is not None:
        envelope = setup.work_envelope
        constraints.append(
            "work_envelope: "
            f"X{_format_optional_range(envelope.x_min, envelope.x_max)}, "
            f"Y{_format_optional_range(envelope.y_min, envelope.y_max)}, "
            f"Z{_format_optional_range(envelope.z_min, envelope.z_max)}"
        )
    if setup.safe_z_min is not None:
        constraints.append(f"safe_z_min: {setup.safe_z_min:g}")
    return tuple(constraints)
