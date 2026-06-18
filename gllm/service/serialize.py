from __future__ import annotations

import json

from gllm.loop.packet import result_to_dict


def attempt_to_dict(attempt) -> dict:
    return {
        "number": attempt.number,
        "gcode": attempt.gcode,
        "score": attempt.score,
        "objective_value": attempt.objective_value,
        "blocking_count": len(attempt.blocking_findings),
        "findings": [f.to_dict() for f in attempt.findings],
    }


def event_to_dict(event) -> dict:
    d: dict = {"type": event.type}
    if event.attempt is not None:
        d["attempt"] = attempt_to_dict(event.attempt)
    if event.prompt is not None:
        d["prompt"] = event.prompt
    if event.result is not None:
        d["result"] = result_to_dict(event.result)
    if event.payload is not None:
        d["payload"] = event.payload
    return d


def sse_frame(event_dict: dict) -> str:
    return f"event: {event_dict['type']}\ndata: {json.dumps(event_dict)}\n\n"
