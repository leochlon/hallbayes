from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .paths import audit_log_path, ensure_berry_home

_SENSITIVE_KEY_RE = re.compile(r"(api[_-]?key|token|secret|password)", re.IGNORECASE)
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9]{10,}\b")


def _redact_value(v: Any) -> Any:
    if isinstance(v, str):
        v = _OPENAI_KEY_RE.sub("sk-REDACTED", v)
    return v


def redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if _SENSITIVE_KEY_RE.search(str(k)):
                out[str(k)] = "REDACTED"
            else:
                out[str(k)] = redact(v)
        return out
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    return _redact_value(obj)


@dataclass(frozen=True)
class AuditEvent:
    ts: float
    kind: str
    payload: Dict[str, Any]

    def to_json_line(self) -> str:
        return json.dumps(
            {"ts": self.ts, "kind": self.kind, "payload": redact(self.payload)},
            sort_keys=True,
        )


def append_event(kind: str, payload: Dict[str, Any], *, log_path: Optional[Path] = None) -> Path:
    ensure_berry_home()
    p = log_path or audit_log_path()
    ev = AuditEvent(ts=time.time(), kind=kind, payload=payload)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(ev.to_json_line() + "\n")
    return p


def iter_events(path: Optional[Path] = None) -> Iterable[Dict[str, Any]]:
    p = path or audit_log_path()
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
    except FileNotFoundError:
        return


def export_events(out_path: Path, *, log_path: Optional[Path] = None) -> Path:
    events = list(iter_events(log_path))
    out_path.write_text(json.dumps(events, indent=2) + "\n", encoding="utf-8")
    return out_path


def prune_events(*, retention_days: int, log_path: Optional[Path] = None) -> int:
    """Remove audit events older than retention_days (in-place rewrite)."""
    p = log_path or audit_log_path()
    try:
        raw = p.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return 0

    cutoff = time.time() - (float(retention_days) * 86400.0)
    kept: list[str] = []
    removed = 0
    for line in raw:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            # Keep malformed lines for forensics.
            kept.append(line)
            continue
        ts = float(ev.get("ts", 0.0) or 0.0)
        if ts and ts < cutoff:
            removed += 1
            continue
        kept.append(line)

    p.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed
