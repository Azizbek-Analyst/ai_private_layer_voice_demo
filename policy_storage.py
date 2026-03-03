"""Utilities to persist insurance policy update requests to JSON."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from config import POLICY_FILE, DATA_DIR


def _ensure_storage_file() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not POLICY_FILE.exists():
        POLICY_FILE.write_text("[]", encoding="utf-8")
    return POLICY_FILE


@dataclass
class PolicyUpdate:
    policy_number: str
    old_name: str
    new_name: str
    date_of_changes: str
    phone_number: str
    details: str = ""
    raw: str = ""
    log_filename: str = ""


def save_policy_update(update: PolicyUpdate) -> str:
    """Append a policy update to the JSON file and return the file path."""
    path = _ensure_storage_file()
    try:
        existing: List[Dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        existing = []

    record = asdict(update)
    record["id"] = len(existing) + 1
    record["created_at"] = datetime.utcnow().isoformat() + "Z"
    existing.append(record)

    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def all_policy_updates() -> List[Dict[str, Any]]:
    """Return the list of stored policy updates."""
    path = _ensure_storage_file()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
