from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence, Set, Tuple


def make_item_tag(item_id: Any, item_idx: int) -> str:
    return str(item_id) if item_id is not None else f"idx_{int(item_idx)}"


def load_completed_outputs(
    data: Sequence[Dict[str, Any]],
    save_intermediate_dir: str,
) -> Tuple[Dict[str, Dict[str, Any]], Set[str]]:
    completed: Dict[str, Dict[str, Any]] = {}
    invalid_tags: Set[str] = set()

    if not save_intermediate_dir:
        return completed, invalid_tags

    for item_idx, item in enumerate(data):
        item_id = item.get("id", None)
        tag = make_item_tag(item_id, item_idx)
        final_path = os.path.join(save_intermediate_dir, tag, "final.json")
        if not os.path.exists(final_path):
            continue

        try:
            with open(final_path, "r", encoding="utf-8") as f:
                out = json.load(f)
        except Exception:
            invalid_tags.add(tag)
            continue

        if not isinstance(out, dict):
            invalid_tags.add(tag)
            continue

        if item_id is not None and out.get("id", None) != item_id:
            invalid_tags.add(tag)
            continue

        completed[tag] = out

    return completed, invalid_tags
