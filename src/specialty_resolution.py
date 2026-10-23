"""Resolve source annotations within an explicit, versioned target scope.

Only raw_group is used here. Support filtering must happen AFTER resolution;
otherwise dropping a rare label can silently turn an ambiguous case into truth.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd


def resolve_specialty_groups(data: pd.DataFrame, scope_path: Path) -> tuple:
    contents = Path(scope_path).read_bytes()
    scope = json.loads(contents)
    included = scope["included_labels"]
    excluded = scope["excluded_labels"]
    if len(included) < 2 or len(set(included)) != len(included) or set(included) & set(excluded):
        raise ValueError("Scope requires >=2 distinct included labels, disjoint from excluded labels.")
    unknown = set(data.medical_specialty) - set(included) - set(excluded)
    if unknown:
        raise ValueError(f"Labels absent from specialty scope; review scope explicitly: {sorted(unknown)}")
    records = []
    for key, group in data.groupby("raw_group", sort=True):
        labels = sorted(set(group.medical_specialty))
        targets = sorted(set(labels) & set(included))
        decision = "no_target_label" if not targets else (
            "single_target_label" if len(targets) == 1 else "ambiguous_target_labels")
        candidates = group.loc[group.medical_specialty.isin(targets)].sort_values("source_row")
        records.append({
            "raw_group": key, "decision": decision,
            "original_labels": json.dumps(labels, ensure_ascii=False),
            "target_labels": json.dumps(targets, ensure_ascii=False),
            "target_label_count": len(targets),
            "resolved_label": targets[0] if len(targets) == 1 else "",
            "representative_source_row": int(candidates.iloc[0].source_row) if len(targets) == 1 else None,
            "source_rows": json.dumps(sorted(int(x) for x in group.source_row)),
            "source_record_count": len(group),
            "text_excerpt": str(group.sort_values("source_row").iloc[0].transcription)[:500],
        })
    decisions = pd.DataFrame(records)
    scope["sha256"] = hashlib.sha256(contents).hexdigest()
    return decisions, scope
