from __future__ import annotations

from veritract.types import GroundedField, QuarantinedField


def _score_results(
    extracted: dict[str, GroundedField],
    quarantined: list[QuarantinedField],
    gt: dict[str, str] | None,
) -> float:
    """Return a 0-1 score for one extraction run.

    Supervised (gt provided): fraction of all GT fields whose extracted value
    case-insensitively contains or is contained by the ground-truth value.
    Quarantined fields count as wrong. Missing fields count as wrong.

    Unsupervised (gt=None): fraction of extracted fields that have a span
    (i.e. were grounded). Quarantined fields count as ungrounded.
    """
    all_fields = list(extracted.keys()) + [q["field_name"] for q in quarantined]
    if not all_fields and gt is None:
        return 0.0

    if gt is not None:
        gt_fields = set(gt.keys())
        correct = 0
        total = len(gt_fields)
        for field, gf in extracted.items():
            if field not in gt_fields:
                continue
            v = gf["value"].strip().lower()
            g = gt[field].strip().lower()
            if v and g and (v in g or g in v):
                correct += 1
        return correct / total if total > 0 else 0.0
    else:
        total = len(all_fields)
        grounded = sum(1 for gf in extracted.values() if gf["span"] is not None)
        return grounded / total if total > 0 else 0.0
