"""
Hallucination checker for the Trip Digest step.

Because Checkout writes a complete, structured `final_record` to the
blackboard before Trip Digest ever runs, we have exact ground truth for
every dollar amount and confirmation number that should appear in the
digest. This lets us detect hallucination by diffing extracted claims
against that record -- no LLM judge needed for this check.

Two kinds of issue are flagged:
  - unsupported_amount: a dollar figure in the digest that doesn't match
    any true amount in the record (fabrication or distortion).
  - unsupported_confirmation: a confirmation-number-shaped token in the
    digest that isn't one of the real confirmation numbers on file
    (a classic small-model habit: inventing a plausible-looking ID).
"""
import re
from typing import Any, Dict, List


AMOUNT_RE = re.compile(r"\$\s?([\d,]+(?:\.\d{1,2})?)")
CONFIRMATION_RE = re.compile(r"\b([A-Z]{2,5}-[A-Z0-9]{4,8})\b")


def _true_amounts(final_record: Dict[str, Any]) -> set:
    amounts = set()
    for key in ("flight_booking", "hotel_booking", "activity_booking"):
        booking = final_record.get(key) or {}
        if isinstance(booking, dict) and "price" in booking:
            amounts.add(round(float(booking["price"]), 2))
    for key in ("total_cost", "budget_cap", "budget_remaining"):
        if key in final_record and final_record[key] is not None:
            try:
                amounts.add(round(float(final_record[key]), 2))
            except (TypeError, ValueError):
                pass
    return amounts


def _true_confirmations(final_record: Dict[str, Any]) -> set:
    confs = set()
    for key in ("flight_booking", "hotel_booking", "activity_booking"):
        booking = final_record.get(key) or {}
        if isinstance(booking, dict) and booking.get("confirmation_number"):
            confs.add(booking["confirmation_number"])
    return confs


def check_hallucination(final_record: Dict[str, Any], digest_text: str) -> Dict[str, Any]:
    """
    Returns a dict:
      {
        "hallucination_detected": bool,
        "unsupported_amounts": [...],
        "unsupported_confirmations": [...],
      }
    """
    true_amounts = _true_amounts(final_record)
    true_confs = _true_confirmations(final_record)

    found_amounts = set()
    for m in AMOUNT_RE.finditer(digest_text or ""):
        try:
            found_amounts.add(round(float(m.group(1).replace(",", "")), 2))
        except ValueError:
            continue

    found_confs = set(CONFIRMATION_RE.findall(digest_text or ""))

    # allow a tiny tolerance for rounding drift in prose
    unsupported_amounts: List[float] = [
        a for a in found_amounts
        if not any(abs(a - t) < 0.5 for t in true_amounts)
    ]
    unsupported_confirmations: List[str] = [c for c in found_confs if c not in true_confs]

    return {
        "hallucination_detected": bool(unsupported_amounts or unsupported_confirmations),
        "unsupported_amounts": unsupported_amounts,
        "unsupported_confirmations": unsupported_confirmations,
        "true_amounts": sorted(true_amounts),
        "true_confirmations": sorted(true_confs),
    }
