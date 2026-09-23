"""
Best-effort extraction of a JSON object from a model's free-text response.
Small local models frequently wrap JSON in prose or leave trailing commas;
this is deliberately forgiving rather than a strict parser, since a parse
failure here should be a genuine "malformed output" MAST-style failure, not
an artifact of over-strict parsing.
"""
import json
import re

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_response(text: str):
    if not text:
        return None
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # common small-model slip: trailing commas before a closing brace/bracket
    cleaned = re.sub(r",\s*}", "}", candidate)
    cleaned = re.sub(r",\s*]", "]", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None
