"""Getting reliably-shaped data out of a language model.

Three mechanisms, tried in order of strength:

1. **Constrained decoding.** If the server supports ``json_schema`` response
   format (llama.cpp's GBNF grammars do, and so does OpenAI), the output cannot
   be malformed. Preferred whenever available.
2. **Forced tool call.** Anthropic has no response format, but forcing a
   single-tool call with the schema as its input schema is equivalent.
3. **Extract and repair.** For anything else: pull the JSON out of whatever the
   model wrote, validate it, and on failure send the *specific* validation
   errors back for one focused repair turn.

The repair prompt matters more than it looks. Sending "that was invalid, try
again" wastes a full generation; sending "``$.files[2].path``: missing required
property" gets a correct answer in a fraction of the tokens because the model
does not have to re-derive the whole structure.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..errors import MalformedOutput
from ..util.jsonschema import describe, validate

_FENCE = re.compile(r"```(?:json|JSON)?\s*\n(.*?)\n?```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Recover a JSON value from free-form model output.

    Handles, in order: clean JSON, fenced code blocks, and a brace/bracket scan
    that finds the largest balanced structure. The scan is what saves the day
    when a model prefixes "Here is the plan:" despite instructions.
    """
    text = (text or "").strip()
    if not text:
        raise MalformedOutput("model returned empty output")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for block in _FENCE.findall(text):
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue

    candidate = _largest_balanced(text)
    if candidate is not None:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = _repair_common(candidate)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    raise MalformedOutput("no JSON value found in model output", preview=text[:300])


def _largest_balanced(text: str) -> str | None:
    """Find the longest balanced ``{...}`` or ``[...]`` span, string-aware."""
    best: str | None = None
    for opener, closer in (("{", "}"), ("[", "]")):
        depth = 0
        start = -1
        in_string = False
        escaped = False
        for i, ch in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == closer and depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    span = text[start : i + 1]
                    if best is None or len(span) > len(best):
                        best = span
    return best


def _repair_common(text: str) -> str:
    """Fix the two malformations small models actually produce.

    Trailing commas and Python literals (``True``/``None``). Anything more
    exotic is left to the model's repair turn -- guessing at intent in a parser
    is how you get silently wrong data.
    """
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    return text


def schema_instruction(schema: dict[str, Any], *, strict: bool = True) -> str:
    """The prompt fragment that describes the required output shape."""
    outline = describe(schema)
    lines = [
        "Respond with a single JSON value and nothing else.",
        "No prose before or after. No markdown fence.",
        "It must conform to this shape (`?` marks optional fields):",
        outline,
    ]
    if strict:
        lines.append("Do not invent fields that are not listed.")
    return "\n".join(lines)


def repair_instruction(errors: list[str], schema: dict[str, Any]) -> str:
    """Ask for a corrected value, naming exactly what was wrong."""
    listed = "\n".join(f"- {e}" for e in errors[:10])
    return (
        "Your previous output did not satisfy the required schema.\n"
        f"Validation errors:\n{listed}\n\n"
        "Return the corrected JSON value only. Keep everything that was already "
        "correct; change only what the errors identify.\n\n"
        f"Required shape:\n{describe(schema)}"
    )


def parse_and_validate(text: str, schema: dict[str, Any]) -> tuple[Any, list[str]]:
    """Parse then validate. Returns ``(value, errors)``; errors empty on success."""
    value = extract_json(text)
    value = _coerce(value, schema)
    return value, validate(value, schema)


def _permits_null(schema: dict[str, Any] | None) -> bool:
    """Does this schema genuinely accept null, rather than merely tolerate it?"""
    if not isinstance(schema, dict):
        return False
    declared = schema.get("type")
    if declared is None:
        return True  # untyped: not ours to second-guess
    return "null" in ([declared] if isinstance(declared, str) else list(declared))


def _coerce(value: Any, schema: dict[str, Any]) -> Any:
    """Nudge near-miss shapes into the schema without changing meaning.

    Two cases are worth handling because they are common and unambiguous: a bare
    object where a single-element array is required, and a numeric string where
    a number is required. Everything else is reported as an error, because
    silently reinterpreting model output is how wrong data enters the ledger and
    stops being questioned.
    """
    expected = schema.get("type")
    if isinstance(expected, list):
        expected = next((t for t in expected if t != "null"), None)

    if expected == "array" and isinstance(value, dict):
        return [_coerce(value, schema.get("items", {}))]
    if expected in ("number", "integer") and isinstance(value, str):
        try:
            return int(value) if expected == "integer" else float(value)
        except ValueError:
            return value
    if expected == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    if expected == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        out: dict[str, Any] = {}
        for key, item in value.items():
            sub = properties.get(key)
            # An explicit null for an optional field means "not applicable".
            # JSON has no `undefined`, so models write null where they mean
            # absent -- and OpenAI's strict mode *requires* them to, since
            # optionality there is spelled as a nullable type. Rejecting it
            # failed a node 70 times over an edit that was perfectly well
            # formed. Dropping the key is exactly what the model meant.
            if item is None and key not in required and not _permits_null(sub):
                continue
            out[key] = _coerce(item, sub) if sub is not None else item
        return out
    if expected == "array" and isinstance(value, list):
        item_schema = schema.get("items", {})
        return [_coerce(v, item_schema) for v in value]
    return value


def object_schema(
    properties: dict[str, Any],
    *,
    required: list[str] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Build an object schema without hand-writing the boilerplate."""
    return {
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties),
        "additionalProperties": not strict,
    }


def string(description: str = "", **kwargs: Any) -> dict[str, Any]:
    return {"type": "string", "description": description, **kwargs}


def integer(description: str = "", **kwargs: Any) -> dict[str, Any]:
    return {"type": "integer", "description": description, **kwargs}


def number(description: str = "", **kwargs: Any) -> dict[str, Any]:
    return {"type": "number", "description": description, **kwargs}


def boolean(description: str = "") -> dict[str, Any]:
    return {"type": "boolean", "description": description}


def array(items: dict[str, Any], description: str = "", **kwargs: Any) -> dict[str, Any]:
    return {"type": "array", "items": items, "description": description, **kwargs}


def enum(values: list[str], description: str = "") -> dict[str, Any]:
    return {"type": "string", "enum": values, "description": description}


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a schema into OpenAI's strict structured-output subset.

    OpenAI's strict mode is narrower than JSON Schema in one specific way: every
    key in ``properties`` must also appear in ``required``. Optionality is not
    expressed by omission from ``required`` but by admitting ``null`` as a type.
    Sending a perfectly valid schema that omits an optional key is rejected with
    ``invalid_json_schema``, and no retry can help.

    Observed live: ``EDIT_PLAN_SCHEMA`` marks only ``path`` and ``op`` required
    because a ``delete`` edit carries no ``content``. Codex rejected every such
    request with a 400, which Forge classified as a transient provider failure
    and retried indefinitely.

    The transform is applied at the codex boundary only. Forge's own validator
    keeps the original schema, so ``content`` stays genuinely optional
    everywhere else -- widening the real contract to satisfy one vendor would
    lose information the rest of the system relies on.
    """
    if not isinstance(schema, dict):
        return schema

    result = dict(schema)

    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(result.get(key), list):
            result[key] = [strict_schema(s) for s in result[key]]
    if isinstance(result.get("items"), dict):
        result["items"] = strict_schema(result["items"])
    if isinstance(result.get("$defs"), dict):
        result["$defs"] = {k: strict_schema(v) for k, v in result["$defs"].items()}

    properties = result.get("properties")
    if isinstance(properties, dict):
        required = list(result.get("required", []))
        optional = [name for name in properties if name not in required]
        result["properties"] = {
            name: _nullable(strict_schema(sub)) if name in optional else strict_schema(sub)
            for name, sub in properties.items()
        }
        # Order is required-first so the emitted schema still reads as intended.
        result["required"] = required + optional
    return result


def _nullable(sub: dict[str, Any]) -> dict[str, Any]:
    """Admit ``null`` as a value, which is how strict mode spells 'optional'."""
    if not isinstance(sub, dict):
        return sub
    declared = sub.get("type")
    if declared is None:
        return sub  # enum/const/$ref: leave alone rather than guess
    types = [declared] if isinstance(declared, str) else list(declared)
    if "null" not in types:
        types.append("null")
    return {**sub, "type": types}
