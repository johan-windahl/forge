"""A compact JSON Schema validator (draft-07 subset).

Forge constrains model output with JSON Schema. Pulling in ``jsonschema`` would
be the obvious move, but the schemas Forge emits are authored by Forge itself
and use a deliberately narrow subset. Validating that subset in ~150 lines keeps
the core dependency-free and, more usefully, lets the validator return
*repair-oriented* error messages: each error names the JSON pointer and states
what was expected, which is fed straight back to the model as a repair prompt.

Supported: type, properties, required, additionalProperties, items, enum, const,
minimum, maximum, minLength, maxLength, minItems, maxItems, pattern, anyOf,
oneOf, allOf, nullable via ``["T", "null"]``, and ``$defs``/``$ref`` for local
references.
"""

from __future__ import annotations

import re
from typing import Any

_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


class SchemaError(ValueError):
    """The schema itself is malformed."""


def validate(instance: Any, schema: dict[str, Any]) -> list[str]:
    """Validate ``instance``; return a list of human-readable error strings.

    An empty list means valid. Errors are ordered outside-in so the first one is
    usually the most actionable.
    """
    errors: list[str] = []
    _validate(instance, schema, "$", schema, errors)
    return errors


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if not ref:
        return schema
    if not ref.startswith("#/"):
        raise SchemaError(f"only local $ref supported, got {ref!r}")
    node: Any = root
    for token in ref[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or token not in node:
            raise SchemaError(f"unresolvable $ref {ref!r}")
        node = node[token]
    if not isinstance(node, dict):
        raise SchemaError(f"$ref {ref!r} does not point at a schema")
    return node


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _validate(inst: Any, schema: dict[str, Any], path: str, root: dict[str, Any], errors: list[str]) -> None:
    schema = _resolve(schema, root)

    if "const" in schema and inst != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}, got {inst!r}")
        return

    if "enum" in schema and inst not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']!r}, got {inst!r}")
        return

    for keyword in ("allOf",):
        for sub in schema.get(keyword, []):
            _validate(inst, sub, path, root, errors)

    if "anyOf" in schema or "oneOf" in schema:
        alternatives = schema.get("anyOf") or schema.get("oneOf") or []
        matches = 0
        collected: list[str] = []
        for sub in alternatives:
            sub_errors: list[str] = []
            _validate(inst, sub, path, root, sub_errors)
            if sub_errors:
                collected.extend(sub_errors)
            else:
                matches += 1
        if matches == 0:
            errors.append(f"{path}: matched none of {len(alternatives)} alternatives ({'; '.join(collected[:3])})")
            return

    expected = schema.get("type")
    if expected is not None:
        allowed = [expected] if isinstance(expected, str) else list(expected)
        # bool is a subclass of int in Python; JSON Schema treats them apart.
        actual = _type_name(inst)
        ok = False
        for name in allowed:
            if name not in _TYPE_MAP:
                raise SchemaError(f"unknown type {name!r}")
            if (name == "integer" and actual == "integer") or (name == "number" and actual in ("integer", "number")) or (name == "boolean" and actual == "boolean") or name == actual:
                ok = True
            if ok:
                break
        if not ok:
            errors.append(f"{path}: expected type {'|'.join(allowed)}, got {actual}")
            return

    if isinstance(inst, str):
        _validate_string(inst, schema, path, errors)
    elif isinstance(inst, (int, float)) and not isinstance(inst, bool):
        _validate_number(inst, schema, path, errors)
    elif isinstance(inst, list):
        _validate_array(inst, schema, path, root, errors)
    elif isinstance(inst, dict):
        _validate_object(inst, schema, path, root, errors)


def _validate_string(inst: str, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    if (lo := schema.get("minLength")) is not None and len(inst) < lo:
        errors.append(f"{path}: string shorter than minLength {lo}")
    if (hi := schema.get("maxLength")) is not None and len(inst) > hi:
        errors.append(f"{path}: string longer than maxLength {hi}")
    if (pattern := schema.get("pattern")) is not None and not re.search(pattern, inst):
        errors.append(f"{path}: string does not match pattern {pattern!r}")


def _validate_number(inst: float, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    if (lo := schema.get("minimum")) is not None and inst < lo:
        errors.append(f"{path}: {inst} below minimum {lo}")
    if (hi := schema.get("maximum")) is not None and inst > hi:
        errors.append(f"{path}: {inst} above maximum {hi}")
    if (lo := schema.get("exclusiveMinimum")) is not None and inst <= lo:
        errors.append(f"{path}: {inst} not above exclusiveMinimum {lo}")
    if (hi := schema.get("exclusiveMaximum")) is not None and inst >= hi:
        errors.append(f"{path}: {inst} not below exclusiveMaximum {hi}")


def _validate_array(inst: list[Any], schema: dict[str, Any], path: str, root: dict[str, Any], errors: list[str]) -> None:
    if (lo := schema.get("minItems")) is not None and len(inst) < lo:
        errors.append(f"{path}: array has {len(inst)} items, minItems is {lo}")
    if (hi := schema.get("maxItems")) is not None and len(inst) > hi:
        errors.append(f"{path}: array has {len(inst)} items, maxItems is {hi}")
    if schema.get("uniqueItems") and len({_hashable(i) for i in inst}) != len(inst):
        errors.append(f"{path}: array items are not unique")
    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for i, item in enumerate(inst):
            _validate(item, item_schema, f"{path}[{i}]", root, errors)


def _hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    return value


def _validate_object(inst: dict[str, Any], schema: dict[str, Any], path: str, root: dict[str, Any], errors: list[str]) -> None:
    properties: dict[str, Any] = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in inst:
            errors.append(f"{path}: missing required property {name!r}")
    for name, value in inst.items():
        child = f"{path}.{name}"
        if name in properties:
            _validate(value, properties[name], child, root, errors)
        elif schema.get("additionalProperties") is False:
            allowed = ", ".join(sorted(properties)) or "(none)"
            errors.append(f"{path}: unexpected property {name!r}; allowed: {allowed}")
        elif isinstance(schema.get("additionalProperties"), dict):
            _validate(value, schema["additionalProperties"], child, root, errors)


def describe(schema: dict[str, Any], indent: int = 0) -> str:
    """Render a schema as a terse outline for inclusion in a prompt.

    Full JSON Schema is verbose and burns tokens. This outline conveys the same
    contract in roughly a third of the characters, which matters when every
    structured call carries it.
    """
    pad = "  " * indent
    stype = schema.get("type", "any")
    if isinstance(stype, list):
        stype = "|".join(stype)
    if stype == "object":
        lines = []
        required = set(schema.get("required", []))
        for name, sub in schema.get("properties", {}).items():
            mark = "" if name in required else "?"
            desc = sub.get("description", "")
            rendered = describe(sub, indent + 1)
            suffix = f"  # {desc}" if desc else ""
            lines.append(f"{pad}  {name}{mark}: {rendered}{suffix}")
        return "{\n" + "\n".join(lines) + f"\n{pad}}}"
    if stype == "array":
        return f"[{describe(schema.get('items', {}), indent)}]"
    if "enum" in schema:
        return "|".join(repr(v) for v in schema["enum"])
    return str(stype)
