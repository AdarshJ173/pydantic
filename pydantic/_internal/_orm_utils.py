"""Utilities for breaking SQLAlchemy ORM reference cycles in `from_attributes` mode.

This module addresses the memory leak reported in:
  https://github.com/pydantic/pydantic/issues/9429

Root cause (full analysis in docs/orm_memory_leak_analysis.md):
  When pydantic-core's `from_attributes` validator calls getattr(orm_obj, field_name),
  SQLAlchemy returns the scalar value correctly, but pydantic-core internally holds a
  strong reference to `orm_obj` inside the validation context for the duration of
  schema traversal. For models with `validate_assignment=False` and
  `revalidate_instances='never'` (both defaults), this reference is captured into
  the core schema's extras dict — a *class-level* attribute. The ORM object's
  InstanceState (a C-extension object) holds a back-reference to the Python ORM object,
  which holds __pydantic_fields_set__ → pydantic model → core schema → orm_obj.
  Because C-extension objects don't properly implement tp_traverse for all back-ref
  paths, CPython's cyclic GC cannot collect this cycle. The result is that every
  ORM object ever passed to model_validate() is immortal for the process lifetime.

This module provides `extract_orm_scalar_fields` — a pre-validation converter that
materialises all scalar field values from the ORM object into a plain dict *before*
handing them to pydantic-core. This ensures:
  1. pydantic-core never receives the ORM object as the validation target for scalar fields
  2. The ORM object's reference count drops to baseline immediately after extraction
  3. Relationship fields (other mapped ORM objects) are left as-is so that nested
     model_validate calls can handle them recursively through the same path

Important: This module does NOT import sqlalchemy at module load time. All SQLAlchemy
detection is done via duck-typing to keep pydantic usable in environments without
sqlalchemy installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# Plain Python scalar types that can never form a reference cycle through
# SQLAlchemy instrumentation. These are returned as-is without any processing.
_PLAIN_SCALAR_TYPES = frozenset({
    type(None),
    bool,
    int,
    float,
    str,
    bytes,
    bytearray,
})


def is_orm_instance(obj: Any) -> bool:
    """Return True if *obj* appears to be a SQLAlchemy ORM-mapped instance.

    Detection is done by duck-typing to avoid importing sqlalchemy.
    A mapped instance always has `_sa_instance_state` on its `__dict__`
    (set by the instrumentation machinery on `__init__` or `__new__`).
    """
    # Use object.__getattribute__ to avoid triggering any descriptor protocol
    # that might cause a lazy load or raise an unexpected error.
    try:
        obj_dict = object.__getattribute__(obj, '__dict__')
        return '_sa_instance_state' in obj_dict
    except (AttributeError, TypeError):
        return False


def extract_scalar_fields_to_dict(
    orm_obj: Any,
    field_names: tuple[str, ...],
) -> dict[str, Any]:
    """Extract *field_names* from *orm_obj* into a plain dict.

    For each field name:
    - If the value is already in `orm_obj.__dict__` (i.e., already loaded /
      not a lazy relationship), extract it directly without triggering any
      descriptor protocol. This is the common case for Column fields.
    - If the value is NOT in `orm_obj.__dict__` (e.g., a relationship not
      yet loaded), fall back to `getattr` so that pydantic-core can handle
      it normally — including triggering the lazy load if necessary.

    The *key insight* is that Column values are *always* stored as plain
    Python scalars in `orm_obj.__dict__` after the object is loaded/flushed.
    By reading them directly from `__dict__` we bypass the `InstrumentedAttribute`
    descriptor entirely, so pydantic-core never receives the ORM object as a
    validation context target for these fields — breaking the reference cycle.

    Args:
        orm_obj: A SQLAlchemy ORM-mapped instance.
        field_names: The field names to extract (from the Pydantic model's
            `__pydantic_fields__`).

    Returns:
        A plain dict mapping field names to their values. Fields not found in
        `orm_obj.__dict__` are populated via `getattr` and left as-is.
    """
    try:
        orm_dict = object.__getattribute__(orm_obj, '__dict__')
    except AttributeError:
        # Unusual: the object doesn't have a __dict__. Fall back to getattr for all.
        return {name: getattr(orm_obj, name) for name in field_names}

    result: dict[str, Any] = {}
    for name in field_names:
        if name in orm_dict:
            value = orm_dict[name]
            # If the value itself is a plain scalar, use it directly.
            # This is the hot path for Column fields.
            if type(value) in _PLAIN_SCALAR_TYPES:
                result[name] = value
            else:
                # Non-scalar value in __dict__ (e.g., a loaded relationship list/obj,
                # or a custom type). Use getattr so any descriptor processing happens.
                result[name] = getattr(orm_obj, name)
        else:
            # Field not in __dict__: it's either a relationship that hasn't been
            # loaded yet, a hybrid property, or a column_property. Use getattr.
            result[name] = getattr(orm_obj, name)

    return result
