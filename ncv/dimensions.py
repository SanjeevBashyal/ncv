"""
Toolkit-neutral dimension selector helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .ncvutils import DIMMETHODS, selvar, spinbox_values, vardim2var


@dataclass(frozen=True)
class DimensionSpec:
    """State needed by one dimension selector control."""

    label: str
    values: tuple[str, ...]
    value: str
    enabled: bool
    tooltip: str


def empty_dimension_specs(maxdim: int) -> list[DimensionSpec]:
    return [
        DimensionSpec(str(i), ("0",), "0", False, "")
        for i in range(max(maxdim, 1))
    ]


def _as_text_values(values: Sequence[object]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _dim_names(var) -> Sequence[str]:
    if hasattr(var, "dims"):
        return var.dims
    return var.dimensions


def _dim_tooltip(length: int) -> str:
    if length > 1:
        return (
            f"Specific dimension value: 0-{length - 1}\n"
            "or arithmetic operation on axis:\n"
            "  " + ", ".join(DIMMETHODS)
        )
    return "Single dimension: 0"


def _set_spec(specs: list[DimensionSpec], index: int, label: str, length: int,
              value: str) -> None:
    specs[index] = DimensionSpec(
        label=str(label),
        values=_as_text_values(spinbox_values(length)),
        value=str(value),
        enabled=True,
        tooltip=_dim_tooltip(length),
    )


def resolve_selected_variable(owner, vardim: str):
    """Return group index, variable name, and variable object for a combobox item."""
    group, variable_name = vardim2var(vardim, owner.groups)
    if owner.usex:
        if variable_name == owner.tname:
            variable_name = owner.tvar
    else:
        if owner.tname and variable_name == owner.tname[group]:
            variable_name = owner.tvar[group]
    return group, variable_name, selvar(owner, variable_name)


def dimension_specs(owner, vardim: str, role: str) -> list[DimensionSpec]:
    """
    Return dimension selector specs for a selected variable.

    ``role`` is one of ``x``, ``y``, ``y2``, ``z``, ``lat``, ``lon``, or
    ``var`` and preserves the defaults from the original Tk implementation.
    """
    specs = empty_dimension_specs(owner.maxdim)
    if not vardim:
        return specs

    group, _variable_name, variable = resolve_selected_variable(owner, vardim)
    if variable is None:
        return specs

    dims = _dim_names(variable)
    shape = variable.shape
    ndim = variable.ndim

    if owner.usex:
        dunlim = owner.dunlim
        latdim = owner.latdim
        londim = owner.londim
    else:
        dunlim = owner.dunlim[group] if owner.dunlim else ""
        latdim = owner.latdim[group] if owner.latdim else ""
        londim = owner.londim[group] if owner.londim else ""

    if role in {"lat", "lon"}:
        for i in range(ndim):
            default = "all" if shape[i] > 1 else "0"
            _set_spec(specs, i, dims[i], shape[i], default)
        return specs

    if role in {"x", "y", "y2"}:
        nall = 0
        if dunlim in dims:
            i = dims.index(dunlim)
            _set_spec(specs, i, dims[i], shape[i], "all")
            nall = 1
        for i in range(ndim):
            if dims[i] == dunlim:
                continue
            if nall == 0 and shape[i] > 1:
                default = "all"
                nall = 1
            else:
                default = "0"
            _set_spec(specs, i, dims[i], shape[i], default)
        return specs

    if role == "z":
        nall = 0
        if dunlim in dims:
            i = dims.index(dunlim)
            _set_spec(specs, i, dims[i], shape[i], "all")
            nall = 1
        for i in range(ndim):
            if dims[i] == dunlim:
                continue
            if nall <= 1 and shape[i] > 1:
                default = "all"
                nall += 1
            else:
                default = "0"
            _set_spec(specs, i, dims[i], shape[i], default)
        return specs

    if role == "var":
        nall = 0
        for dim_name in (latdim, londim):
            if dim_name and dim_name in dims:
                i = dims.index(dim_name)
                _set_spec(specs, i, dims[i], shape[i], "all")
                nall += 1
        for i in range(ndim):
            if dims[i] in {latdim, londim}:
                continue
            if (dims[i] != dunlim) and (nall <= 1) and shape[i] > 1:
                default = "all"
                nall += 1
            else:
                default = "0"
            _set_spec(specs, i, dims[i], shape[i], default)
        return specs

    raise ValueError(f"Unknown dimension role: {role}")


def dimension_values(selectors) -> list[str]:
    """Return current selector text from Qt combo boxes or compatible objects."""
    out = []
    for selector in selectors:
        if hasattr(selector, "currentText"):
            out.append(selector.currentText())
        else:
            out.append(str(selector.get()))
    return out
