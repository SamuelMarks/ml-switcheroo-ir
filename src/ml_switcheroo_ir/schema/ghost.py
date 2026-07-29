"""Shared Data Models for ML Framework Snapshot introspection.

Provides the Ghost Protocol (GhostRef, GhostParam) schemas used to communicate
API structures between the ml-framework-snapshots scraper and the ml-switcheroo compiler.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ParameterKind(str, Enum):
    """Standardized enumeration for parameter kinds."""

    POSITIONAL_ONLY = "POSITIONAL_ONLY"
    POSITIONAL_OR_KEYWORD = "POSITIONAL_OR_KEYWORD"
    VAR_POSITIONAL = "VAR_POSITIONAL"
    KEYWORD_ONLY = "KEYWORD_ONLY"
    VAR_KEYWORD = "VAR_KEYWORD"


class SemanticTier(str, Enum):
    """Categorization of API operations to distinct knowledge base tiers."""

    ARRAY_API = "array"
    NEURAL = "neural"
    NEURAL_OPS = "neural_ops"
    EXTRAS = "extras"
    LOSS = "loss"
    OPTIMIZER = "optimizer"
    LAYER = "layer"
    ACTIVATION = "activation"
    METRIC = "metric"
    UTIL = "util"
    SCHEDULER = "scheduler"
    MODEL = "model"
    INITIALIZER = "initializer"
    DATALOADER = "dataloader"


class GhostParam(BaseModel):
    """Serializable representation of a function parameter."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Parameter name.")
    kind: ParameterKind = Field(
        description="Kind of parameter (e.g. POSITIONAL_OR_KEYWORD)."
    )
    default: str | None = Field(None, description="Default value as string.")
    annotation: str | None = Field(None, description="Type annotation as string.")
    description: str | None = Field(default=None, description="Description")

    standardized_name: str | None = Field(default=None, description="Standardized name")


class GhostRef(BaseModel):
    """Serializable snapshot of a Framework API component."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Short name of the object.")
    api_path: str = Field(description="Fully qualified import path.")
    kind: str = Field(description="One of: 'class', 'function'")
    params: list[GhostParam] = Field(
        default_factory=list, description="List of parameters."
    )
    docstring: str | None = Field(None, description="Extracted docstring.")
    has_varargs: bool = Field(False, description="True if signature accepts *args.")
    schema_version: str = Field("1.2", description="Version of the schema format.")

    is_public: bool | None = Field(default=None, description="Is public")

    aliases: list[str] | None = Field(default_factory=list, description="Aliases")
    returns_type: str | None = Field(default=None, description="Returns type")
    returns_description: str | None = Field(
        default=None, description="Returns description"
    )
    raises: list[str] | None = Field(default_factory=list, description="Raises")
    environment_tags: list[str] | None = Field(
        default_factory=list, description="Environment tags"
    )
    overloads: list[Any] | None = Field(default_factory=list, description="Overloads")

    def has_arg(self, arg_name: str) -> bool:
        """Check if a specific argument exists in the signature.

        Args:
            arg_name: The argument name to find.

        Returns:
            True if found.
        """
        return any(p.name == arg_name for p in self.params)


class LogicOp(str, Enum):
    """Supported operators for conditional logic rules in operations."""

    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    LT = "lt"
    GTE = "gte"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    IS_TYPE = "is_type"


class StandardMap(BaseModel):
    """Defines how a Framework implements a Middle Layer standard."""

    model_config = ConfigDict(extra="ignore")

    api: str | None = Field(default=None)
    args: dict[str, str | float | int | None] | None = Field(default=None)
    inject_args: dict[str, Any] | None = Field(default=None)
    requires_plugin: str | None = Field(default=None)
    transformation_type: str | None = Field(default=None)
    operator: str | None = Field(default=None)
    pack_to_tuple: str | None = Field(default=None)
    macro_template: str | None = Field(default=None)


def migrate_ghost_ref(data: dict[str, Any]) -> GhostRef:
    """Migrates a v1.x JSON dict to a v2.x compatible GhostRef instance.

    Args:
        data: The dictionary representation of a GhostRef, possibly from an older schema.

    Returns:
        A valid GhostRef object.
    """
    if "schema_version" not in data:
        data["schema_version"] = "1.2"

    # Ensure parameter kinds map to valid ParameterKind enum values
    if "params" in data and isinstance(data["params"], list):
        for param in data["params"]:
            if "kind" in param and isinstance(param["kind"], str):
                # Standardize inspect._ParameterKind strings to our enum values
                # e.g., 'POSITIONAL_OR_KEYWORD' or '_ParameterKind.POSITIONAL_OR_KEYWORD'
                kind_str = param["kind"].split(".")[-1]
                param["kind"] = kind_str

    return GhostRef.model_validate(data)
