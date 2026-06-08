"""Shared Data Models for ML Framework Snapshot introspection.

Provides the Ghost Protocol (GhostRef, GhostParam) schemas used to communicate
API structures between the ml-framework-snapshots scraper and the ml-switcheroo compiler.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


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
    default: Optional[str] = Field(None, description="Default value as string.")
    annotation: Optional[str] = Field(None, description="Type annotation as string.")
    description: Optional[str] = Field(default=None, description="Description")

    standardized_name: Optional[str] = Field(
        default=None, description="Standardized name"
    )


class GhostRef(BaseModel):
    """Serializable snapshot of a Framework API component."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Short name of the object.")
    api_path: str = Field(description="Fully qualified import path.")
    kind: str = Field(description="One of: 'class', 'function'")
    params: List[GhostParam] = Field(
        default_factory=list, description="List of parameters."
    )
    docstring: Optional[str] = Field(None, description="Extracted docstring.")
    has_varargs: bool = Field(False, description="True if signature accepts *args.")
    schema_version: str = Field("1.2", description="Version of the schema format.")

    is_public: Optional[bool] = Field(default=None, description="Is public")

    aliases: Optional[List[str]] = Field(default_factory=list, description="Aliases")
    returns_type: Optional[str] = Field(default=None, description="Returns type")
    returns_description: Optional[str] = Field(
        default=None, description="Returns description"
    )
    raises: Optional[List[str]] = Field(default_factory=list, description="Raises")
    environment_tags: Optional[List[str]] = Field(
        default_factory=list, description="Environment tags"
    )
    overloads: Optional[List[Any]] = Field(
        default_factory=list, description="Overloads"
    )

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

    api: Optional[str] = Field(default=None)
    args: Optional[Dict[str, Optional[Union[str, float, int]]]] = Field(default=None)
    inject_args: Optional[Dict[str, Any]] = Field(default=None)
    requires_plugin: Optional[str] = Field(default=None)
    transformation_type: Optional[str] = Field(default=None)
    operator: Optional[str] = Field(default=None)
    pack_to_tuple: Optional[str] = Field(default=None)
    macro_template: Optional[str] = Field(default=None)


def migrate_ghost_ref(data: Dict[str, Any]) -> "GhostRef":
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
