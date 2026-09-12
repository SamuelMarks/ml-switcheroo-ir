"""Custom operator schemas for ml_switcheroo_ir."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ml_switcheroo_ir.schema.onnx_registry import OpAttribute, OpSchema


@dataclass
class CustomAttributeSchema:
    """Represents an attribute for a custom operator.

    Attributes:
        name (str): The name of the attribute.
        type (str): The expected Python type (e.g. 'int', 'List[float]').
        required (bool): Whether the attribute is required.
        default (Any): The default value if not provided.
    """

    name: str
    type: str
    required: bool = False
    default: Any = None

    def to_op_attribute(self) -> OpAttribute:
        """Convert to the internal OpAttribute type."""
        return OpAttribute(
            name=self.name, type=self.type, required=self.required, default=self.default
        )


@dataclass
class CustomOpSchema:
    """Represents a custom operator schema.

    Attributes:
        name (str): The name of the operator.
        domain (str): The custom domain (e.g., 'ai.custom').
        attributes (List[CustomAttributeSchema]): The list of attributes.
        inputs (List[str]): List of expected inputs.
        outputs (List[str]): List of expected outputs.
    """

    name: str
    domain: str
    attributes: list[CustomAttributeSchema] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def to_op_schema(self) -> OpSchema:
        """Convert to the internal OpSchema type.

        Note:
            Version is set to 1 by default for custom ops.
        """
        attr_dict = {attr.name: attr.to_op_attribute() for attr in self.attributes}
        return OpSchema(
            name=self.name,
            domain=self.domain,
            version=1,
            attributes=attr_dict,
            inputs=self.inputs,
            outputs=self.outputs,
        )


class Registry:
    """A dynamic registry of operator schemas."""

    def __init__(self, base_registry: dict[str, OpSchema] | None = None) -> None:
        """Initialize the registry.

        Args:
            base_registry (Dict[str, OpSchema], optional): A base registry to copy from.
        """
        self.schemas: dict[str, OpSchema] = {}
        if base_registry is not None:
            self.schemas.update(base_registry)

    def register_custom_op(self, schema: CustomOpSchema) -> None:
        """Register a custom operator schema.

        Args:
            schema (CustomOpSchema): The custom schema to add.
        """
        self.schemas[schema.name] = schema.to_op_schema()

    def load_custom_ops_from_json(self, file_path: str) -> None:
        """Load and register custom operators from a JSON file.

        Args:
            file_path (str): The path to the JSON file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for op_data in data.get("ops", []):
            attrs = []
            for attr_data in op_data.get("attributes", []):
                attrs.append(
                    CustomAttributeSchema(
                        name=attr_data["name"],
                        type=attr_data["type"],
                        required=attr_data.get("required", False),
                        default=attr_data.get("default", None),
                    )
                )

            schema = CustomOpSchema(
                name=op_data["name"],
                domain=op_data.get("domain", "ai.custom"),
                attributes=attrs,
                inputs=op_data.get("inputs", []),
                outputs=op_data.get("outputs", []),
            )
            self.register_custom_op(schema)


RMSNORM_SCHEMA = CustomOpSchema(
    name="RMSNorm",
    domain="ml.switcheroo.custom",
    inputs=["X", "weight"],
    outputs=["Y"],
    attributes=[
        CustomAttributeSchema(
            name="eps",
            type="float",
            required=False,
            default=1e-6,
        )
    ],
)

SWIGLU_SCHEMA = CustomOpSchema(
    name="SwiGLU",
    domain="ml.switcheroo.custom",
    inputs=["X"],
    outputs=["Y"],
    attributes=[
        CustomAttributeSchema(
            name="dim",
            type="int",
            required=False,
            default=-1,
        )
    ],
)

ROPE_SCHEMA = CustomOpSchema(
    name="RoPE",
    domain="ml.switcheroo.custom",
    inputs=["X", "cos", "sin"],
    outputs=["Y"],
    attributes=[
        CustomAttributeSchema(
            name="dim",
            type="int",
            required=False,
            default=-1,
        )
    ],
)

FLASH_ATTENTION_SCHEMA = CustomOpSchema(
    name="FlashAttention",
    domain="ml.switcheroo.custom",
    inputs=["Q", "K", "V"],
    outputs=["Y"],
    attributes=[
        CustomAttributeSchema(
            name="causal",
            type="bool",
            required=False,
            default=False,
        ),
        CustomAttributeSchema(
            name="scale",
            type="float",
            required=False,
            default=None,
        ),
    ],
)

VISION_PATCH_EMBEDDING_SCHEMA = CustomOpSchema(
    name="VisionPatchEmbedding",
    domain="ml.switcheroo.custom",
    inputs=["X", "weight"],
    outputs=["Y"],
    attributes=[
        CustomAttributeSchema(
            name="patch_size",
            type="List[int]",
            required=True,
        ),
        CustomAttributeSchema(
            name="embed_dim",
            type="int",
            required=True,
        ),
    ],
)

SCALED_DOT_PRODUCT_ATTENTION_SCHEMA = CustomOpSchema(
    name="ScaledDotProductAttention",
    domain="ml.switcheroo.custom",
    inputs=["query", "key", "value", "attn_mask"],
    outputs=["output"],
    attributes=[
        CustomAttributeSchema(
            name="scale",
            type="float",
            required=False,
            default=None,
        ),
        CustomAttributeSchema(
            name="dropout_p",
            type="float",
            required=False,
            default=0.0,
        ),
        CustomAttributeSchema(
            name="is_causal",
            type="bool",
            required=False,
            default=False,
        ),
    ],
)

LAYER_NORM_SCHEMA = CustomOpSchema(
    name="LayerNorm",
    domain="ml.switcheroo.custom",
    inputs=["X", "scale", "bias"],
    outputs=["Y"],
    attributes=[
        CustomAttributeSchema(
            name="axis",
            type="int",
            required=False,
            default=-1,
        ),
        CustomAttributeSchema(
            name="epsilon",
            type="float",
            required=False,
            default=1e-5,
        ),
        CustomAttributeSchema(
            name="elementwise_affine",
            type="bool",
            required=False,
            default=True,
        ),
    ],
)

GROUP_NORM_SCHEMA = CustomOpSchema(
    name="GroupNorm",
    domain="ml.switcheroo.custom",
    inputs=["X", "scale", "bias"],
    outputs=["Y"],
    attributes=[
        CustomAttributeSchema(
            name="num_groups",
            type="int",
            required=True,
        ),
        CustomAttributeSchema(
            name="epsilon",
            type="float",
            required=False,
            default=1e-5,
        ),
    ],
)

GELU_SCHEMA = CustomOpSchema(
    name="GELU",
    domain="ml.switcheroo.custom",
    inputs=["X"],
    outputs=["Y"],
    attributes=[
        CustomAttributeSchema(
            name="approximate",
            type="str",
            required=False,
            default="none",
        ),
    ],
)

EMBEDDING_LOOKUP_SCHEMA = CustomOpSchema(
    name="EmbeddingLookup",
    domain="ml.switcheroo.custom",
    inputs=["indices", "weight"],
    outputs=["output"],
    attributes=[
        CustomAttributeSchema(
            name="padding_idx",
            type="int",
            required=False,
            default=None,
        ),
        CustomAttributeSchema(
            name="scale_grad_by_freq",
            type="bool",
            required=False,
            default=False,
        ),
    ],
)

RMSNORM_BACKWARD_SCHEMA = CustomOpSchema(
    name="RMSNormBackward",
    domain="ml.switcheroo.custom",
    inputs=["grad_output", "X", "weight"],
    outputs=["grad_X", "grad_weight"],
    attributes=[
        CustomAttributeSchema(
            name="eps",
            type="float",
            required=False,
            default=1e-6,
        ),
    ],
)

CUSTOM_OPS_REGISTRY: dict[str, OpSchema] = {
    RMSNORM_SCHEMA.name: RMSNORM_SCHEMA.to_op_schema(),
    SWIGLU_SCHEMA.name: SWIGLU_SCHEMA.to_op_schema(),
    ROPE_SCHEMA.name: ROPE_SCHEMA.to_op_schema(),
    FLASH_ATTENTION_SCHEMA.name: FLASH_ATTENTION_SCHEMA.to_op_schema(),
    VISION_PATCH_EMBEDDING_SCHEMA.name: VISION_PATCH_EMBEDDING_SCHEMA.to_op_schema(),
    SCALED_DOT_PRODUCT_ATTENTION_SCHEMA.name: SCALED_DOT_PRODUCT_ATTENTION_SCHEMA.to_op_schema(),
    LAYER_NORM_SCHEMA.name: LAYER_NORM_SCHEMA.to_op_schema(),
    GROUP_NORM_SCHEMA.name: GROUP_NORM_SCHEMA.to_op_schema(),
    GELU_SCHEMA.name: GELU_SCHEMA.to_op_schema(),
    EMBEDDING_LOOKUP_SCHEMA.name: EMBEDDING_LOOKUP_SCHEMA.to_op_schema(),
    RMSNORM_BACKWARD_SCHEMA.name: RMSNORM_BACKWARD_SCHEMA.to_op_schema(),
}
