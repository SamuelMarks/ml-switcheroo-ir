"""Custom, collective, quantization, and state operator schemas for ml_switcheroo_ir."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from ml_switcheroo_ir.schema.base import (
    REGISTRY_LOCK,
    AttributeType,
    OpAttribute,
    SchemaRegistry,
)
from ml_switcheroo_ir.schema.base import (
    CustomAttributeSchema as CustomAttributeSchema,
)
from ml_switcheroo_ir.schema.base import (
    CustomOpSchema as CustomOpSchema,
)
from ml_switcheroo_ir.schema.base import (
    OpSchema as OpSchema,
)
from ml_switcheroo_ir.schema.base import (
    load_op_schema_catalog as load_op_schema_catalog,
)
from ml_switcheroo_ir.schema.base import (
    parse_op_schema_entry as parse_op_schema_entry,
)


class Registry:
    """Dynamic operator schema registry supporting custom JSON operator definitions.

    Attributes:
        schemas (dict[str, OpSchema]): Registered canonical operator schemas.
    """

    schemas: dict[str, OpSchema]

    def __init__(self, base_registry: Mapping[str, OpSchema] | None = None) -> None:
        """Initialize the registry.

        Args:
            base_registry (Mapping[str, OpSchema] | None): Optional base registry to copy from.
        """
        self.schemas = {}
        if base_registry is not None:
            self.schemas.update(base_registry)

    def register(self, schema: OpSchema) -> None:
        """Register an operator schema under its canonical name.

        Args:
            schema (OpSchema): The operator schema to register.
        """
        self.schemas[schema.name] = schema.to_op_schema()

    def register_custom_op(self, schema: OpSchema) -> None:
        """Register a custom operator schema.

        Args:
            schema (OpSchema): The custom operator schema to add.
        """
        self.schemas[schema.name] = schema.to_op_schema()

    def load_custom_ops_from_json(self, file_path: str) -> None:
        """Load and register custom operators from a JSON file.

        Args:
            file_path (str): Path to the JSON file containing an ``'ops'`` list.
        """
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        for op_data in data.get("ops", []):
            self.register(
                parse_op_schema_entry(
                    op_data,
                    domain_default="ai.custom",
                    attr_type_default=AttributeType.ANY,
                )
            )


RMSNORM_SCHEMA = OpSchema(
    name="RMSNorm",
    domain="ml.switcheroo.custom",
    inputs=["X", "weight"],
    outputs=["Y"],
    attributes=[
        OpAttribute(name="eps", type=AttributeType.FLOAT, required=False, default=1e-6),
    ],
)

SWIGLU_SCHEMA = OpSchema(
    name="SwiGLU",
    domain="ml.switcheroo.custom",
    inputs=["X"],
    outputs=["Y"],
    attributes=[
        OpAttribute(name="dim", type=AttributeType.INT, required=False, default=-1),
    ],
)

ROPE_SCHEMA = OpSchema(
    name="RoPE",
    domain="ml.switcheroo.custom",
    inputs=["X", "cos", "sin"],
    outputs=["Y"],
    attributes=[
        OpAttribute(name="dim", type=AttributeType.INT, required=False, default=-1),
    ],
)

FLASH_ATTENTION_SCHEMA = OpSchema(
    name="FlashAttention",
    domain="ml.switcheroo.custom",
    inputs=["Q", "K", "V"],
    outputs=["Y"],
    attributes=[
        OpAttribute(name="causal", type=AttributeType.BOOL, required=False, default=False),
        OpAttribute(name="scale", type=AttributeType.FLOAT, required=False, default=None),
    ],
    aliases=("flash_attention",),
)

VISION_PATCH_EMBEDDING_SCHEMA = OpSchema(
    name="VisionPatchEmbedding",
    domain="ml.switcheroo.custom",
    inputs=["X", "weight"],
    outputs=["Y"],
    attributes=[
        OpAttribute(name="patch_size", type=AttributeType.LIST_INT, required=True),
        OpAttribute(name="embed_dim", type=AttributeType.INT, required=True),
    ],
)

SCALED_DOT_PRODUCT_ATTENTION_SCHEMA = OpSchema(
    name="ScaledDotProductAttention",
    domain="ml.switcheroo.custom",
    inputs=["query", "key", "value", "attn_mask"],
    outputs=["output"],
    attributes=[
        OpAttribute(name="scale", type=AttributeType.FLOAT, required=False, default=None),
        OpAttribute(name="dropout_p", type=AttributeType.FLOAT, required=False, default=0.0),
        OpAttribute(name="is_causal", type=AttributeType.BOOL, required=False, default=False),
    ],
    aliases=("scaled_dot_product_attention",),
)

PAGED_ATTENTION_SCHEMA = OpSchema(
    name="PagedAttention",
    domain="ml.switcheroo.custom",
    inputs=["query", "key_cache", "value_cache", "block_tables", "context_lens"],
    outputs=["out"],
    attributes=[
        OpAttribute(name="num_kv_heads", type=AttributeType.INT, required=False, default=None),
        OpAttribute(name="scale", type=AttributeType.FLOAT, required=False, default=None),
        OpAttribute(name="block_size", type=AttributeType.INT, required=False, default=16),
        OpAttribute(name="max_context_len", type=AttributeType.INT, required=False, default=None),
        OpAttribute(name="alibi_slopes", type=AttributeType.ANY, required=False, default=None),
        OpAttribute(name="kv_cache_dtype", type=AttributeType.STR, required=False, default="auto"),
    ],
    aliases=("paged_attention",),
)

RAGGED_PAGED_ATTENTION_SCHEMA = OpSchema(
    name="RaggedPagedAttention",
    domain="ml.switcheroo.custom",
    inputs=["query", "key_cache", "value_cache", "block_tables", "context_lens"],
    outputs=["out"],
    attributes=[
        OpAttribute(name="block_size", type=AttributeType.INT, required=False, default=16),
    ],
    aliases=("ragged_paged_attention",),
)

CANONICAL_ATTENTION_SCHEMAS: tuple[OpSchema, ...] = (
    FLASH_ATTENTION_SCHEMA,
    PAGED_ATTENTION_SCHEMA,
    RAGGED_PAGED_ATTENTION_SCHEMA,
)

LAYER_NORM_SCHEMA = OpSchema(
    name="LayerNorm",
    domain="ml.switcheroo.custom",
    inputs=["X", "scale", "bias"],
    outputs=["Y"],
    attributes=[
        OpAttribute(name="axis", type=AttributeType.INT, required=False, default=-1),
        OpAttribute(name="epsilon", type=AttributeType.FLOAT, required=False, default=1e-5),
        OpAttribute(name="elementwise_affine", type=AttributeType.BOOL, required=False, default=True),
    ],
)

GROUP_NORM_SCHEMA = OpSchema(
    name="GroupNorm",
    domain="ml.switcheroo.custom",
    inputs=["X", "scale", "bias"],
    outputs=["Y"],
    attributes=[
        OpAttribute(name="num_groups", type=AttributeType.INT, required=True),
        OpAttribute(name="epsilon", type=AttributeType.FLOAT, required=False, default=1e-5),
    ],
)

GELU_SCHEMA = OpSchema(
    name="GELU",
    domain="ml.switcheroo.custom",
    inputs=["X"],
    outputs=["Y"],
    attributes=[
        OpAttribute(name="approximate", type=AttributeType.STR, required=False, default="none"),
    ],
)

EMBEDDING_LOOKUP_SCHEMA = OpSchema(
    name="EmbeddingLookup",
    domain="ml.switcheroo.custom",
    inputs=["indices", "weight"],
    outputs=["output"],
    attributes=[
        OpAttribute(name="padding_idx", type=AttributeType.INT, required=False, default=None),
        OpAttribute(name="scale_grad_by_freq", type=AttributeType.BOOL, required=False, default=False),
    ],
)

RMSNORM_BACKWARD_SCHEMA = OpSchema(
    name="RMSNormBackward",
    domain="ml.switcheroo.custom",
    inputs=["grad_output", "X", "weight"],
    outputs=["grad_X", "grad_weight"],
    attributes=[
        OpAttribute(name="eps", type=AttributeType.FLOAT, required=False, default=1e-6),
    ],
)

COLLECTIVE_ALL_REDUCE_SCHEMA = OpSchema(
    name="collective.all_reduce",
    domain="collective",
    inputs=["input"],
    outputs=["output"],
    attributes=[
        OpAttribute(name="reduction_op", type=AttributeType.STR, required=True),
        OpAttribute(name="mesh_axis", type=AttributeType.STR, required=True),
        OpAttribute(name="channel_id", type=AttributeType.INT, required=False, default=0),
        OpAttribute(name="comm", type=AttributeType.STR, required=False, default=None),
        OpAttribute(name="stream", type=AttributeType.INT, required=False, default=0),
        OpAttribute(name="datatype", type=AttributeType.STR, required=False, default=None),
    ],
    aliases=("all_reduce",),
)

COLLECTIVE_ALL_GATHER_SCHEMA = OpSchema(
    name="collective.all_gather",
    domain="collective",
    inputs=["input"],
    outputs=["output"],
    attributes=[
        OpAttribute(name="gather_dimension", type=AttributeType.INT, required=True),
        OpAttribute(name="mesh_axis", type=AttributeType.STR, required=True),
        OpAttribute(name="comm", type=AttributeType.STR, required=False, default=None),
        OpAttribute(name="stream", type=AttributeType.INT, required=False, default=0),
        OpAttribute(name="datatype", type=AttributeType.STR, required=False, default=None),
    ],
    aliases=("all_gather",),
)

COLLECTIVE_REDUCE_SCATTER_SCHEMA = OpSchema(
    name="collective.reduce_scatter",
    domain="collective",
    inputs=["input"],
    outputs=["output"],
    attributes=[
        OpAttribute(name="scatter_dimension", type=AttributeType.INT, required=True),
        OpAttribute(name="reduction_op", type=AttributeType.STR, required=True),
        OpAttribute(name="mesh_axis", type=AttributeType.STR, required=True),
        OpAttribute(name="comm", type=AttributeType.STR, required=False, default=None),
        OpAttribute(name="stream", type=AttributeType.INT, required=False, default=0),
        OpAttribute(name="datatype", type=AttributeType.STR, required=False, default=None),
    ],
    aliases=("reduce_scatter",),
)

COLLECTIVE_ALL_TO_ALL_SCHEMA = OpSchema(
    name="collective.all_to_all",
    domain="collective",
    inputs=["input"],
    outputs=["output"],
    attributes=[
        OpAttribute(name="split_dimension", type=AttributeType.INT, required=True),
        OpAttribute(name="concat_dimension", type=AttributeType.INT, required=True),
        OpAttribute(name="mesh_axis", type=AttributeType.STR, required=True),
        OpAttribute(name="comm", type=AttributeType.STR, required=False, default=None),
        OpAttribute(name="stream", type=AttributeType.INT, required=False, default=0),
        OpAttribute(name="datatype", type=AttributeType.STR, required=False, default=None),
    ],
    aliases=("all_to_all",),
)

_CANONICAL_COLLECTIVE_OPS: tuple[OpSchema, ...] = (
    COLLECTIVE_ALL_REDUCE_SCHEMA,
    COLLECTIVE_ALL_GATHER_SCHEMA,
    COLLECTIVE_REDUCE_SCATTER_SCHEMA,
    COLLECTIVE_ALL_TO_ALL_SCHEMA,
)

QUANTIZATION_BLOCK_QUANTIZE_SCHEMA = OpSchema(
    name="quantization.block_quantize",
    domain="quantization",
    inputs=["input"],
    outputs=["output", "scale"],
    attributes=[
        OpAttribute(name="block_size", type=AttributeType.INT, required=True),
        OpAttribute(name="quant_dtype", type=AttributeType.STR, required=True),
        OpAttribute(name="axis", type=AttributeType.INT, required=False, default=-1),
    ],
    aliases=("block_quantize",),
)

QUANTIZATION_DEQUANTIZE_GROUPED_INT4_SCHEMA = OpSchema(
    name="quantization.dequantize_grouped_int4",
    domain="quantization",
    inputs=["weight", "scales", "qzeros"],
    outputs=["output"],
    attributes=[
        OpAttribute(name="group_size", type=AttributeType.INT, required=True),
        OpAttribute(name="packing_format", type=AttributeType.STR, required=True),
        OpAttribute(name="symmetric", type=AttributeType.BOOL, required=False, default=False),
        OpAttribute(name="axis", type=AttributeType.INT, required=False, default=0),
        OpAttribute(name="scales_shape", type=AttributeType.LIST_INT, required=False),
    ],
    aliases=("dequantize_grouped_int4",),
)

_CANONICAL_QUANTIZATION_OPS: tuple[OpSchema, ...] = (
    QUANTIZATION_BLOCK_QUANTIZE_SCHEMA,
    QUANTIZATION_DEQUANTIZE_GROUPED_INT4_SCHEMA,
)

_CANONICAL_CUSTOM_OPS: tuple[OpSchema, ...] = (
    RMSNORM_SCHEMA,
    SWIGLU_SCHEMA,
    ROPE_SCHEMA,
    FLASH_ATTENTION_SCHEMA,
    PAGED_ATTENTION_SCHEMA,
    RAGGED_PAGED_ATTENTION_SCHEMA,
    VISION_PATCH_EMBEDDING_SCHEMA,
    SCALED_DOT_PRODUCT_ATTENTION_SCHEMA,
    LAYER_NORM_SCHEMA,
    GROUP_NORM_SCHEMA,
    GELU_SCHEMA,
    EMBEDDING_LOOKUP_SCHEMA,
    RMSNORM_BACKWARD_SCHEMA,
    *_CANONICAL_COLLECTIVE_OPS,
    *_CANONICAL_QUANTIZATION_OPS,
)

_CANONICAL_STATE_OPS: tuple[OpSchema, ...] = (
    OpSchema(
        name="ReadVariable",
        domain="ml.switcheroo.state",
        attributes=[
            OpAttribute(name="variable_name", type=AttributeType.STR, required=True),
            OpAttribute(name="dtype", type=AttributeType.STR, required=False),
        ],
        inputs=[],
        outputs=["w_val"],
    ),
    OpSchema(
        name="AssignVariable",
        domain="ml.switcheroo.state",
        attributes=[
            OpAttribute(name="variable_name", type=AttributeType.STR, required=True),
        ],
        inputs=["w_val"],
        outputs=["w_updated"],
    ),
    OpSchema(
        name="ScatterUpdate",
        domain="ml.switcheroo.state",
        attributes=[
            OpAttribute(name="variable_name", type=AttributeType.STR, required=False),
            OpAttribute(name="axis", type=AttributeType.INT, required=False, default=0),
        ],
        inputs=["w_val", "indices", "updates"],
        outputs=["w_scattered"],
    ),
)


def _load_collective_schemas(json_path: Path | str | None = None) -> None:
    """Populate ``COLLECTIVE_OPS_REGISTRY`` lazily under ``REGISTRY_LOCK``.

    Args:
        json_path (Path | str | None): Optional JSON catalog path override.
    """
    with REGISTRY_LOCK:
        loaded = load_op_schema_catalog(
            "collective_ops.json",
            domain_default="collective",
            json_path=json_path,
            attr_type_default=AttributeType.ANY,
        )
        if loaded:
            COLLECTIVE_OPS_REGISTRY.update(loaded)
        elif json_path is None:
            for schema in _CANONICAL_COLLECTIVE_OPS:
                COLLECTIVE_OPS_REGISTRY[schema.name] = schema


def _load_quantization_schemas(json_path: Path | str | None = None) -> None:
    """Populate ``QUANTIZATION_OPS_REGISTRY`` lazily under ``REGISTRY_LOCK``.

    Args:
        json_path (Path | str | None): Optional JSON catalog path override.
    """
    with REGISTRY_LOCK:
        loaded = load_op_schema_catalog(
            "quantization_ops.json",
            domain_default="quantization",
            json_path=json_path,
            attr_type_default=AttributeType.ANY,
        )
        if loaded:
            QUANTIZATION_OPS_REGISTRY.update(loaded)
        elif json_path is None:
            for schema in _CANONICAL_QUANTIZATION_OPS:
                QUANTIZATION_OPS_REGISTRY[schema.name] = schema


def _load_custom_schemas(json_path: Path | str | None = None) -> None:
    """Populate ``CUSTOM_OPS_REGISTRY`` lazily under ``REGISTRY_LOCK``.

    Args:
        json_path (Path | str | None): Optional JSON catalog path override.
    """
    with REGISTRY_LOCK:
        loaded = load_op_schema_catalog(
            "custom_ops.json",
            domain_default="ml.switcheroo.custom",
            json_path=json_path,
            attr_type_default=AttributeType.ANY,
        )
        if loaded:
            CUSTOM_OPS_REGISTRY.update(loaded)
        elif json_path is None:
            for schema in _CANONICAL_CUSTOM_OPS:
                CUSTOM_OPS_REGISTRY[schema.name] = schema


def _load_state_schemas(json_path: Path | str | None = None) -> None:
    """Load state mutation schemas from state_ops.json into STATE_OPS_REGISTRY.

    Args:
        json_path (Path | str | None): Explicit path to state_ops.json.
    """
    with REGISTRY_LOCK:
        loaded = load_op_schema_catalog(
            "state_ops.json",
            domain_default="ml.switcheroo.state",
            json_path=json_path,
            attr_type_default=AttributeType.ANY,
        )
        if loaded:
            STATE_OPS_REGISTRY.update(loaded)
        elif json_path is None:
            for schema in _CANONICAL_STATE_OPS:
                STATE_OPS_REGISTRY[schema.name] = schema


COLLECTIVE_OPS_REGISTRY: SchemaRegistry = SchemaRegistry("COLLECTIVE_OPS_REGISTRY", loader=_load_collective_schemas)
QUANTIZATION_OPS_REGISTRY: SchemaRegistry = SchemaRegistry(
    "QUANTIZATION_OPS_REGISTRY", loader=_load_quantization_schemas
)
CUSTOM_OPS_REGISTRY: SchemaRegistry = SchemaRegistry("CUSTOM_OPS_REGISTRY", loader=_load_custom_schemas)
STATE_OPS_REGISTRY: SchemaRegistry = SchemaRegistry("STATE_OPS_REGISTRY", loader=_load_state_schemas)
