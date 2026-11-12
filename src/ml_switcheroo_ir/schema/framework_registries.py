"""Framework operator registries and universal Operation Definition Language (ODL) catalog."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ml_switcheroo_ir.schema.base import (
    REGISTRY_LOCK,
    AttributeType,
    LazyMetadataMap,
    OpAttribute,
    OpSchema,
    SchemaRegistry,
    load_op_schema_catalog,
)
from ml_switcheroo_ir.snapshots import find_schema_file
from ml_switcheroo_ir.types import JsonValue

_CANONICAL_ODL_OPS: list[OpSchema] = [
    OpSchema(name="add", domain="odl", inputs=["a", "b"], outputs=["result"]),
    OpSchema(name="subtract", domain="odl", inputs=["a", "b"], outputs=["result"]),
    OpSchema(name="multiply", domain="odl", inputs=["a", "b"], outputs=["result"]),
    OpSchema(name="divide", domain="odl", inputs=["a", "b"], outputs=["result"]),
    OpSchema(name="matmul", domain="odl", inputs=["a", "b"], outputs=["result"]),
    OpSchema(name="relu", domain="odl", inputs=["x"], outputs=["result"]),
    OpSchema(name="gelu", domain="odl", inputs=["x"], outputs=["result"]),
    OpSchema(name="silu", domain="odl", inputs=["x"], outputs=["result"]),
    OpSchema(
        name="softmax",
        domain="odl",
        attributes=[OpAttribute(name="dim", type=AttributeType.INT, required=False, default=-1)],
        inputs=["x"],
        outputs=["result"],
    ),
    OpSchema(
        name="layer_norm",
        domain="odl",
        attributes=[OpAttribute(name="eps", type=AttributeType.FLOAT, required=False, default=1e-5)],
        inputs=["x", "scale", "bias"],
        outputs=["result"],
    ),
    OpSchema(
        name="rms_norm",
        domain="odl",
        attributes=[OpAttribute(name="eps", type=AttributeType.FLOAT, required=False, default=1e-6)],
        inputs=["x", "scale"],
        outputs=["result"],
    ),
    OpSchema(name="conv2d", domain="odl", inputs=["x", "weight"], outputs=["result"]),
    OpSchema(
        name="scaled_dot_product_attention",
        domain="odl",
        inputs=["query", "key", "value"],
        outputs=["result"],
    ),
]

_CANONICAL_ODL_DIALECT_MAPPINGS: dict[str, dict[str, str]] = {
    "matmul": {
        "torch": "torch.matmul",
        "jax": "jax.numpy.matmul",
        "tensorflow": "tf.linalg.matmul",
        "keras3": "keras.ops.matmul",
        "mlx": "mlx.core.matmul",
        "onnx": "MatMul",
        "stablehlo": "stablehlo.dot_general",
    },
    "add": {
        "torch": "torch.add",
        "jax": "jax.numpy.add",
        "tensorflow": "tf.math.add",
        "keras3": "keras.ops.add",
        "mlx": "mlx.core.add",
        "onnx": "Add",
        "stablehlo": "stablehlo.add",
    },
}

_CANONICAL_ATEN_OPS: list[OpSchema] = [
    OpSchema(
        name="add",
        domain="aten",
        attributes=[OpAttribute(name="alpha", type=AttributeType.FLOAT, required=False)],
        inputs=["self", "other"],
        outputs=["out"],
    ),
    OpSchema(name="matmul", domain="aten", inputs=["self", "other"], outputs=["out"]),
    OpSchema(name="relu", domain="aten", inputs=["self"], outputs=["out"]),
    OpSchema(
        name="layer_norm",
        domain="aten",
        inputs=["input", "normalized_shape", "weight", "bias"],
        outputs=["out"],
    ),
    OpSchema(
        name="scaled_dot_product_attention",
        domain="aten",
        inputs=["query", "key", "value"],
        outputs=["out"],
    ),
]

_CANONICAL_ARRAY_API_OPS: list[OpSchema] = [
    OpSchema(name="abs", domain="array_api", inputs=["x"], outputs=["out"]),
    OpSchema(name="add", domain="array_api", inputs=["x1", "x2"], outputs=["out"]),
    OpSchema(name="matmul", domain="array_api", inputs=["x1", "x2"], outputs=["out"]),
    OpSchema(name="mean", domain="array_api", inputs=["x"], outputs=["out"]),
    OpSchema(
        name="sum",
        domain="array_api",
        attributes=[
            OpAttribute(name="axis", type=AttributeType.INT, required=False),
            OpAttribute(name="keepdims", type=AttributeType.BOOL, required=False),
        ],
        inputs=["x"],
        outputs=["out"],
    ),
]

_CANONICAL_CHEX_OPS: list[OpSchema] = [
    OpSchema(name="assert_shape", domain="chex", inputs=["inputs", "expected_shapes"], outputs=["out"]),
    OpSchema(name="assert_rank", domain="chex", inputs=["inputs", "expected_ranks"], outputs=["out"]),
    OpSchema(name="assert_type", domain="chex", inputs=["inputs", "expected_types"], outputs=["out"]),
    OpSchema(
        name="assert_equal_shape",
        domain="chex",
        attributes=[OpAttribute(name="dims", type=AttributeType.LIST_INT, required=False)],
        inputs=["inputs"],
        outputs=["out"],
    ),
    OpSchema(
        name="assert_trees_all_close",
        domain="chex",
        attributes=[
            OpAttribute(name="rtol", type=AttributeType.FLOAT, required=False),
            OpAttribute(name="atol", type=AttributeType.FLOAT, required=False),
        ],
        inputs=["trees"],
        outputs=["out"],
    ),
]

_CANONICAL_GRAIN_OPS: list[OpSchema] = [
    OpSchema(name="MapDataset", domain="grain", inputs=["parents"], outputs=["dataset"]),
    OpSchema(name="IterDataset", domain="grain", inputs=["parents"], outputs=["dataset"]),
    OpSchema(
        name="DataLoader",
        domain="grain",
        attributes=[
            OpAttribute(name="worker_count", type=AttributeType.INT, required=False),
            OpAttribute(name="worker_buffer_size", type=AttributeType.INT, required=False),
        ],
        inputs=["data_source", "sampler", "operations"],
        outputs=["loader"],
    ),
    OpSchema(
        name="batch",
        domain="grain",
        attributes=[
            OpAttribute(name="batch_size", type=AttributeType.INT, required=True),
            OpAttribute(name="drop_remainder", type=AttributeType.BOOL, required=False),
        ],
        inputs=["dataset"],
        outputs=["batched"],
    ),
    OpSchema(name="map", domain="grain", inputs=["dataset", "transform"], outputs=["mapped"]),
]

_CANONICAL_ORBAX_OPS: list[OpSchema] = [
    OpSchema(
        name="PyTreeCheckpointer",
        domain="orbax",
        attributes=[
            OpAttribute(name="primary_host", type=AttributeType.INT, required=False),
            OpAttribute(name="use_ocdbt", type=AttributeType.BOOL, required=False),
            OpAttribute(name="use_zarr3", type=AttributeType.BOOL, required=False),
        ],
        inputs=[],
        outputs=["checkpointer"],
    ),
    OpSchema(name="StandardCheckpointer", domain="orbax", inputs=[], outputs=["checkpointer"]),
    OpSchema(
        name="CheckpointManager",
        domain="orbax",
        inputs=["directory", "checkpointers"],
        outputs=["manager"],
    ),
    OpSchema(
        name="AsyncCheckpointer",
        domain="orbax",
        attributes=[OpAttribute(name="timeout_secs", type=AttributeType.INT, required=False)],
        inputs=["handler"],
        outputs=["checkpointer"],
    ),
    OpSchema(
        name="SaveArgs",
        domain="orbax",
        attributes=[
            OpAttribute(name="aggregate", type=AttributeType.BOOL, required=False),
            OpAttribute(name="dtype", type=AttributeType.STR, required=False),
            OpAttribute(name="chunk_byte_size", type=AttributeType.INT, required=False),
        ],
        inputs=[],
        outputs=["save_args"],
    ),
]

_CANONICAL_OPTAX_OPS: list[OpSchema] = [
    OpSchema(
        name="adam",
        domain="optax",
        attributes=[
            OpAttribute(name="learning_rate", type=AttributeType.FLOAT, required=True),
            OpAttribute(name="b1", type=AttributeType.FLOAT, required=False, default=0.9),
            OpAttribute(name="b2", type=AttributeType.FLOAT, required=False, default=0.999),
            OpAttribute(name="eps", type=AttributeType.FLOAT, required=False, default=1e-8),
            OpAttribute(name="nesterov", type=AttributeType.BOOL, required=False, default=False),
        ],
        inputs=[],
        outputs=["tx"],
    ),
    OpSchema(
        name="adamw",
        domain="optax",
        attributes=[
            OpAttribute(name="learning_rate", type=AttributeType.FLOAT, required=True),
            OpAttribute(name="b1", type=AttributeType.FLOAT, required=False, default=0.9),
            OpAttribute(name="b2", type=AttributeType.FLOAT, required=False, default=0.999),
            OpAttribute(name="eps", type=AttributeType.FLOAT, required=False, default=1e-8),
            OpAttribute(name="weight_decay", type=AttributeType.FLOAT, required=False, default=0.0001),
            OpAttribute(name="nesterov", type=AttributeType.BOOL, required=False, default=False),
        ],
        inputs=[],
        outputs=["tx"],
    ),
    OpSchema(
        name="sgd",
        domain="optax",
        attributes=[
            OpAttribute(name="learning_rate", type=AttributeType.FLOAT, required=True),
            OpAttribute(name="momentum", type=AttributeType.FLOAT, required=False),
            OpAttribute(name="nesterov", type=AttributeType.BOOL, required=False, default=False),
        ],
        inputs=[],
        outputs=["tx"],
    ),
    OpSchema(
        name="cosine_decay_schedule",
        domain="optax",
        attributes=[
            OpAttribute(name="init_value", type=AttributeType.FLOAT, required=True),
            OpAttribute(name="decay_steps", type=AttributeType.INT, required=True),
            OpAttribute(name="alpha", type=AttributeType.FLOAT, required=False, default=0.0),
            OpAttribute(name="exponent", type=AttributeType.FLOAT, required=False, default=1.0),
        ],
        inputs=[],
        outputs=["schedule"],
    ),
    OpSchema(
        name="softmax_cross_entropy",
        domain="optax",
        attributes=[OpAttribute(name="axis", type=AttributeType.INT, required=False, default=-1)],
        inputs=["logits", "labels"],
        outputs=["loss"],
    ),
]


def _load_framework_schema_file(
    filename: str, domain_default: str, json_path: Path | str | None = None
) -> dict[str, OpSchema]:
    """Load operator schemas from a JSON file into OpSchema mappings.

    Args:
        filename (str): JSON filename in schema directory.
        domain_default (str): Default domain identifier.
        json_path (Path | str | None): Explicit path override for schema JSON.

    Returns:
        dict[str, OpSchema]: Mapping from operator names to OpSchema instances.
    """

    def _record_dialects(op_data: Mapping[str, JsonValue]) -> None:
        """Record dialect mappings from an ODL operator entry if present.

        Args:
            op_data (Mapping[str, JsonValue]): Raw operator definition dictionary.
        """
        dialects = op_data.get("dialects")
        if isinstance(dialects, dict):
            ODL_DIALECT_MAPPINGS[str(op_data["name"])] = {str(k): str(v) for k, v in dialects.items()}

    ODL_DIALECT_MAPPINGS.mark_loaded()
    return load_op_schema_catalog(
        filename,
        domain_default=domain_default,
        json_path=json_path,
        find_file_fn=find_schema_file,
        on_entry=_record_dialects,
    )


def _initialize_framework_registries() -> None:
    """Initialize all framework registries and the ODL catalog lazily under REGISTRY_LOCK."""
    specs: tuple[tuple[SchemaRegistry, str, str, list[OpSchema]], ...] = (
        (ATEN_REGISTRY, "aten_ops.json", "aten", _CANONICAL_ATEN_OPS),
        (ARRAY_API_REGISTRY, "array_api_ops.json", "array_api", _CANONICAL_ARRAY_API_OPS),
        (CHEX_REGISTRY, "chex_ops.json", "chex", _CANONICAL_CHEX_OPS),
        (GRAIN_REGISTRY, "grain_ops.json", "grain", _CANONICAL_GRAIN_OPS),
        (ORBAX_REGISTRY, "orbax_ops.json", "orbax", _CANONICAL_ORBAX_OPS),
        (OPTAX_REGISTRY, "optax_ops.json", "optax", _CANONICAL_OPTAX_OPS),
        (ODL_CATALOG, "odl_ops.json", "odl", _CANONICAL_ODL_OPS),
    )
    with REGISTRY_LOCK:
        ODL_DIALECT_MAPPINGS.mark_loaded()
        for reg, _filename, _domain, _canonical_ops in specs:
            reg._loaded = True
        for reg, filename, domain, canonical_ops in specs:
            loaded = _load_framework_schema_file(filename, domain)
            if not loaded:
                for schema in canonical_ops:
                    loaded[schema.name] = schema
            reg.update(loaded)
        for op_name, mapping in _CANONICAL_ODL_DIALECT_MAPPINGS.items():
            if op_name not in ODL_DIALECT_MAPPINGS:
                ODL_DIALECT_MAPPINGS[op_name] = dict(mapping)


ATEN_REGISTRY: SchemaRegistry = SchemaRegistry("ATEN_REGISTRY", loader=_initialize_framework_registries)
ARRAY_API_REGISTRY: SchemaRegistry = SchemaRegistry("ARRAY_API_REGISTRY", loader=_initialize_framework_registries)
CHEX_REGISTRY: SchemaRegistry = SchemaRegistry("CHEX_REGISTRY", loader=_initialize_framework_registries)
GRAIN_REGISTRY: SchemaRegistry = SchemaRegistry("GRAIN_REGISTRY", loader=_initialize_framework_registries)
ORBAX_REGISTRY: SchemaRegistry = SchemaRegistry("ORBAX_REGISTRY", loader=_initialize_framework_registries)
OPTAX_REGISTRY: SchemaRegistry = SchemaRegistry("OPTAX_REGISTRY", loader=_initialize_framework_registries)
ODL_CATALOG: SchemaRegistry = SchemaRegistry("ODL_CATALOG", loader=_initialize_framework_registries)
ODL_DIALECT_MAPPINGS: LazyMetadataMap[dict[str, str]] = LazyMetadataMap(
    "ODL_DIALECT_MAPPINGS", loader=_initialize_framework_registries
)


def get_abstract_op(name: str) -> OpSchema | None:
    """Look up a universal abstract operation in the ODL catalog.

    Args:
        name (str): Abstract operation name (e.g. 'add', 'matmul', 'relu', 'layer_norm').

    Returns:
        OpSchema | None: Canonical OpSchema if defined in ODL catalog, else None.
    """
    return ODL_CATALOG.get(name)


__all__ = [
    "ARRAY_API_REGISTRY",
    "ATEN_REGISTRY",
    "CHEX_REGISTRY",
    "GRAIN_REGISTRY",
    "ODL_CATALOG",
    "ODL_DIALECT_MAPPINGS",
    "OPTAX_REGISTRY",
    "ORBAX_REGISTRY",
    "get_abstract_op",
]
