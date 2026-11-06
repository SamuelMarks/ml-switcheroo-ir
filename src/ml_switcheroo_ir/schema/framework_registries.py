"""Framework operator registries and universal Operation Definition Language (ODL) catalog."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ml_switcheroo_ir.schema.base import (
    REGISTRY_LOCK,
    CustomAttributeSchema,
    CustomOpSchema,
    OpSchema,
    load_op_schema_catalog,
)
from ml_switcheroo_ir.snapshots import find_schema_file
from ml_switcheroo_ir.types import JsonValue

ATEN_REGISTRY: dict[str, OpSchema] = {}
ARRAY_API_REGISTRY: dict[str, OpSchema] = {}
CHEX_REGISTRY: dict[str, OpSchema] = {}
GRAIN_REGISTRY: dict[str, OpSchema] = {}
ORBAX_REGISTRY: dict[str, OpSchema] = {}
OPTAX_REGISTRY: dict[str, OpSchema] = {}
ODL_CATALOG: dict[str, OpSchema] = {}
ODL_DIALECT_MAPPINGS: dict[str, dict[str, str]] = {}

_CANONICAL_ODL_OPS: list[CustomOpSchema] = [
    CustomOpSchema(
        name="add",
        domain="odl",
        attributes=[],
        inputs=["a", "b"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="subtract",
        domain="odl",
        attributes=[],
        inputs=["a", "b"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="multiply",
        domain="odl",
        attributes=[],
        inputs=["a", "b"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="divide",
        domain="odl",
        attributes=[],
        inputs=["a", "b"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="matmul",
        domain="odl",
        attributes=[],
        inputs=["a", "b"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="relu",
        domain="odl",
        attributes=[],
        inputs=["x"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="gelu",
        domain="odl",
        attributes=[],
        inputs=["x"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="silu",
        domain="odl",
        attributes=[],
        inputs=["x"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="softmax",
        domain="odl",
        attributes=[
            CustomAttributeSchema(
                name="dim",
                type="int",
                required=False,
                default=-1,
            )
        ],
        inputs=["x"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="layer_norm",
        domain="odl",
        attributes=[
            CustomAttributeSchema(
                name="eps",
                type="float",
                required=False,
                default=1e-5,
            )
        ],
        inputs=["x", "scale", "bias"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="rms_norm",
        domain="odl",
        attributes=[
            CustomAttributeSchema(
                name="eps",
                type="float",
                required=False,
                default=1e-6,
            )
        ],
        inputs=["x", "scale"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="conv2d",
        domain="odl",
        attributes=[],
        inputs=["x", "weight"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="scaled_dot_product_attention",
        domain="odl",
        attributes=[],
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

_CANONICAL_ATEN_OPS: list[CustomOpSchema] = [
    CustomOpSchema(
        name="add",
        domain="aten",
        attributes=[
            CustomAttributeSchema(name="alpha", type="float", required=False),
        ],
        inputs=["self", "other"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="matmul",
        domain="aten",
        attributes=[],
        inputs=["self", "other"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="relu",
        domain="aten",
        attributes=[],
        inputs=["self"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="layer_norm",
        domain="aten",
        attributes=[],
        inputs=["input", "normalized_shape", "weight", "bias"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="scaled_dot_product_attention",
        domain="aten",
        attributes=[],
        inputs=["query", "key", "value"],
        outputs=["out"],
    ),
]

_CANONICAL_ARRAY_API_OPS: list[CustomOpSchema] = [
    CustomOpSchema(
        name="abs",
        domain="array_api",
        attributes=[],
        inputs=["x"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="add",
        domain="array_api",
        attributes=[],
        inputs=["x1", "x2"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="matmul",
        domain="array_api",
        attributes=[],
        inputs=["x1", "x2"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="mean",
        domain="array_api",
        attributes=[],
        inputs=["x"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="sum",
        domain="array_api",
        attributes=[
            CustomAttributeSchema(name="axis", type="int", required=False),
            CustomAttributeSchema(name="keepdims", type="bool", required=False),
        ],
        inputs=["x"],
        outputs=["out"],
    ),
]

_CANONICAL_CHEX_OPS: list[CustomOpSchema] = [
    CustomOpSchema(
        name="assert_shape",
        domain="chex",
        attributes=[],
        inputs=["inputs", "expected_shapes"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="assert_rank",
        domain="chex",
        attributes=[],
        inputs=["inputs", "expected_ranks"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="assert_type",
        domain="chex",
        attributes=[],
        inputs=["inputs", "expected_types"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="assert_equal_shape",
        domain="chex",
        attributes=[
            CustomAttributeSchema(name="dims", type="ints", required=False),
        ],
        inputs=["inputs"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="assert_trees_all_close",
        domain="chex",
        attributes=[
            CustomAttributeSchema(name="rtol", type="float", required=False),
            CustomAttributeSchema(name="atol", type="float", required=False),
        ],
        inputs=["trees"],
        outputs=["out"],
    ),
]

_CANONICAL_GRAIN_OPS: list[CustomOpSchema] = [
    CustomOpSchema(
        name="MapDataset",
        domain="grain",
        attributes=[],
        inputs=["parents"],
        outputs=["dataset"],
    ),
    CustomOpSchema(
        name="IterDataset",
        domain="grain",
        attributes=[],
        inputs=["parents"],
        outputs=["dataset"],
    ),
    CustomOpSchema(
        name="DataLoader",
        domain="grain",
        attributes=[
            CustomAttributeSchema(name="worker_count", type="int", required=False),
            CustomAttributeSchema(name="worker_buffer_size", type="int", required=False),
        ],
        inputs=["data_source", "sampler", "operations"],
        outputs=["loader"],
    ),
    CustomOpSchema(
        name="batch",
        domain="grain",
        attributes=[
            CustomAttributeSchema(name="batch_size", type="int", required=True),
            CustomAttributeSchema(name="drop_remainder", type="bool", required=False),
        ],
        inputs=["dataset"],
        outputs=["batched"],
    ),
    CustomOpSchema(
        name="map",
        domain="grain",
        attributes=[],
        inputs=["dataset", "transform"],
        outputs=["mapped"],
    ),
]

_CANONICAL_ORBAX_OPS: list[CustomOpSchema] = [
    CustomOpSchema(
        name="PyTreeCheckpointer",
        domain="orbax",
        attributes=[
            CustomAttributeSchema(name="primary_host", type="int", required=False),
            CustomAttributeSchema(name="use_ocdbt", type="bool", required=False),
            CustomAttributeSchema(name="use_zarr3", type="bool", required=False),
        ],
        inputs=[],
        outputs=["checkpointer"],
    ),
    CustomOpSchema(
        name="StandardCheckpointer",
        domain="orbax",
        attributes=[],
        inputs=[],
        outputs=["checkpointer"],
    ),
    CustomOpSchema(
        name="CheckpointManager",
        domain="orbax",
        attributes=[],
        inputs=["directory", "checkpointers"],
        outputs=["manager"],
    ),
    CustomOpSchema(
        name="AsyncCheckpointer",
        domain="orbax",
        attributes=[
            CustomAttributeSchema(name="timeout_secs", type="int", required=False),
        ],
        inputs=["handler"],
        outputs=["checkpointer"],
    ),
    CustomOpSchema(
        name="SaveArgs",
        domain="orbax",
        attributes=[
            CustomAttributeSchema(name="aggregate", type="bool", required=False),
            CustomAttributeSchema(name="dtype", type="string", required=False),
            CustomAttributeSchema(name="chunk_byte_size", type="int", required=False),
        ],
        inputs=[],
        outputs=["save_args"],
    ),
]

_CANONICAL_OPTAX_OPS: list[CustomOpSchema] = [
    CustomOpSchema(
        name="adam",
        domain="optax",
        attributes=[
            CustomAttributeSchema(name="learning_rate", type="float", required=True),
            CustomAttributeSchema(name="b1", type="float", required=False, default=0.9),
            CustomAttributeSchema(name="b2", type="float", required=False, default=0.999),
            CustomAttributeSchema(name="eps", type="float", required=False, default=1e-8),
            CustomAttributeSchema(name="nesterov", type="bool", required=False, default=False),
        ],
        inputs=[],
        outputs=["tx"],
    ),
    CustomOpSchema(
        name="adamw",
        domain="optax",
        attributes=[
            CustomAttributeSchema(name="learning_rate", type="float", required=True),
            CustomAttributeSchema(name="b1", type="float", required=False, default=0.9),
            CustomAttributeSchema(name="b2", type="float", required=False, default=0.999),
            CustomAttributeSchema(name="eps", type="float", required=False, default=1e-8),
            CustomAttributeSchema(name="weight_decay", type="float", required=False, default=0.0001),
            CustomAttributeSchema(name="nesterov", type="bool", required=False, default=False),
        ],
        inputs=[],
        outputs=["tx"],
    ),
    CustomOpSchema(
        name="sgd",
        domain="optax",
        attributes=[
            CustomAttributeSchema(name="learning_rate", type="float", required=True),
            CustomAttributeSchema(name="momentum", type="float", required=False),
            CustomAttributeSchema(name="nesterov", type="bool", required=False, default=False),
        ],
        inputs=[],
        outputs=["tx"],
    ),
    CustomOpSchema(
        name="cosine_decay_schedule",
        domain="optax",
        attributes=[
            CustomAttributeSchema(name="init_value", type="float", required=True),
            CustomAttributeSchema(name="decay_steps", type="int", required=True),
            CustomAttributeSchema(name="alpha", type="float", required=False, default=0.0),
            CustomAttributeSchema(name="exponent", type="float", required=False, default=1.0),
        ],
        inputs=[],
        outputs=["schedule"],
    ),
    CustomOpSchema(
        name="softmax_cross_entropy",
        domain="optax",
        attributes=[
            CustomAttributeSchema(name="axis", type="int", required=False, default=-1),
        ],
        inputs=["logits", "labels"],
        outputs=["loss"],
    ),
]


def _load_framework_schema_file(filename: str, domain_default: str, json_path: Path | str | None = None) -> dict[str, OpSchema]:
    """Load operator schemas from a JSON file into OpSchema mappings.

    Args:
        filename (str): JSON filename in schema directory.
        domain_default (str): Default domain identifier.
        json_path (Optional[Union[Path, str]]): Explicit path override for schema JSON.

    Returns:
        dict[str, OpSchema]: Mapping from operator names to OpSchema instances.
    """

    def _record_dialects(op_data: Mapping[str, JsonValue]) -> None:
        """Record dialect mappings from an ODL operator entry if present.

        Args:
            op_data (Mapping[str, JsonValue]): Raw operator definition dictionary.
        """
        raw_dialects = op_data.get("dialects")
        if isinstance(raw_dialects, Mapping):
            ODL_DIALECT_MAPPINGS[str(op_data["name"])] = {str(k): str(v) for k, v in raw_dialects.items()}

    return load_op_schema_catalog(
        filename,
        domain_default=domain_default,
        json_path=json_path,
        find_file_fn=find_schema_file,
        on_entry=_record_dialects,
    )


def _initialize_framework_registries() -> None:
    """Initialize ATen, Array API, Chex, Grain, Orbax, Optax, and ODL catalog registries."""
    specs: tuple[tuple[dict[str, OpSchema], str, str, list[CustomOpSchema]], ...] = (
        (ATEN_REGISTRY, "aten_ops.json", "aten", _CANONICAL_ATEN_OPS),
        (ARRAY_API_REGISTRY, "array_api_ops.json", "array_api", _CANONICAL_ARRAY_API_OPS),
        (CHEX_REGISTRY, "chex_ops.json", "chex", _CANONICAL_CHEX_OPS),
        (GRAIN_REGISTRY, "grain_ops.json", "grain", _CANONICAL_GRAIN_OPS),
        (ORBAX_REGISTRY, "orbax_ops.json", "orbax", _CANONICAL_ORBAX_OPS),
        (OPTAX_REGISTRY, "optax_ops.json", "optax", _CANONICAL_OPTAX_OPS),
    )
    with REGISTRY_LOCK:
        for reg, filename, domain, canonical_ops in specs:
            loaded = _load_framework_schema_file(filename, domain)
            if not loaded:
                for s in canonical_ops:
                    loaded[s.name] = s.to_op_schema()
            reg.update(loaded)

        loaded_odl = _load_framework_schema_file("odl_catalog.json", "odl")
        if not loaded_odl:
            for s in _CANONICAL_ODL_OPS:
                loaded_odl[s.name] = s.to_op_schema()
            ODL_DIALECT_MAPPINGS.update(_CANONICAL_ODL_DIALECT_MAPPINGS)
        ODL_CATALOG.update(loaded_odl)


_initialize_framework_registries()


def get_abstract_op(name: str) -> OpSchema | None:
    """Retrieve canonical OpSchema definition for an abstract mathematical operation.

    Args:
        name (str): Abstract operation name (e.g. 'add', 'matmul', 'relu', 'layer_norm').

    Returns:
        OpSchema | None: Canonical OpSchema if defined in ODL catalog, else None.
    """
    return ODL_CATALOG.get(name)
