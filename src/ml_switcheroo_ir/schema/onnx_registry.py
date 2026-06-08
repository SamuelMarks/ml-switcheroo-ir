"""Generated ONNX Operator Registry."""

from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class OpAttribute:
    """Represents a single operator attribute schema.

    Attributes:
        name (str): The name of the attribute.
        type (str): The Python type annotation as a string.
        required (bool): Whether the attribute is required.
        default (Any): The default value of the attribute, if any.
    """

    name: str
    type: str
    required: bool
    default: Any


@dataclass
class OpSchema:
    """Represents a single operator schema.

    Attributes:
        name (str): The name of the operator.
        domain (str): The domain of the operator.
        version (int): The opset version.
        attributes (Dict[str, OpAttribute]): Dictionary mapping attribute name to schema.
        inputs (List[str]): List of expected input names.
        outputs (List[str]): List of expected output names.
    """

    name: str
    domain: str
    version: int
    attributes: Dict[str, OpAttribute]
    inputs: List[str]
    outputs: List[str]


ONNX_REGISTRY: Dict[str, OpSchema] = {
    "Abs": OpSchema(
        name="Abs",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["X"],
        outputs=["Y"],
    ),
    "Acos": OpSchema(
        name="Acos",
        domain="ai.onnx",
        version=22,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "Acosh": OpSchema(
        name="Acosh",
        domain="ai.onnx",
        version=22,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "Add": OpSchema(
        name="Add",
        domain="ai.onnx",
        version=14,
        attributes={},
        inputs=["A", "B"],
        outputs=["C"],
    ),
    "AffineGrid": OpSchema(
        name="AffineGrid",
        domain="ai.onnx",
        version=20,
        attributes={
            "align_corners": OpAttribute(
                name="align_corners", type="int", required=False, default=0
            ),
        },
        inputs=["theta", "size"],
        outputs=["grid"],
    ),
    "And": OpSchema(
        name="And",
        domain="ai.onnx",
        version=7,
        attributes={},
        inputs=["A", "B"],
        outputs=["C"],
    ),
    "ArgMax": OpSchema(
        name="ArgMax",
        domain="ai.onnx",
        version=13,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=0),
            "keepdims": OpAttribute(
                name="keepdims", type="int", required=False, default=1
            ),
            "select_last_index": OpAttribute(
                name="select_last_index", type="int", required=False, default=0
            ),
        },
        inputs=["data"],
        outputs=["reduced"],
    ),
    "ArgMin": OpSchema(
        name="ArgMin",
        domain="ai.onnx",
        version=13,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=0),
            "keepdims": OpAttribute(
                name="keepdims", type="int", required=False, default=1
            ),
            "select_last_index": OpAttribute(
                name="select_last_index", type="int", required=False, default=0
            ),
        },
        inputs=["data"],
        outputs=["reduced"],
    ),
    "Asin": OpSchema(
        name="Asin",
        domain="ai.onnx",
        version=22,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "Asinh": OpSchema(
        name="Asinh",
        domain="ai.onnx",
        version=22,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "Atan": OpSchema(
        name="Atan",
        domain="ai.onnx",
        version=22,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "Atanh": OpSchema(
        name="Atanh",
        domain="ai.onnx",
        version=22,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "Attention": OpSchema(
        name="Attention",
        domain="ai.onnx",
        version=24,
        attributes={
            "is_causal": OpAttribute(
                name="is_causal", type="int", required=False, default=0
            ),
            "kv_num_heads": OpAttribute(
                name="kv_num_heads", type="int", required=False, default=None
            ),
            "q_num_heads": OpAttribute(
                name="q_num_heads", type="int", required=False, default=None
            ),
            "qk_matmul_output_mode": OpAttribute(
                name="qk_matmul_output_mode", type="int", required=False, default=0
            ),
            "scale": OpAttribute(
                name="scale", type="float", required=False, default=None
            ),
            "softcap": OpAttribute(
                name="softcap", type="float", required=False, default=0.0
            ),
            "softmax_precision": OpAttribute(
                name="softmax_precision", type="int", required=False, default=None
            ),
        },
        inputs=[
            "Q",
            "K",
            "V",
            "attn_mask",
            "past_key",
            "past_value",
            "nonpad_kv_seqlen",
        ],
        outputs=["Y", "present_key", "present_value", "qk_matmul_output"],
    ),
    "AveragePool": OpSchema(
        name="AveragePool",
        domain="ai.onnx",
        version=22,
        attributes={
            "auto_pad": OpAttribute(
                name="auto_pad", type="str", required=False, default="NOTSET"
            ),
            "ceil_mode": OpAttribute(
                name="ceil_mode", type="int", required=False, default=0
            ),
            "count_include_pad": OpAttribute(
                name="count_include_pad", type="int", required=False, default=0
            ),
            "dilations": OpAttribute(
                name="dilations", type="List[int]", required=False, default=None
            ),
            "kernel_shape": OpAttribute(
                name="kernel_shape", type="List[int]", required=True, default=None
            ),
            "pads": OpAttribute(
                name="pads", type="List[int]", required=False, default=None
            ),
            "strides": OpAttribute(
                name="strides", type="List[int]", required=False, default=None
            ),
        },
        inputs=["X"],
        outputs=["Y"],
    ),
    "BatchNormalization": OpSchema(
        name="BatchNormalization",
        domain="ai.onnx",
        version=15,
        attributes={
            "epsilon": OpAttribute(
                name="epsilon", type="float", required=False, default=1e-05
            ),
            "momentum": OpAttribute(
                name="momentum", type="float", required=False, default=0.9
            ),
            "training_mode": OpAttribute(
                name="training_mode", type="int", required=False, default=0
            ),
        },
        inputs=["X", "scale", "B", "input_mean", "input_var"],
        outputs=["Y", "running_mean", "running_var"],
    ),
    "Bernoulli": OpSchema(
        name="Bernoulli",
        domain="ai.onnx",
        version=22,
        attributes={
            "dtype": OpAttribute(
                name="dtype", type="int", required=False, default=None
            ),
            "seed": OpAttribute(
                name="seed", type="float", required=False, default=None
            ),
        },
        inputs=["input"],
        outputs=["output"],
    ),
    "BitCast": OpSchema(
        name="BitCast",
        domain="ai.onnx",
        version=26,
        attributes={
            "to": OpAttribute(name="to", type="int", required=True, default=None),
        },
        inputs=["input"],
        outputs=["output"],
    ),
    "BitShift": OpSchema(
        name="BitShift",
        domain="ai.onnx",
        version=11,
        attributes={
            "direction": OpAttribute(
                name="direction", type="str", required=True, default=None
            ),
        },
        inputs=["X", "Y"],
        outputs=["Z"],
    ),
    "BitwiseAnd": OpSchema(
        name="BitwiseAnd",
        domain="ai.onnx",
        version=18,
        attributes={},
        inputs=["A", "B"],
        outputs=["C"],
    ),
    "BitwiseNot": OpSchema(
        name="BitwiseNot",
        domain="ai.onnx",
        version=18,
        attributes={},
        inputs=["X"],
        outputs=["Y"],
    ),
    "BitwiseOr": OpSchema(
        name="BitwiseOr",
        domain="ai.onnx",
        version=18,
        attributes={},
        inputs=["A", "B"],
        outputs=["C"],
    ),
    "BitwiseXor": OpSchema(
        name="BitwiseXor",
        domain="ai.onnx",
        version=18,
        attributes={},
        inputs=["A", "B"],
        outputs=["C"],
    ),
    "BlackmanWindow": OpSchema(
        name="BlackmanWindow",
        domain="ai.onnx",
        version=17,
        attributes={
            "output_datatype": OpAttribute(
                name="output_datatype", type="int", required=False, default=1
            ),
            "periodic": OpAttribute(
                name="periodic", type="int", required=False, default=1
            ),
        },
        inputs=["size"],
        outputs=["output"],
    ),
    "Cast": OpSchema(
        name="Cast",
        domain="ai.onnx",
        version=25,
        attributes={
            "round_mode": OpAttribute(
                name="round_mode", type="str", required=False, default="up"
            ),
            "saturate": OpAttribute(
                name="saturate", type="int", required=False, default=1
            ),
            "to": OpAttribute(name="to", type="int", required=True, default=None),
        },
        inputs=["input"],
        outputs=["output"],
    ),
    "CastLike": OpSchema(
        name="CastLike",
        domain="ai.onnx",
        version=25,
        attributes={
            "round_mode": OpAttribute(
                name="round_mode", type="str", required=False, default="up"
            ),
            "saturate": OpAttribute(
                name="saturate", type="int", required=False, default=1
            ),
        },
        inputs=["input", "target_type"],
        outputs=["output"],
    ),
    "CausalConvWithState": OpSchema(
        name="CausalConvWithState",
        domain="ai.onnx",
        version=27,
        attributes={
            "activation": OpAttribute(
                name="activation", type="str", required=False, default="none"
            ),
        },
        inputs=["input", "weight", "bias", "past_state"],
        outputs=["output", "present_state"],
    ),
    "Ceil": OpSchema(
        name="Ceil",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["X"],
        outputs=["Y"],
    ),
    "Celu": OpSchema(
        name="Celu",
        domain="ai.onnx",
        version=12,
        attributes={
            "alpha": OpAttribute(
                name="alpha", type="float", required=False, default=1.0
            ),
        },
        inputs=["X"],
        outputs=["Y"],
    ),
    "CenterCropPad": OpSchema(
        name="CenterCropPad",
        domain="ai.onnx",
        version=18,
        attributes={
            "axes": OpAttribute(
                name="axes", type="List[int]", required=False, default=None
            ),
        },
        inputs=["input_data", "shape"],
        outputs=["output_data"],
    ),
    "Clip": OpSchema(
        name="Clip",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["input", "min", "max"],
        outputs=["output"],
    ),
    "Col2Im": OpSchema(
        name="Col2Im",
        domain="ai.onnx",
        version=18,
        attributes={
            "dilations": OpAttribute(
                name="dilations", type="List[int]", required=False, default=None
            ),
            "pads": OpAttribute(
                name="pads", type="List[int]", required=False, default=None
            ),
            "strides": OpAttribute(
                name="strides", type="List[int]", required=False, default=None
            ),
        },
        inputs=["input", "image_shape", "block_shape"],
        outputs=["output"],
    ),
    "Compress": OpSchema(
        name="Compress",
        domain="ai.onnx",
        version=11,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=None),
        },
        inputs=["input", "condition"],
        outputs=["output"],
    ),
    "Concat": OpSchema(
        name="Concat",
        domain="ai.onnx",
        version=13,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=True, default=None),
        },
        inputs=["inputs"],
        outputs=["concat_result"],
    ),
    "ConcatFromSequence": OpSchema(
        name="ConcatFromSequence",
        domain="ai.onnx",
        version=11,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=True, default=None),
            "new_axis": OpAttribute(
                name="new_axis", type="int", required=False, default=0
            ),
        },
        inputs=["input_sequence"],
        outputs=["concat_result"],
    ),
    "Constant": OpSchema(
        name="Constant",
        domain="ai.onnx",
        version=25,
        attributes={
            "sparse_value": OpAttribute(
                name="sparse_value", type="Any", required=False, default=None
            ),
            "value": OpAttribute(
                name="value", type="Any", required=False, default=None
            ),
            "value_float": OpAttribute(
                name="value_float", type="float", required=False, default=None
            ),
            "value_floats": OpAttribute(
                name="value_floats", type="List[float]", required=False, default=None
            ),
            "value_int": OpAttribute(
                name="value_int", type="int", required=False, default=None
            ),
            "value_ints": OpAttribute(
                name="value_ints", type="List[int]", required=False, default=None
            ),
            "value_string": OpAttribute(
                name="value_string", type="str", required=False, default=None
            ),
            "value_strings": OpAttribute(
                name="value_strings", type="List[str]", required=False, default=None
            ),
        },
        inputs=["output"],
        outputs=["output"],
    ),
    "ConstantOfShape": OpSchema(
        name="ConstantOfShape",
        domain="ai.onnx",
        version=25,
        attributes={
            "value": OpAttribute(
                name="value", type="Any", required=False, default=None
            ),
        },
        inputs=["input"],
        outputs=["output"],
    ),
    "Conv": OpSchema(
        name="Conv",
        domain="ai.onnx",
        version=22,
        attributes={
            "auto_pad": OpAttribute(
                name="auto_pad", type="str", required=False, default="NOTSET"
            ),
            "dilations": OpAttribute(
                name="dilations", type="List[int]", required=False, default=None
            ),
            "group": OpAttribute(name="group", type="int", required=False, default=1),
            "kernel_shape": OpAttribute(
                name="kernel_shape", type="List[int]", required=False, default=None
            ),
            "pads": OpAttribute(
                name="pads", type="List[int]", required=False, default=None
            ),
            "strides": OpAttribute(
                name="strides", type="List[int]", required=False, default=None
            ),
        },
        inputs=["X", "W", "B"],
        outputs=["Y"],
    ),
    "ConvInteger": OpSchema(
        name="ConvInteger",
        domain="ai.onnx",
        version=10,
        attributes={
            "auto_pad": OpAttribute(
                name="auto_pad", type="str", required=False, default="NOTSET"
            ),
            "dilations": OpAttribute(
                name="dilations", type="List[int]", required=False, default=None
            ),
            "group": OpAttribute(name="group", type="int", required=False, default=1),
            "kernel_shape": OpAttribute(
                name="kernel_shape", type="List[int]", required=False, default=None
            ),
            "pads": OpAttribute(
                name="pads", type="List[int]", required=False, default=None
            ),
            "strides": OpAttribute(
                name="strides", type="List[int]", required=False, default=None
            ),
        },
        inputs=["x", "w", "x_zero_point", "w_zero_point"],
        outputs=["y"],
    ),
    "ConvTranspose": OpSchema(
        name="ConvTranspose",
        domain="ai.onnx",
        version=22,
        attributes={
            "auto_pad": OpAttribute(
                name="auto_pad", type="str", required=False, default="NOTSET"
            ),
            "dilations": OpAttribute(
                name="dilations", type="List[int]", required=False, default=None
            ),
            "group": OpAttribute(name="group", type="int", required=False, default=1),
            "kernel_shape": OpAttribute(
                name="kernel_shape", type="List[int]", required=False, default=None
            ),
            "output_padding": OpAttribute(
                name="output_padding", type="List[int]", required=False, default=None
            ),
            "output_shape": OpAttribute(
                name="output_shape", type="List[int]", required=False, default=None
            ),
            "pads": OpAttribute(
                name="pads", type="List[int]", required=False, default=None
            ),
            "strides": OpAttribute(
                name="strides", type="List[int]", required=False, default=None
            ),
        },
        inputs=["X", "W", "B"],
        outputs=["Y"],
    ),
    "Cos": OpSchema(
        name="Cos",
        domain="ai.onnx",
        version=22,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "Cosh": OpSchema(
        name="Cosh",
        domain="ai.onnx",
        version=22,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "CumProd": OpSchema(
        name="CumProd",
        domain="ai.onnx",
        version=26,
        attributes={
            "exclusive": OpAttribute(
                name="exclusive", type="int", required=False, default=0
            ),
            "reverse": OpAttribute(
                name="reverse", type="int", required=False, default=0
            ),
        },
        inputs=["x", "axis"],
        outputs=["y"],
    ),
    "CumSum": OpSchema(
        name="CumSum",
        domain="ai.onnx",
        version=14,
        attributes={
            "exclusive": OpAttribute(
                name="exclusive", type="int", required=False, default=0
            ),
            "reverse": OpAttribute(
                name="reverse", type="int", required=False, default=0
            ),
        },
        inputs=["x", "axis"],
        outputs=["y"],
    ),
    "DFT": OpSchema(
        name="DFT",
        domain="ai.onnx",
        version=20,
        attributes={
            "inverse": OpAttribute(
                name="inverse", type="int", required=False, default=0
            ),
            "onesided": OpAttribute(
                name="onesided", type="int", required=False, default=0
            ),
        },
        inputs=["input", "dft_length", "axis"],
        outputs=["output"],
    ),
    "DeformConv": OpSchema(
        name="DeformConv",
        domain="ai.onnx",
        version=22,
        attributes={
            "dilations": OpAttribute(
                name="dilations", type="List[int]", required=False, default=None
            ),
            "group": OpAttribute(name="group", type="int", required=False, default=1),
            "kernel_shape": OpAttribute(
                name="kernel_shape", type="List[int]", required=False, default=None
            ),
            "offset_group": OpAttribute(
                name="offset_group", type="int", required=False, default=1
            ),
            "pads": OpAttribute(
                name="pads", type="List[int]", required=False, default=None
            ),
            "strides": OpAttribute(
                name="strides", type="List[int]", required=False, default=None
            ),
        },
        inputs=["X", "W", "offset", "B", "mask"],
        outputs=["Y"],
    ),
    "DepthToSpace": OpSchema(
        name="DepthToSpace",
        domain="ai.onnx",
        version=13,
        attributes={
            "blocksize": OpAttribute(
                name="blocksize", type="int", required=True, default=None
            ),
            "mode": OpAttribute(name="mode", type="str", required=False, default="DCR"),
        },
        inputs=["input"],
        outputs=["output"],
    ),
    "DequantizeLinear": OpSchema(
        name="DequantizeLinear",
        domain="ai.onnx",
        version=25,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=1),
            "block_size": OpAttribute(
                name="block_size", type="int", required=False, default=0
            ),
            "output_dtype": OpAttribute(
                name="output_dtype", type="int", required=False, default=0
            ),
        },
        inputs=["x", "x_scale", "x_zero_point"],
        outputs=["y"],
    ),
    "Det": OpSchema(
        name="Det",
        domain="ai.onnx",
        version=22,
        attributes={},
        inputs=["X"],
        outputs=["Y"],
    ),
    "Div": OpSchema(
        name="Div",
        domain="ai.onnx",
        version=14,
        attributes={},
        inputs=["A", "B"],
        outputs=["C"],
    ),
    "Dropout": OpSchema(
        name="Dropout",
        domain="ai.onnx",
        version=22,
        attributes={
            "seed": OpAttribute(name="seed", type="int", required=False, default=None),
        },
        inputs=["data", "ratio", "training_mode"],
        outputs=["output", "mask"],
    ),
    "DynamicQuantizeLinear": OpSchema(
        name="DynamicQuantizeLinear",
        domain="ai.onnx",
        version=11,
        attributes={},
        inputs=["x"],
        outputs=["y", "y_scale", "y_zero_point"],
    ),
    "Einsum": OpSchema(
        name="Einsum",
        domain="ai.onnx",
        version=12,
        attributes={
            "equation": OpAttribute(
                name="equation", type="str", required=True, default=None
            ),
        },
        inputs=["Inputs"],
        outputs=["Output"],
    ),
    "Elu": OpSchema(
        name="Elu",
        domain="ai.onnx",
        version=22,
        attributes={
            "alpha": OpAttribute(
                name="alpha", type="float", required=False, default=1.0
            ),
        },
        inputs=["X"],
        outputs=["Y"],
    ),
    "Equal": OpSchema(
        name="Equal",
        domain="ai.onnx",
        version=19,
        attributes={},
        inputs=["A", "B"],
        outputs=["C"],
    ),
    "Erf": OpSchema(
        name="Erf",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "Exp": OpSchema(
        name="Exp",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "Expand": OpSchema(
        name="Expand",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["input", "shape"],
        outputs=["output"],
    ),
    "EyeLike": OpSchema(
        name="EyeLike",
        domain="ai.onnx",
        version=22,
        attributes={
            "dtype": OpAttribute(
                name="dtype", type="int", required=False, default=None
            ),
            "k": OpAttribute(name="k", type="int", required=False, default=0),
        },
        inputs=["input"],
        outputs=["output"],
    ),
    "Flatten": OpSchema(
        name="Flatten",
        domain="ai.onnx",
        version=25,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=1),
        },
        inputs=["input"],
        outputs=["output"],
    ),
    "Floor": OpSchema(
        name="Floor",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["X"],
        outputs=["Y"],
    ),
    "GRU": OpSchema(
        name="GRU",
        domain="ai.onnx",
        version=22,
        attributes={
            "activation_alpha": OpAttribute(
                name="activation_alpha",
                type="List[float]",
                required=False,
                default=None,
            ),
            "activation_beta": OpAttribute(
                name="activation_beta", type="List[float]", required=False, default=None
            ),
            "activations": OpAttribute(
                name="activations", type="List[str]", required=False, default=None
            ),
            "clip": OpAttribute(
                name="clip", type="float", required=False, default=None
            ),
            "direction": OpAttribute(
                name="direction", type="str", required=False, default="forward"
            ),
            "hidden_size": OpAttribute(
                name="hidden_size", type="int", required=False, default=None
            ),
            "layout": OpAttribute(name="layout", type="int", required=False, default=0),
            "linear_before_reset": OpAttribute(
                name="linear_before_reset", type="int", required=False, default=0
            ),
        },
        inputs=["X", "W", "R", "B", "sequence_lens", "initial_h"],
        outputs=["Y", "Y_h"],
    ),
    "Gather": OpSchema(
        name="Gather",
        domain="ai.onnx",
        version=13,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=0),
        },
        inputs=["data", "indices"],
        outputs=["output"],
    ),
    "GatherElements": OpSchema(
        name="GatherElements",
        domain="ai.onnx",
        version=13,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=0),
        },
        inputs=["data", "indices"],
        outputs=["output"],
    ),
    "GatherND": OpSchema(
        name="GatherND",
        domain="ai.onnx",
        version=13,
        attributes={
            "batch_dims": OpAttribute(
                name="batch_dims", type="int", required=False, default=0
            ),
        },
        inputs=["data", "indices"],
        outputs=["output"],
    ),
    "Gelu": OpSchema(
        name="Gelu",
        domain="ai.onnx",
        version=20,
        attributes={
            "approximate": OpAttribute(
                name="approximate", type="str", required=False, default="none"
            ),
        },
        inputs=["X"],
        outputs=["Y"],
    ),
    "Gemm": OpSchema(
        name="Gemm",
        domain="ai.onnx",
        version=13,
        attributes={
            "alpha": OpAttribute(
                name="alpha", type="float", required=False, default=1.0
            ),
            "beta": OpAttribute(name="beta", type="float", required=False, default=1.0),
            "transA": OpAttribute(name="transA", type="int", required=False, default=0),
            "transB": OpAttribute(name="transB", type="int", required=False, default=0),
        },
        inputs=["A", "B", "C"],
        outputs=["Y"],
    ),
    "GlobalAveragePool": OpSchema(
        name="GlobalAveragePool",
        domain="ai.onnx",
        version=22,
        attributes={},
        inputs=["X"],
        outputs=["Y"],
    ),
    "GlobalLpPool": OpSchema(
        name="GlobalLpPool",
        domain="ai.onnx",
        version=22,
        attributes={
            "p": OpAttribute(name="p", type="int", required=False, default=2),
        },
        inputs=["X"],
        outputs=["Y"],
    ),
    "GlobalMaxPool": OpSchema(
        name="GlobalMaxPool",
        domain="ai.onnx",
        version=22,
        attributes={},
        inputs=["X"],
        outputs=["Y"],
    ),
    "Greater": OpSchema(
        name="Greater",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["A", "B"],
        outputs=["C"],
    ),
    "GreaterOrEqual": OpSchema(
        name="GreaterOrEqual",
        domain="ai.onnx",
        version=16,
        attributes={},
        inputs=["A", "B"],
        outputs=["C"],
    ),
    "GridSample": OpSchema(
        name="GridSample",
        domain="ai.onnx",
        version=22,
        attributes={
            "align_corners": OpAttribute(
                name="align_corners", type="int", required=False, default=0
            ),
            "mode": OpAttribute(
                name="mode", type="str", required=False, default="linear"
            ),
            "padding_mode": OpAttribute(
                name="padding_mode", type="str", required=False, default="zeros"
            ),
        },
        inputs=["X", "grid"],
        outputs=["Y"],
    ),
    "GroupNormalization": OpSchema(
        name="GroupNormalization",
        domain="ai.onnx",
        version=21,
        attributes={
            "epsilon": OpAttribute(
                name="epsilon", type="float", required=False, default=1e-05
            ),
            "num_groups": OpAttribute(
                name="num_groups", type="int", required=True, default=None
            ),
            "stash_type": OpAttribute(
                name="stash_type", type="int", required=False, default=1
            ),
        },
        inputs=["X", "scale", "bias"],
        outputs=["Y"],
    ),
    "HammingWindow": OpSchema(
        name="HammingWindow",
        domain="ai.onnx",
        version=17,
        attributes={
            "output_datatype": OpAttribute(
                name="output_datatype", type="int", required=False, default=1
            ),
            "periodic": OpAttribute(
                name="periodic", type="int", required=False, default=1
            ),
        },
        inputs=["size"],
        outputs=["output"],
    ),
    "HannWindow": OpSchema(
        name="HannWindow",
        domain="ai.onnx",
        version=17,
        attributes={
            "output_datatype": OpAttribute(
                name="output_datatype", type="int", required=False, default=1
            ),
            "periodic": OpAttribute(
                name="periodic", type="int", required=False, default=1
            ),
        },
        inputs=["size"],
        outputs=["output"],
    ),
    "HardSigmoid": OpSchema(
        name="HardSigmoid",
        domain="ai.onnx",
        version=22,
        attributes={
            "alpha": OpAttribute(
                name="alpha", type="float", required=False, default=0.2
            ),
            "beta": OpAttribute(name="beta", type="float", required=False, default=0.5),
        },
        inputs=["X"],
        outputs=["Y"],
    ),
    "HardSwish": OpSchema(
        name="HardSwish",
        domain="ai.onnx",
        version=22,
        attributes={},
        inputs=["X"],
        outputs=["Y"],
    ),
    "Hardmax": OpSchema(
        name="Hardmax",
        domain="ai.onnx",
        version=13,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=-1),
        },
        inputs=["input"],
        outputs=["output"],
    ),
    "Identity": OpSchema(
        name="Identity",
        domain="ai.onnx",
        version=25,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "If": OpSchema(
        name="If",
        domain="ai.onnx",
        version=25,
        attributes={
            "else_branch": OpAttribute(
                name="else_branch", type="Any", required=True, default=None
            ),
            "then_branch": OpAttribute(
                name="then_branch", type="Any", required=True, default=None
            ),
        },
        inputs=["cond"],
        outputs=["outputs"],
    ),
    "ImageDecoder": OpSchema(
        name="ImageDecoder",
        domain="ai.onnx",
        version=20,
        attributes={
            "pixel_format": OpAttribute(
                name="pixel_format", type="str", required=False, default="RGB"
            ),
        },
        inputs=["encoded_stream"],
        outputs=["image"],
    ),
    "InstanceNormalization": OpSchema(
        name="InstanceNormalization",
        domain="ai.onnx",
        version=22,
        attributes={
            "epsilon": OpAttribute(
                name="epsilon", type="float", required=False, default=1e-05
            ),
        },
        inputs=["input", "scale", "B"],
        outputs=["output"],
    ),
    "IsInf": OpSchema(
        name="IsInf",
        domain="ai.onnx",
        version=20,
        attributes={
            "detect_negative": OpAttribute(
                name="detect_negative", type="int", required=False, default=1
            ),
            "detect_positive": OpAttribute(
                name="detect_positive", type="int", required=False, default=1
            ),
        },
        inputs=["X"],
        outputs=["Y"],
    ),
    "IsNaN": OpSchema(
        name="IsNaN",
        domain="ai.onnx",
        version=20,
        attributes={},
        inputs=["X"],
        outputs=["Y"],
    ),
    "LRN": OpSchema(
        name="LRN",
        domain="ai.onnx",
        version=13,
        attributes={
            "alpha": OpAttribute(
                name="alpha", type="float", required=False, default=0.0001
            ),
            "beta": OpAttribute(
                name="beta", type="float", required=False, default=0.75
            ),
            "bias": OpAttribute(name="bias", type="float", required=False, default=1.0),
            "size": OpAttribute(name="size", type="int", required=True, default=None),
        },
        inputs=["X"],
        outputs=["Y"],
    ),
    "LSTM": OpSchema(
        name="LSTM",
        domain="ai.onnx",
        version=22,
        attributes={
            "activation_alpha": OpAttribute(
                name="activation_alpha",
                type="List[float]",
                required=False,
                default=None,
            ),
            "activation_beta": OpAttribute(
                name="activation_beta", type="List[float]", required=False, default=None
            ),
            "activations": OpAttribute(
                name="activations", type="List[str]", required=False, default=None
            ),
            "clip": OpAttribute(
                name="clip", type="float", required=False, default=None
            ),
            "direction": OpAttribute(
                name="direction", type="str", required=False, default="forward"
            ),
            "hidden_size": OpAttribute(
                name="hidden_size", type="int", required=False, default=None
            ),
            "input_forget": OpAttribute(
                name="input_forget", type="int", required=False, default=0
            ),
            "layout": OpAttribute(name="layout", type="int", required=False, default=0),
        },
        inputs=["X", "W", "R", "B", "sequence_lens", "initial_h", "initial_c", "P"],
        outputs=["Y", "Y_h", "Y_c"],
    ),
    "LayerNormalization": OpSchema(
        name="LayerNormalization",
        domain="ai.onnx",
        version=17,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=-1),
            "epsilon": OpAttribute(
                name="epsilon", type="float", required=False, default=1e-05
            ),
            "stash_type": OpAttribute(
                name="stash_type", type="int", required=False, default=1
            ),
        },
        inputs=["X", "Scale", "B"],
        outputs=["Y", "Mean", "InvStdDev"],
    ),
    "LeakyRelu": OpSchema(
        name="LeakyRelu",
        domain="ai.onnx",
        version=16,
        attributes={
            "alpha": OpAttribute(
                name="alpha", type="float", required=False, default=0.01
            ),
        },
        inputs=["X"],
        outputs=["Y"],
    ),
    "Less": OpSchema(
        name="Less",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["A", "B"],
        outputs=["C"],
    ),
    "LessOrEqual": OpSchema(
        name="LessOrEqual",
        domain="ai.onnx",
        version=16,
        attributes={},
        inputs=["A", "B"],
        outputs=["C"],
    ),
    "LinearAttention": OpSchema(
        name="LinearAttention",
        domain="ai.onnx",
        version=27,
        attributes={
            "chunk_size": OpAttribute(
                name="chunk_size", type="int", required=False, default=64
            ),
            "kv_num_heads": OpAttribute(
                name="kv_num_heads", type="int", required=True, default=None
            ),
            "q_num_heads": OpAttribute(
                name="q_num_heads", type="int", required=True, default=None
            ),
            "scale": OpAttribute(
                name="scale", type="float", required=False, default=0.0
            ),
            "update_rule": OpAttribute(
                name="update_rule", type="str", required=False, default="gated_delta"
            ),
        },
        inputs=["query", "key", "value", "past_state", "decay", "beta"],
        outputs=["output", "present_state"],
    ),
    "Log": OpSchema(
        name="Log",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "LogSoftmax": OpSchema(
        name="LogSoftmax",
        domain="ai.onnx",
        version=13,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=-1),
        },
        inputs=["input"],
        outputs=["output"],
    ),
    "Loop": OpSchema(
        name="Loop",
        domain="ai.onnx",
        version=25,
        attributes={
            "body": OpAttribute(name="body", type="Any", required=True, default=None),
        },
        inputs=["M", "cond", "v_initial"],
        outputs=["v_final_and_scan_outputs"],
    ),
    "LpNormalization": OpSchema(
        name="LpNormalization",
        domain="ai.onnx",
        version=22,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=-1),
            "p": OpAttribute(name="p", type="int", required=False, default=2),
        },
        inputs=["input"],
        outputs=["output"],
    ),
    "LpPool": OpSchema(
        name="LpPool",
        domain="ai.onnx",
        version=22,
        attributes={
            "auto_pad": OpAttribute(
                name="auto_pad", type="str", required=False, default="NOTSET"
            ),
            "ceil_mode": OpAttribute(
                name="ceil_mode", type="int", required=False, default=0
            ),
            "dilations": OpAttribute(
                name="dilations", type="List[int]", required=False, default=None
            ),
            "kernel_shape": OpAttribute(
                name="kernel_shape", type="List[int]", required=True, default=None
            ),
            "p": OpAttribute(name="p", type="int", required=False, default=2),
            "pads": OpAttribute(
                name="pads", type="List[int]", required=False, default=None
            ),
            "strides": OpAttribute(
                name="strides", type="List[int]", required=False, default=None
            ),
        },
        inputs=["X"],
        outputs=["Y"],
    ),
    "MatMul": OpSchema(
        name="MatMul",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["A", "B"],
        outputs=["Y"],
    ),
    "MatMulInteger": OpSchema(
        name="MatMulInteger",
        domain="ai.onnx",
        version=10,
        attributes={},
        inputs=["A", "B", "a_zero_point", "b_zero_point"],
        outputs=["Y"],
    ),
    "Max": OpSchema(
        name="Max",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["data_0"],
        outputs=["max"],
    ),
    "MaxPool": OpSchema(
        name="MaxPool",
        domain="ai.onnx",
        version=22,
        attributes={
            "auto_pad": OpAttribute(
                name="auto_pad", type="str", required=False, default="NOTSET"
            ),
            "ceil_mode": OpAttribute(
                name="ceil_mode", type="int", required=False, default=0
            ),
            "dilations": OpAttribute(
                name="dilations", type="List[int]", required=False, default=None
            ),
            "kernel_shape": OpAttribute(
                name="kernel_shape", type="List[int]", required=True, default=None
            ),
            "pads": OpAttribute(
                name="pads", type="List[int]", required=False, default=None
            ),
            "storage_order": OpAttribute(
                name="storage_order", type="int", required=False, default=0
            ),
            "strides": OpAttribute(
                name="strides", type="List[int]", required=False, default=None
            ),
        },
        inputs=["X"],
        outputs=["Y", "Indices"],
    ),
    "MaxRoiPool": OpSchema(
        name="MaxRoiPool",
        domain="ai.onnx",
        version=22,
        attributes={
            "pooled_shape": OpAttribute(
                name="pooled_shape", type="List[int]", required=True, default=None
            ),
            "spatial_scale": OpAttribute(
                name="spatial_scale", type="float", required=False, default=1.0
            ),
        },
        inputs=["X", "rois"],
        outputs=["Y"],
    ),
    "MaxUnpool": OpSchema(
        name="MaxUnpool",
        domain="ai.onnx",
        version=22,
        attributes={
            "kernel_shape": OpAttribute(
                name="kernel_shape", type="List[int]", required=True, default=None
            ),
            "pads": OpAttribute(
                name="pads", type="List[int]", required=False, default=None
            ),
            "strides": OpAttribute(
                name="strides", type="List[int]", required=False, default=None
            ),
        },
        inputs=["X", "I", "output_shape"],
        outputs=["output"],
    ),
    "Mean": OpSchema(
        name="Mean",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["data_0"],
        outputs=["mean"],
    ),
    "MeanVarianceNormalization": OpSchema(
        name="MeanVarianceNormalization",
        domain="ai.onnx",
        version=13,
        attributes={
            "axes": OpAttribute(
                name="axes", type="List[int]", required=False, default=["0", "2", "3"]
            ),
        },
        inputs=["X"],
        outputs=["Y"],
    ),
    "MelWeightMatrix": OpSchema(
        name="MelWeightMatrix",
        domain="ai.onnx",
        version=17,
        attributes={
            "output_datatype": OpAttribute(
                name="output_datatype", type="int", required=False, default=1
            ),
        },
        inputs=[
            "num_mel_bins",
            "dft_length",
            "sample_rate",
            "lower_edge_hertz",
            "upper_edge_hertz",
        ],
        outputs=["output"],
    ),
    "Min": OpSchema(
        name="Min",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["data_0"],
        outputs=["min"],
    ),
    "Mish": OpSchema(
        name="Mish",
        domain="ai.onnx",
        version=22,
        attributes={},
        inputs=["X"],
        outputs=["Y"],
    ),
    "Mod": OpSchema(
        name="Mod",
        domain="ai.onnx",
        version=13,
        attributes={
            "fmod": OpAttribute(name="fmod", type="int", required=False, default=0),
        },
        inputs=["A", "B"],
        outputs=["C"],
    ),
    "Mul": OpSchema(
        name="Mul",
        domain="ai.onnx",
        version=14,
        attributes={},
        inputs=["A", "B"],
        outputs=["C"],
    ),
    "Multinomial": OpSchema(
        name="Multinomial",
        domain="ai.onnx",
        version=22,
        attributes={
            "dtype": OpAttribute(name="dtype", type="int", required=False, default=6),
            "sample_size": OpAttribute(
                name="sample_size", type="int", required=False, default=1
            ),
            "seed": OpAttribute(
                name="seed", type="float", required=False, default=None
            ),
        },
        inputs=["input"],
        outputs=["output"],
    ),
    "Neg": OpSchema(
        name="Neg",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["X"],
        outputs=["Y"],
    ),
    "NegativeLogLikelihoodLoss": OpSchema(
        name="NegativeLogLikelihoodLoss",
        domain="ai.onnx",
        version=22,
        attributes={
            "ignore_index": OpAttribute(
                name="ignore_index", type="int", required=False, default=None
            ),
            "reduction": OpAttribute(
                name="reduction", type="str", required=False, default="mean"
            ),
        },
        inputs=["input", "target", "weight"],
        outputs=["loss"],
    ),
    "NonMaxSuppression": OpSchema(
        name="NonMaxSuppression",
        domain="ai.onnx",
        version=11,
        attributes={
            "center_point_box": OpAttribute(
                name="center_point_box", type="int", required=False, default=0
            ),
        },
        inputs=[
            "boxes",
            "scores",
            "max_output_boxes_per_class",
            "iou_threshold",
            "score_threshold",
        ],
        outputs=["selected_indices"],
    ),
    "NonZero": OpSchema(
        name="NonZero",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["X"],
        outputs=["Y"],
    ),
    "Not": OpSchema(
        name="Not",
        domain="ai.onnx",
        version=1,
        attributes={},
        inputs=["X"],
        outputs=["Y"],
    ),
    "OneHot": OpSchema(
        name="OneHot",
        domain="ai.onnx",
        version=11,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=-1),
        },
        inputs=["indices", "depth", "values"],
        outputs=["output"],
    ),
    "Optional": OpSchema(
        name="Optional",
        domain="ai.onnx",
        version=15,
        attributes={
            "type": OpAttribute(name="type", type="Any", required=False, default=None),
        },
        inputs=["input"],
        outputs=["output"],
    ),
    "OptionalGetElement": OpSchema(
        name="OptionalGetElement",
        domain="ai.onnx",
        version=18,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "OptionalHasElement": OpSchema(
        name="OptionalHasElement",
        domain="ai.onnx",
        version=18,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "Or": OpSchema(
        name="Or",
        domain="ai.onnx",
        version=7,
        attributes={},
        inputs=["A", "B"],
        outputs=["C"],
    ),
    "PRelu": OpSchema(
        name="PRelu",
        domain="ai.onnx",
        version=16,
        attributes={},
        inputs=["X", "slope"],
        outputs=["Y"],
    ),
    "Pad": OpSchema(
        name="Pad",
        domain="ai.onnx",
        version=25,
        attributes={
            "mode": OpAttribute(
                name="mode", type="str", required=False, default="constant"
            ),
        },
        inputs=["data", "pads", "constant_value", "axes"],
        outputs=["output"],
    ),
    "Pow": OpSchema(
        name="Pow",
        domain="ai.onnx",
        version=15,
        attributes={},
        inputs=["X", "Y"],
        outputs=["Z"],
    ),
    "QLinearConv": OpSchema(
        name="QLinearConv",
        domain="ai.onnx",
        version=10,
        attributes={
            "auto_pad": OpAttribute(
                name="auto_pad", type="str", required=False, default="NOTSET"
            ),
            "dilations": OpAttribute(
                name="dilations", type="List[int]", required=False, default=None
            ),
            "group": OpAttribute(name="group", type="int", required=False, default=1),
            "kernel_shape": OpAttribute(
                name="kernel_shape", type="List[int]", required=False, default=None
            ),
            "pads": OpAttribute(
                name="pads", type="List[int]", required=False, default=None
            ),
            "strides": OpAttribute(
                name="strides", type="List[int]", required=False, default=None
            ),
        },
        inputs=[
            "x",
            "x_scale",
            "x_zero_point",
            "w",
            "w_scale",
            "w_zero_point",
            "y_scale",
            "y_zero_point",
            "B",
        ],
        outputs=["y"],
    ),
    "QLinearMatMul": OpSchema(
        name="QLinearMatMul",
        domain="ai.onnx",
        version=21,
        attributes={},
        inputs=[
            "a",
            "a_scale",
            "a_zero_point",
            "b",
            "b_scale",
            "b_zero_point",
            "y_scale",
            "y_zero_point",
        ],
        outputs=["y"],
    ),
    "QuantizeLinear": OpSchema(
        name="QuantizeLinear",
        domain="ai.onnx",
        version=25,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=1),
            "block_size": OpAttribute(
                name="block_size", type="int", required=False, default=0
            ),
            "output_dtype": OpAttribute(
                name="output_dtype", type="int", required=False, default=0
            ),
            "precision": OpAttribute(
                name="precision", type="int", required=False, default=0
            ),
            "saturate": OpAttribute(
                name="saturate", type="int", required=False, default=1
            ),
        },
        inputs=["x", "y_scale", "y_zero_point"],
        outputs=["y"],
    ),
    "RMSNormalization": OpSchema(
        name="RMSNormalization",
        domain="ai.onnx",
        version=23,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=-1),
            "epsilon": OpAttribute(
                name="epsilon", type="float", required=False, default=1e-05
            ),
            "stash_type": OpAttribute(
                name="stash_type", type="int", required=False, default=1
            ),
        },
        inputs=["X", "scale"],
        outputs=["Y"],
    ),
    "RNN": OpSchema(
        name="RNN",
        domain="ai.onnx",
        version=22,
        attributes={
            "activation_alpha": OpAttribute(
                name="activation_alpha",
                type="List[float]",
                required=False,
                default=None,
            ),
            "activation_beta": OpAttribute(
                name="activation_beta", type="List[float]", required=False, default=None
            ),
            "activations": OpAttribute(
                name="activations",
                type="List[str]",
                required=False,
                default=["Tanh", "Tanh"],
            ),
            "clip": OpAttribute(
                name="clip", type="float", required=False, default=None
            ),
            "direction": OpAttribute(
                name="direction", type="str", required=False, default="forward"
            ),
            "hidden_size": OpAttribute(
                name="hidden_size", type="int", required=False, default=None
            ),
            "layout": OpAttribute(name="layout", type="int", required=False, default=0),
        },
        inputs=["X", "W", "R", "B", "sequence_lens", "initial_h"],
        outputs=["Y", "Y_h"],
    ),
    "RandomNormal": OpSchema(
        name="RandomNormal",
        domain="ai.onnx",
        version=22,
        attributes={
            "dtype": OpAttribute(name="dtype", type="int", required=False, default=1),
            "mean": OpAttribute(name="mean", type="float", required=False, default=0.0),
            "scale": OpAttribute(
                name="scale", type="float", required=False, default=1.0
            ),
            "seed": OpAttribute(
                name="seed", type="float", required=False, default=None
            ),
            "shape": OpAttribute(
                name="shape", type="List[int]", required=True, default=None
            ),
        },
        inputs=["output"],
        outputs=["output"],
    ),
    "RandomNormalLike": OpSchema(
        name="RandomNormalLike",
        domain="ai.onnx",
        version=22,
        attributes={
            "dtype": OpAttribute(
                name="dtype", type="int", required=False, default=None
            ),
            "mean": OpAttribute(name="mean", type="float", required=False, default=0.0),
            "scale": OpAttribute(
                name="scale", type="float", required=False, default=1.0
            ),
            "seed": OpAttribute(
                name="seed", type="float", required=False, default=None
            ),
        },
        inputs=["input"],
        outputs=["output"],
    ),
    "RandomUniform": OpSchema(
        name="RandomUniform",
        domain="ai.onnx",
        version=22,
        attributes={
            "dtype": OpAttribute(name="dtype", type="int", required=False, default=1),
            "high": OpAttribute(name="high", type="float", required=False, default=1.0),
            "low": OpAttribute(name="low", type="float", required=False, default=0.0),
            "seed": OpAttribute(
                name="seed", type="float", required=False, default=None
            ),
            "shape": OpAttribute(
                name="shape", type="List[int]", required=True, default=None
            ),
        },
        inputs=["output"],
        outputs=["output"],
    ),
    "RandomUniformLike": OpSchema(
        name="RandomUniformLike",
        domain="ai.onnx",
        version=22,
        attributes={
            "dtype": OpAttribute(
                name="dtype", type="int", required=False, default=None
            ),
            "high": OpAttribute(name="high", type="float", required=False, default=1.0),
            "low": OpAttribute(name="low", type="float", required=False, default=0.0),
            "seed": OpAttribute(
                name="seed", type="float", required=False, default=None
            ),
        },
        inputs=["input"],
        outputs=["output"],
    ),
    "Range": OpSchema(
        name="Range",
        domain="ai.onnx",
        version=27,
        attributes={
            "stash_type": OpAttribute(
                name="stash_type", type="int", required=False, default=1
            ),
        },
        inputs=["start", "limit", "delta"],
        outputs=["output"],
    ),
    "Reciprocal": OpSchema(
        name="Reciprocal",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["X"],
        outputs=["Y"],
    ),
    "ReduceL1": OpSchema(
        name="ReduceL1",
        domain="ai.onnx",
        version=18,
        attributes={
            "keepdims": OpAttribute(
                name="keepdims", type="int", required=False, default=1
            ),
            "noop_with_empty_axes": OpAttribute(
                name="noop_with_empty_axes", type="int", required=False, default=0
            ),
        },
        inputs=["data", "axes"],
        outputs=["reduced"],
    ),
    "ReduceL2": OpSchema(
        name="ReduceL2",
        domain="ai.onnx",
        version=18,
        attributes={
            "keepdims": OpAttribute(
                name="keepdims", type="int", required=False, default=1
            ),
            "noop_with_empty_axes": OpAttribute(
                name="noop_with_empty_axes", type="int", required=False, default=0
            ),
        },
        inputs=["data", "axes"],
        outputs=["reduced"],
    ),
    "ReduceLogSum": OpSchema(
        name="ReduceLogSum",
        domain="ai.onnx",
        version=18,
        attributes={
            "keepdims": OpAttribute(
                name="keepdims", type="int", required=False, default=1
            ),
            "noop_with_empty_axes": OpAttribute(
                name="noop_with_empty_axes", type="int", required=False, default=0
            ),
        },
        inputs=["data", "axes"],
        outputs=["reduced"],
    ),
    "ReduceLogSumExp": OpSchema(
        name="ReduceLogSumExp",
        domain="ai.onnx",
        version=18,
        attributes={
            "keepdims": OpAttribute(
                name="keepdims", type="int", required=False, default=1
            ),
            "noop_with_empty_axes": OpAttribute(
                name="noop_with_empty_axes", type="int", required=False, default=0
            ),
        },
        inputs=["data", "axes"],
        outputs=["reduced"],
    ),
    "ReduceMax": OpSchema(
        name="ReduceMax",
        domain="ai.onnx",
        version=20,
        attributes={
            "keepdims": OpAttribute(
                name="keepdims", type="int", required=False, default=1
            ),
            "noop_with_empty_axes": OpAttribute(
                name="noop_with_empty_axes", type="int", required=False, default=0
            ),
        },
        inputs=["data", "axes"],
        outputs=["reduced"],
    ),
    "ReduceMean": OpSchema(
        name="ReduceMean",
        domain="ai.onnx",
        version=18,
        attributes={
            "keepdims": OpAttribute(
                name="keepdims", type="int", required=False, default=1
            ),
            "noop_with_empty_axes": OpAttribute(
                name="noop_with_empty_axes", type="int", required=False, default=0
            ),
        },
        inputs=["data", "axes"],
        outputs=["reduced"],
    ),
    "ReduceMin": OpSchema(
        name="ReduceMin",
        domain="ai.onnx",
        version=20,
        attributes={
            "keepdims": OpAttribute(
                name="keepdims", type="int", required=False, default=1
            ),
            "noop_with_empty_axes": OpAttribute(
                name="noop_with_empty_axes", type="int", required=False, default=0
            ),
        },
        inputs=["data", "axes"],
        outputs=["reduced"],
    ),
    "ReduceProd": OpSchema(
        name="ReduceProd",
        domain="ai.onnx",
        version=18,
        attributes={
            "keepdims": OpAttribute(
                name="keepdims", type="int", required=False, default=1
            ),
            "noop_with_empty_axes": OpAttribute(
                name="noop_with_empty_axes", type="int", required=False, default=0
            ),
        },
        inputs=["data", "axes"],
        outputs=["reduced"],
    ),
    "ReduceSum": OpSchema(
        name="ReduceSum",
        domain="ai.onnx",
        version=13,
        attributes={
            "keepdims": OpAttribute(
                name="keepdims", type="int", required=False, default=1
            ),
            "noop_with_empty_axes": OpAttribute(
                name="noop_with_empty_axes", type="int", required=False, default=0
            ),
        },
        inputs=["data", "axes"],
        outputs=["reduced"],
    ),
    "ReduceSumSquare": OpSchema(
        name="ReduceSumSquare",
        domain="ai.onnx",
        version=18,
        attributes={
            "keepdims": OpAttribute(
                name="keepdims", type="int", required=False, default=1
            ),
            "noop_with_empty_axes": OpAttribute(
                name="noop_with_empty_axes", type="int", required=False, default=0
            ),
        },
        inputs=["data", "axes"],
        outputs=["reduced"],
    ),
    "RegexFullMatch": OpSchema(
        name="RegexFullMatch",
        domain="ai.onnx",
        version=20,
        attributes={
            "pattern": OpAttribute(
                name="pattern", type="str", required=False, default=None
            ),
        },
        inputs=["X"],
        outputs=["Y"],
    ),
    "Relu": OpSchema(
        name="Relu",
        domain="ai.onnx",
        version=14,
        attributes={},
        inputs=["X"],
        outputs=["Y"],
    ),
    "Reshape": OpSchema(
        name="Reshape",
        domain="ai.onnx",
        version=25,
        attributes={
            "allowzero": OpAttribute(
                name="allowzero", type="int", required=False, default=0
            ),
        },
        inputs=["data", "shape"],
        outputs=["reshaped"],
    ),
    "Resize": OpSchema(
        name="Resize",
        domain="ai.onnx",
        version=19,
        attributes={
            "antialias": OpAttribute(
                name="antialias", type="int", required=False, default=0
            ),
            "axes": OpAttribute(
                name="axes", type="List[int]", required=False, default=None
            ),
            "coordinate_transformation_mode": OpAttribute(
                name="coordinate_transformation_mode",
                type="str",
                required=False,
                default="half_pixel",
            ),
            "cubic_coeff_a": OpAttribute(
                name="cubic_coeff_a", type="float", required=False, default=-0.75
            ),
            "exclude_outside": OpAttribute(
                name="exclude_outside", type="int", required=False, default=0
            ),
            "extrapolation_value": OpAttribute(
                name="extrapolation_value", type="float", required=False, default=0.0
            ),
            "keep_aspect_ratio_policy": OpAttribute(
                name="keep_aspect_ratio_policy",
                type="str",
                required=False,
                default="stretch",
            ),
            "mode": OpAttribute(
                name="mode", type="str", required=False, default="nearest"
            ),
            "nearest_mode": OpAttribute(
                name="nearest_mode",
                type="str",
                required=False,
                default="round_prefer_floor",
            ),
        },
        inputs=["X", "roi", "scales", "sizes"],
        outputs=["Y"],
    ),
    "ReverseSequence": OpSchema(
        name="ReverseSequence",
        domain="ai.onnx",
        version=10,
        attributes={
            "batch_axis": OpAttribute(
                name="batch_axis", type="int", required=False, default=1
            ),
            "time_axis": OpAttribute(
                name="time_axis", type="int", required=False, default=0
            ),
        },
        inputs=["input", "sequence_lens"],
        outputs=["Y"],
    ),
    "RoiAlign": OpSchema(
        name="RoiAlign",
        domain="ai.onnx",
        version=22,
        attributes={
            "coordinate_transformation_mode": OpAttribute(
                name="coordinate_transformation_mode",
                type="str",
                required=False,
                default="half_pixel",
            ),
            "mode": OpAttribute(name="mode", type="str", required=False, default="avg"),
            "output_height": OpAttribute(
                name="output_height", type="int", required=False, default=1
            ),
            "output_width": OpAttribute(
                name="output_width", type="int", required=False, default=1
            ),
            "sampling_ratio": OpAttribute(
                name="sampling_ratio", type="int", required=False, default=0
            ),
            "spatial_scale": OpAttribute(
                name="spatial_scale", type="float", required=False, default=1.0
            ),
        },
        inputs=["X", "rois", "batch_indices"],
        outputs=["Y"],
    ),
    "RotaryEmbedding": OpSchema(
        name="RotaryEmbedding",
        domain="ai.onnx",
        version=23,
        attributes={
            "interleaved": OpAttribute(
                name="interleaved", type="int", required=False, default=0
            ),
            "num_heads": OpAttribute(
                name="num_heads", type="int", required=False, default=None
            ),
            "rotary_embedding_dim": OpAttribute(
                name="rotary_embedding_dim", type="int", required=False, default=0
            ),
        },
        inputs=["X", "cos_cache", "sin_cache", "position_ids"],
        outputs=["Y"],
    ),
    "Round": OpSchema(
        name="Round",
        domain="ai.onnx",
        version=22,
        attributes={},
        inputs=["X"],
        outputs=["Y"],
    ),
    "STFT": OpSchema(
        name="STFT",
        domain="ai.onnx",
        version=17,
        attributes={
            "onesided": OpAttribute(
                name="onesided", type="int", required=False, default=1
            ),
        },
        inputs=["signal", "frame_step", "window", "frame_length"],
        outputs=["output"],
    ),
    "Scan": OpSchema(
        name="Scan",
        domain="ai.onnx",
        version=25,
        attributes={
            "body": OpAttribute(name="body", type="Any", required=True, default=None),
            "num_scan_inputs": OpAttribute(
                name="num_scan_inputs", type="int", required=True, default=None
            ),
            "scan_input_axes": OpAttribute(
                name="scan_input_axes", type="List[int]", required=False, default=None
            ),
            "scan_input_directions": OpAttribute(
                name="scan_input_directions",
                type="List[int]",
                required=False,
                default=None,
            ),
            "scan_output_axes": OpAttribute(
                name="scan_output_axes", type="List[int]", required=False, default=None
            ),
            "scan_output_directions": OpAttribute(
                name="scan_output_directions",
                type="List[int]",
                required=False,
                default=None,
            ),
        },
        inputs=["initial_state_and_scan_inputs"],
        outputs=["final_state_and_scan_outputs"],
    ),
    "ScatterElements": OpSchema(
        name="ScatterElements",
        domain="ai.onnx",
        version=18,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=0),
            "reduction": OpAttribute(
                name="reduction", type="str", required=False, default="none"
            ),
        },
        inputs=["data", "indices", "updates"],
        outputs=["output"],
    ),
    "ScatterND": OpSchema(
        name="ScatterND",
        domain="ai.onnx",
        version=18,
        attributes={
            "reduction": OpAttribute(
                name="reduction", type="str", required=False, default="none"
            ),
        },
        inputs=["data", "indices", "updates"],
        outputs=["output"],
    ),
    "Selu": OpSchema(
        name="Selu",
        domain="ai.onnx",
        version=22,
        attributes={
            "alpha": OpAttribute(
                name="alpha", type="float", required=False, default=1.67326
            ),
            "gamma": OpAttribute(
                name="gamma", type="float", required=False, default=1.0507
            ),
        },
        inputs=["X"],
        outputs=["Y"],
    ),
    "SequenceAt": OpSchema(
        name="SequenceAt",
        domain="ai.onnx",
        version=11,
        attributes={},
        inputs=["input_sequence", "position"],
        outputs=["tensor"],
    ),
    "SequenceConstruct": OpSchema(
        name="SequenceConstruct",
        domain="ai.onnx",
        version=11,
        attributes={},
        inputs=["inputs"],
        outputs=["output_sequence"],
    ),
    "SequenceEmpty": OpSchema(
        name="SequenceEmpty",
        domain="ai.onnx",
        version=11,
        attributes={
            "dtype": OpAttribute(
                name="dtype", type="int", required=False, default=None
            ),
        },
        inputs=["output"],
        outputs=["output"],
    ),
    "SequenceErase": OpSchema(
        name="SequenceErase",
        domain="ai.onnx",
        version=11,
        attributes={},
        inputs=["input_sequence", "position"],
        outputs=["output_sequence"],
    ),
    "SequenceInsert": OpSchema(
        name="SequenceInsert",
        domain="ai.onnx",
        version=11,
        attributes={},
        inputs=["input_sequence", "tensor", "position"],
        outputs=["output_sequence"],
    ),
    "SequenceLength": OpSchema(
        name="SequenceLength",
        domain="ai.onnx",
        version=11,
        attributes={},
        inputs=["input_sequence"],
        outputs=["length"],
    ),
    "SequenceMap": OpSchema(
        name="SequenceMap",
        domain="ai.onnx",
        version=17,
        attributes={
            "body": OpAttribute(name="body", type="Any", required=True, default=None),
        },
        inputs=["input_sequence", "additional_inputs"],
        outputs=["out_sequence"],
    ),
    "Shape": OpSchema(
        name="Shape",
        domain="ai.onnx",
        version=25,
        attributes={
            "end": OpAttribute(name="end", type="int", required=False, default=None),
            "start": OpAttribute(name="start", type="int", required=False, default=0),
        },
        inputs=["data"],
        outputs=["shape"],
    ),
    "Shrink": OpSchema(
        name="Shrink",
        domain="ai.onnx",
        version=9,
        attributes={
            "bias": OpAttribute(name="bias", type="float", required=False, default=0.0),
            "lambd": OpAttribute(
                name="lambd", type="float", required=False, default=0.5
            ),
        },
        inputs=["input"],
        outputs=["output"],
    ),
    "Sigmoid": OpSchema(
        name="Sigmoid",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["X"],
        outputs=["Y"],
    ),
    "Sign": OpSchema(
        name="Sign",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "Sin": OpSchema(
        name="Sin",
        domain="ai.onnx",
        version=22,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "Sinh": OpSchema(
        name="Sinh",
        domain="ai.onnx",
        version=22,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "Size": OpSchema(
        name="Size",
        domain="ai.onnx",
        version=25,
        attributes={},
        inputs=["data"],
        outputs=["size"],
    ),
    "Slice": OpSchema(
        name="Slice",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["data", "starts", "ends", "axes", "steps"],
        outputs=["output"],
    ),
    "Softmax": OpSchema(
        name="Softmax",
        domain="ai.onnx",
        version=13,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=-1),
        },
        inputs=["input"],
        outputs=["output"],
    ),
    "SoftmaxCrossEntropyLoss": OpSchema(
        name="SoftmaxCrossEntropyLoss",
        domain="ai.onnx",
        version=13,
        attributes={
            "ignore_index": OpAttribute(
                name="ignore_index", type="int", required=False, default=None
            ),
            "reduction": OpAttribute(
                name="reduction", type="str", required=False, default="mean"
            ),
        },
        inputs=["scores", "labels", "weights"],
        outputs=["output", "log_prob"],
    ),
    "Softplus": OpSchema(
        name="Softplus",
        domain="ai.onnx",
        version=22,
        attributes={},
        inputs=["X"],
        outputs=["Y"],
    ),
    "Softsign": OpSchema(
        name="Softsign",
        domain="ai.onnx",
        version=22,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "SpaceToDepth": OpSchema(
        name="SpaceToDepth",
        domain="ai.onnx",
        version=13,
        attributes={
            "blocksize": OpAttribute(
                name="blocksize", type="int", required=True, default=None
            ),
        },
        inputs=["input"],
        outputs=["output"],
    ),
    "Split": OpSchema(
        name="Split",
        domain="ai.onnx",
        version=18,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=0),
            "num_outputs": OpAttribute(
                name="num_outputs", type="int", required=False, default=None
            ),
        },
        inputs=["input", "split"],
        outputs=["outputs"],
    ),
    "SplitToSequence": OpSchema(
        name="SplitToSequence",
        domain="ai.onnx",
        version=24,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=0),
            "keepdims": OpAttribute(
                name="keepdims", type="int", required=False, default=1
            ),
        },
        inputs=["input", "split"],
        outputs=["output_sequence"],
    ),
    "Sqrt": OpSchema(
        name="Sqrt",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["X"],
        outputs=["Y"],
    ),
    "Squeeze": OpSchema(
        name="Squeeze",
        domain="ai.onnx",
        version=25,
        attributes={},
        inputs=["data", "axes"],
        outputs=["squeezed"],
    ),
    "StringConcat": OpSchema(
        name="StringConcat",
        domain="ai.onnx",
        version=20,
        attributes={},
        inputs=["X", "Y"],
        outputs=["Z"],
    ),
    "StringNormalizer": OpSchema(
        name="StringNormalizer",
        domain="ai.onnx",
        version=10,
        attributes={
            "case_change_action": OpAttribute(
                name="case_change_action", type="str", required=False, default="NONE"
            ),
            "is_case_sensitive": OpAttribute(
                name="is_case_sensitive", type="int", required=False, default=0
            ),
            "locale": OpAttribute(
                name="locale", type="str", required=False, default=None
            ),
            "stopwords": OpAttribute(
                name="stopwords", type="List[str]", required=False, default=None
            ),
        },
        inputs=["X"],
        outputs=["Y"],
    ),
    "StringSplit": OpSchema(
        name="StringSplit",
        domain="ai.onnx",
        version=20,
        attributes={
            "delimiter": OpAttribute(
                name="delimiter", type="str", required=False, default=None
            ),
            "maxsplit": OpAttribute(
                name="maxsplit", type="int", required=False, default=None
            ),
        },
        inputs=["X"],
        outputs=["Y", "Z"],
    ),
    "Sub": OpSchema(
        name="Sub",
        domain="ai.onnx",
        version=14,
        attributes={},
        inputs=["A", "B"],
        outputs=["C"],
    ),
    "Sum": OpSchema(
        name="Sum",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["data_0"],
        outputs=["sum"],
    ),
    "Swish": OpSchema(
        name="Swish",
        domain="ai.onnx",
        version=24,
        attributes={
            "alpha": OpAttribute(
                name="alpha", type="float", required=False, default=1.0
            ),
        },
        inputs=["X"],
        outputs=["Y"],
    ),
    "Tan": OpSchema(
        name="Tan",
        domain="ai.onnx",
        version=22,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "Tanh": OpSchema(
        name="Tanh",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["input"],
        outputs=["output"],
    ),
    "TensorScatter": OpSchema(
        name="TensorScatter",
        domain="ai.onnx",
        version=24,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=-2),
            "mode": OpAttribute(
                name="mode", type="str", required=False, default="linear"
            ),
        },
        inputs=["past_cache", "update", "write_indices"],
        outputs=["present_cache"],
    ),
    "TfIdfVectorizer": OpSchema(
        name="TfIdfVectorizer",
        domain="ai.onnx",
        version=9,
        attributes={
            "max_gram_length": OpAttribute(
                name="max_gram_length", type="int", required=True, default=None
            ),
            "max_skip_count": OpAttribute(
                name="max_skip_count", type="int", required=True, default=None
            ),
            "min_gram_length": OpAttribute(
                name="min_gram_length", type="int", required=True, default=None
            ),
            "mode": OpAttribute(name="mode", type="str", required=True, default=None),
            "ngram_counts": OpAttribute(
                name="ngram_counts", type="List[int]", required=True, default=None
            ),
            "ngram_indexes": OpAttribute(
                name="ngram_indexes", type="List[int]", required=True, default=None
            ),
            "pool_int64s": OpAttribute(
                name="pool_int64s", type="List[int]", required=False, default=None
            ),
            "pool_strings": OpAttribute(
                name="pool_strings", type="List[str]", required=False, default=None
            ),
            "weights": OpAttribute(
                name="weights", type="List[float]", required=False, default=None
            ),
        },
        inputs=["X"],
        outputs=["Y"],
    ),
    "ThresholdedRelu": OpSchema(
        name="ThresholdedRelu",
        domain="ai.onnx",
        version=22,
        attributes={
            "alpha": OpAttribute(
                name="alpha", type="float", required=False, default=1.0
            ),
        },
        inputs=["X"],
        outputs=["Y"],
    ),
    "Tile": OpSchema(
        name="Tile",
        domain="ai.onnx",
        version=13,
        attributes={},
        inputs=["input", "repeats"],
        outputs=["output"],
    ),
    "TopK": OpSchema(
        name="TopK",
        domain="ai.onnx",
        version=24,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=-1),
            "largest": OpAttribute(
                name="largest", type="int", required=False, default=1
            ),
            "sorted": OpAttribute(name="sorted", type="int", required=False, default=1),
        },
        inputs=["X", "K"],
        outputs=["Values", "Indices"],
    ),
    "Transpose": OpSchema(
        name="Transpose",
        domain="ai.onnx",
        version=25,
        attributes={
            "perm": OpAttribute(
                name="perm", type="List[int]", required=False, default=None
            ),
        },
        inputs=["data"],
        outputs=["transposed"],
    ),
    "Trilu": OpSchema(
        name="Trilu",
        domain="ai.onnx",
        version=14,
        attributes={
            "upper": OpAttribute(name="upper", type="int", required=False, default=1),
        },
        inputs=["input", "k"],
        outputs=["output"],
    ),
    "Unique": OpSchema(
        name="Unique",
        domain="ai.onnx",
        version=11,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=False, default=None),
            "sorted": OpAttribute(name="sorted", type="int", required=False, default=1),
        },
        inputs=["X"],
        outputs=["Y", "indices", "inverse_indices", "counts"],
    ),
    "Unsqueeze": OpSchema(
        name="Unsqueeze",
        domain="ai.onnx",
        version=25,
        attributes={},
        inputs=["data", "axes"],
        outputs=["expanded"],
    ),
    "Where": OpSchema(
        name="Where",
        domain="ai.onnx",
        version=16,
        attributes={},
        inputs=["condition", "X", "Y"],
        outputs=["output"],
    ),
    "Xor": OpSchema(
        name="Xor",
        domain="ai.onnx",
        version=7,
        attributes={
            "prob_mod": OpAttribute(
                name="prob_mod", type="Any", required=False, default=None
            ),
            "scale": OpAttribute(
                name="scale", type="float", required=False, default=None
            ),
            "score_mod": OpAttribute(
                name="score_mod", type="Any", required=False, default=None
            ),
            "softmax_precision": OpAttribute(
                name="softmax_precision", type="int", required=False, default=None
            ),
        },
        inputs=["A", "B"],
        outputs=["C"],
    ),
    "ai.onnx.preview.training.Adagrad": OpSchema(
        name="ai.onnx.preview.training.Adagrad",
        domain="ai.onnx",
        version=1,
        attributes={
            "decay_factor": OpAttribute(
                name="decay_factor", type="float", required=False, default=0.0
            ),
            "epsilon": OpAttribute(
                name="epsilon", type="float", required=False, default=0.0
            ),
            "norm_coefficient": OpAttribute(
                name="norm_coefficient", type="float", required=False, default=0.0
            ),
        },
        inputs=["R", "T", "inputs"],
        outputs=["outputs"],
    ),
    "ai.onnx.preview.training.Adam": OpSchema(
        name="ai.onnx.preview.training.Adam",
        domain="ai.onnx",
        version=1,
        attributes={
            "alpha": OpAttribute(
                name="alpha", type="float", required=False, default=0.9
            ),
            "beta": OpAttribute(
                name="beta", type="float", required=False, default=0.999
            ),
            "epsilon": OpAttribute(
                name="epsilon", type="float", required=False, default=0.0
            ),
            "norm_coefficient": OpAttribute(
                name="norm_coefficient", type="float", required=False, default=0.0
            ),
            "norm_coefficient_post": OpAttribute(
                name="norm_coefficient_post", type="float", required=False, default=0.0
            ),
        },
        inputs=["R", "T", "inputs"],
        outputs=["outputs"],
    ),
    "ai.onnx.preview.training.Gradient": OpSchema(
        name="ai.onnx.preview.training.Gradient",
        domain="ai.onnx",
        version=1,
        attributes={
            "xs": OpAttribute(name="xs", type="List[str]", required=True, default=None),
            "y": OpAttribute(name="y", type="str", required=True, default=None),
            "zs": OpAttribute(
                name="zs", type="List[str]", required=False, default=None
            ),
        },
        inputs=["Inputs"],
        outputs=["Outputs"],
    ),
    "ai.onnx.preview.training.Momentum": OpSchema(
        name="ai.onnx.preview.training.Momentum",
        domain="ai.onnx",
        version=1,
        attributes={
            "alpha": OpAttribute(
                name="alpha", type="float", required=True, default=None
            ),
            "beta": OpAttribute(name="beta", type="float", required=True, default=None),
            "mode": OpAttribute(name="mode", type="str", required=True, default=None),
            "norm_coefficient": OpAttribute(
                name="norm_coefficient", type="float", required=True, default=None
            ),
        },
        inputs=["R", "T", "inputs"],
        outputs=["outputs"],
    ),
}
