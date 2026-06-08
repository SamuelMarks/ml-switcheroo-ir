# ONNX Dialect Mapping Guide

This document outlines how specific framework concepts map to the canonical `ml-switcheroo-ir` ONNX dialect. This ensures consistent translation across different ingestion frontends (e.g., PyTorch, TensorFlow).

## 1. Linear/Dense Layers

The `Linear` operation (e.g., `torch.nn.Linear`) is mapped to the `Gemm` (General Matrix Multiplication) ONNX operator.

**ONNX `Gemm` Equation:**
`Y = alpha * A * B + beta * C`

### Mapping `torch.nn.Linear`

*   `A`: The input tensor `X`.
*   `B`: The weight tensor `W`. Since PyTorch typically stores weights transposed relative to `Gemm`'s expectation, you may need to set `transB=1` in the metadata, or transpose the weights during ingestion.
*   `C`: The bias tensor `B`.

**Example IR Node:**

```json
{
  "id": "linear1",
  "kind": "Gemm",
  "domain": "ai.onnx",
  "version": 11,
  "metadata": {
    "alpha": 1.0,
    "beta": 1.0,
    "transB": 1
  }
}
```

*Note: If no bias is present (`bias=False`), the `C` input should be omitted from the graph edges for this node.*

## 2. Convolutional Layers

`Conv1d`, `Conv2d`, and `Conv3d` all map to the generic ONNX `Conv` operator. The dimensionality is inferred from the `kernel_shape` attribute.

### Mapping `torch.nn.Conv2d`

*   `kernel_shape`: Mapped directly from `kernel_size` (e.g., `[3, 3]`).
*   `strides`: Mapped directly from `stride` (e.g., `[1, 1]`).
*   `pads`: PyTorch typically uses a single padding value or a tuple. This must be expanded to the ONNX `pads` format: `[x1_begin, x2_begin, ..., x1_end, x2_end, ...]`. For a 2D padding of `1`, this becomes `[1, 1, 1, 1]`.
*   `group`: Mapped directly from `groups`.

## 3. Activation Functions

Common activations map directly to their ONNX counterparts (e.g., `Relu`, `Sigmoid`, `Tanh`).

For parameterized activations like `LeakyReLU`, use the ONNX `LeakyRelu` operator and specify the `alpha` attribute.

## 4. Custom Operations

If a model uses an operation not present in the standard `ai.onnx` domain (e.g., a highly specialized fused kernel or FlashAttention), use a custom domain.

**Example Custom Node:**

```json
{
  "id": "flash_attn1",
  "kind": "FlashAttention",
  "domain": "ml.switcheroo.custom",
  "version": 1,
  "metadata": {
    "causal": true,
    "dropout_p": 0.1
  }
}
```

You must provide a custom ops schema JSON file to the CLI `validate` command using the `--custom-ops` flag to validate these nodes.
