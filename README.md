# zero-* Engine Core

> **Note:** This repository serves as the core execution engine and abstract representation framework for the `zero-*` ecosystem.

# ml-switcheroo-ir

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/ml-switcheroo-ir/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/ml-switcheroo-ir/actions)
[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)

ml-switcheroo-ir

=======


> The universal, dependency-free intermediate representation for the `ml-switcheroo` ecosystem.

`ml-switcheroo-ir` provides the core language-agnostic data structures and interface protocols used to translate neural network architectures between different Deep Learning frameworks (PyTorch, TensorFlow, JAX) and hardware-level formats (TensorRT, XLA, MLIR).

By encapsulating these structures into a standalone package, third-party plugin authors can build complete frontends (parsers) or backends (code generators) without pulling in heavy abstract syntax trees or third-party framework dependencies.

## Why does this exist?

The machine learning ecosystem suffers from the **N-to-M translation problem**. If you have `N` frameworks and `M` hardware targets, writing direct converters requires `N * M` implementations.

`ml-switcheroo` solves this by introducing a central Intermediate Representation (IR). Frontends translate `N` frameworks into this IR, and backends translate this IR into `M` hardware targets, reducing the complexity to `N + M`.

`ml-switcheroo-ir` *is* that central contract.

## Key Features

- **Zero Dependency:** Built entirely with standard Python libraries. Safe to integrate anywhere.
- **ONNX Canonical Dialect:** Standardizes operator kinds and metadata against the rigorous Open Neural Network Exchange (ONNX) specification.
- **Strict Validation:** Includes a built-in schema validator to ensure generated IR graphs perfectly conform to operator signatures, required attributes, and type rules.
- **Superset Extensibility:** First-class support for custom, cutting-edge, or framework-specific operations (e.g., FlashAttention) not yet in the ONNX standard.
- **Developer Tooling:** Bundles a powerful CLI for topology sorting, schema validation, and operator discovery.

## Installation

```bash
pip install ml-switcheroo-ir
```

## Quick Start

### For Frontend Developers
Implement the `GraphFrontend` protocol to parse your domain-specific code into a `LogicalGraph`.

```python
from typing import Any
from ml_switcheroo_ir import LogicalGraph, LogicalNode, GraphFrontend

class MyPyTorchFrontend(GraphFrontend):
    def parse_to_graph(self, model_input: Any) -> LogicalGraph:
        # 1. Parse your PyTorch model
        # 2. Map operations to the ONNX dialect
        node = LogicalNode(
            id="linear1",
            kind="Gemm", # ONNX dialect
            domain="ai.onnx",
            version=11,
            metadata={"alpha": 1.0, "beta": 1.0, "transB": 1}
        )
        # 3. Construct and return the graph
        return LogicalGraph(nodes=[node], edges=[])
```

### For Backend Developers
Implement the `CompilerBackend` protocol to consume a `LogicalGraph` and emit your target payload.

```python
from ml_switcheroo_ir import LogicalGraph, CompilerBackend

class MyTensorRTBackend(CompilerBackend):
    def compile(self, graph: LogicalGraph) -> str:
        for node in graph.nodes:
            if node.kind == "Gemm":
                # Emit TensorRT code for matrix multiplication
                pass
        return "..."
```

## CLI Usage

The package includes the `ml-switcheroo-ir` command-line utility for CI/CD and debugging workflows.

### Validation

Ensure a serialized JSON graph adheres strictly to the ONNX schema:

```bash
ml-switcheroo-ir validate graph.json
```

Use `--strict` to treat warnings as errors, and `--custom-ops` to inject your own operator schemas:

```bash
ml-switcheroo-ir validate graph.json --strict --custom-ops my_custom_ops.json
```

### Operator Discovery

Search the ONNX registry to understand what attributes an operator requires:

```bash
# List all operators in the default ai.onnx domain
ml-switcheroo-ir list-ops --domain ai.onnx

# Search specifically for Convolution operators
ml-switcheroo-ir list-ops --search "^Conv"
```

### Graph Utilities

Topologically sort a graph to ensure nodes are evaluated in the correct dependency order:

```bash
ml-switcheroo-ir toposort graph.json
```

## Extensibility (Custom Operations)

Machine Learning moves fast. While we use ONNX as our base, you are not restricted by it. You can define custom operations via a JSON schema:

```json
{
  "FlashAttention": {
    "domain": "ml.switcheroo.custom",
    "attributes": {
      "causal": {"type": "bool", "required": true},
      "dropout_p": {"type": "float", "required": false, "default": 0.0}
    },
    "inputs": ["query", "key", "value"],
    "outputs": ["output"]
  }
}
```

Load this schema during validation or compilation to seamlessly integrate novel architectures.

## Architecture

For deeper details on the IR design, schema generation, and data flow pipeline, please read the [ARCHITECTURE.md](ARCHITECTURE.md) document.

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
