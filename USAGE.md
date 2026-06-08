# Usage Guide: Measuring Compliance & Compatibility

The `ml-switcheroo-ir` package provides a built-in static analysis CLI to measure how compatible your implementations are with the central Intermediate Representation (IR), Framework Adapter protocols, and the canonical ONNX dialect.

This is especially critical when building zero-dependency pure-Python replicas of major frameworks (like Keras 3 or PyTorch). You can use this tool to track your progress and guarantee you have implemented every required API and operator.

## The `compliance` Command

The core utility for tracking progress is the `compliance` subcommand. It uses Python's Abstract Syntax Tree (AST) to scan your code without executing it, meaning it doesn't require any heavy ML frameworks to be installed to run the analysis.

### 1. Measuring IR Compatibility (Frontends & Backends)

If you are writing a frontend parser or a backend compiler, `ml-switcheroo-ir` requires you to implement specific protocols (`GraphFrontend` and `CompilerBackend`).

To check if a module correctly implements the IR contract, point the tool at your file:

```bash
ml-switcheroo-ir compliance ~/repos/ml-switcheroo/src/ml_switcheroo/compiler/backends/python.py
```

**Output:**
```text
Compliance Report
=================

IR Compliance (CompilerBackend / GraphFrontend):
Target     Compliance %    Reqs Met    Status
---------  --------------  ----------  --------
python.py  100%            5/5         PASS
```

**How it is scored:**
The tool scores IR implementations out of 5 core requirements, outputting a percentage:
1.  **Module Loadable:** The file is a valid Python module.
2.  **Class Defined:** A class is defined in the file.
3.  **Inheritance:** The class explicitly inherits from `CompilerBackend` or `GraphFrontend`.
4.  **Method Present:** The required entrypoint (`compile` or `parse_to_graph`) is defined.
5.  **Signature Correct:** The entrypoint accepts the expected arguments (`graph` or `code`).

### 2. Measuring Framework Adapter Compliance (e.g., Keras 3)

If you are writing a framework adapter (a translation layer between a specific ML framework like Keras 3 and the IR), the tool checks for the structural requirements of an adapter.

```bash
# You can scan an entire directory
ml-switcheroo-ir compliance ~/repos/ml-switcheroo/src/ml_switcheroo/frameworks/
```

**Output:**
```text
Compliance Report
=================

FrameworkAdapter Compliance:
Target                  Compliance %    Reqs Met    Status
----------------------  --------------  ----------  --------
frameworks (Directory)  100%            4/4         PASS

DIALECT Compliance (ONNX Superset, 204 total ops):
Target                  Compliance %    Ops Implemented    Status      Missing Ops
----------------------  --------------  -----------------  --------  -------------
frameworks (Directory)  56.9%           116/204            FAIL                 88
```

**How it is scored:**
Framework adapters are scored out of 4 requirements:
1.  **Module Loadable:** The file is a valid Python module.
2.  **Registration:** A class is present with `"Adapter"` or `"Importer"` in its name, or decorated with `@register_framework`.
3.  **Convert Method:** The class implements an entrypoint like `convert()`, `import_func()`, or `parse()`.
4.  **Definitions Method:** The class implements a definitions loader like `definitions()` or `import_jaxpr()`.

### 3. Measuring API / Dialect Coverage (The Zero-Dependency Goal)

The most powerful feature of the compliance tool is measuring exactly how many canonical operators your pure-Python framework replica actually supports.

The `ml-switcheroo-ir` establishes a dialect based on the ONNX standard (a superset of hundreds of operators). To see how close your Keras 3 implementation is to 100% coverage:

```bash
ml-switcheroo-ir compliance ~/repos/ml-switcheroo/onnx9000/packages/python/onnx9000-jax/src/onnx9000
```

**Output:**
```text
Compliance Report
=================

FrameworkAdapter Compliance:
Target                Compliance %    Reqs Met    Status
--------------------  --------------  ----------  --------
onnx9000 (Directory)  100%            4/4         PASS

DIALECT Compliance (ONNX Superset, 204 total ops):
Target                Compliance %    Ops Implemented    Status      Missing Ops
--------------------  --------------  -----------------  --------  -------------
onnx9000 (Directory)  3.9%            8/204              FAIL                196
```

**How it is scored:**
The tool scans your code looking for:
*   Functions named after ONNX operators (e.g., `def Gemm(...):`)
*   Dictionary keys mapping to ONNX operators (e.g., `{"Conv": ...}`)
*   JSON definition files dynamically loaded via `load_definitions("keras3")`
*   Functions decorated with operation mappers like `@register_op("framework", "Add")`
*   Explicitly passed `.json` operation definition files (e.g., `jax.json`)

It aggregates all discovered operations across all provided files and compares the unique operators found against the total operator count in the active IR registry, outputting a percentage (e.g., `20% (35/175)`).

#### Finding the Missing Pieces (`--verbose` and `--mapping`)

When replicating a framework, knowing *what* is missing is more important than knowing the percentage.

Use the `--verbose` (or `-v`) flag to output a complete checklist of missing operations.
If you also provide a framework definition file via `--mapping` (e.g. `jax.json`), the tool will generate a structured markdown table with the target API paths, signatures, and docstrings. This table is perfect for feeding to an LLM or sharing with a human to quickly implement missing features!

```bash
ml-switcheroo-ir compliance ~/repos/ml-switcheroo/onnx9000/packages/python/onnx9000-jax/src/onnx9000 \
  --mapping ~/repos/ml-switcheroo/src/ml_switcheroo/frameworks/definitions/jax.json --verbose
```

**Output:**
```text
Compliance Report
=================

FrameworkAdapter Compliance:
Target                Compliance %    Reqs Met    Status
--------------------  --------------  ----------  --------
onnx9000 (Directory)  100%            4/4         PASS

DIALECT Compliance (ONNX Superset, 204 total ops):
Target                Compliance %    Ops Implemented    Status      Missing Ops
--------------------  --------------  -----------------  --------  -------------
onnx9000 (Directory)  3.9%            8/204              FAIL                196

Verbose Missing Operations Report
=================================

### onnx9000 (Directory) (Mapped API targets)
| Implemented   | Framework         | Namespace         | Symbol               | FQN                          | Signature   | Docstring               |
|---------------|-------------------|-------------------|----------------------|------------------------------|-------------|-------------------------|
| [ ]           | jax               | jax.numpy         | abs                  | jax.numpy.abs                | Unknown     | No docstring available. |
| [ ]           | jax               | jax.numpy         | arccos               | jax.numpy.arccos             | Unknown     | No docstring available. |
| [ ]           | jax               | jax.numpy         | arccosh              | jax.numpy.arccosh            | Unknown     | No docstring available. |
| [ ]           | optax             | optax             | adagrad              | optax.adagrad                | Unknown     | No docstring available. |
| [ ]           | optax             | optax             | adam                 | optax.adam                   | Unknown     | No docstring available. |
...
```

This generates a list you can drop directly into a GitHub Issue or Pull Request to track the remaining work required to achieve 100% API coverage.

## Examples

The `compliance` command accepts multiple files and directories as targets simultaneously. It will intelligently aggregate the results into a unified report.

### Checking a Core Framework Adapter

To check the compliance of an adapter within the `ml-switcheroo` repository (like JAX), you can provide multiple paths to scan both the Python implementation and its JSON definitions together:

```bash
ml-switcheroo-ir compliance ~/repos/ml-switcheroo/src/ml_switcheroo/frameworks/jax.py ~/repos/ml-switcheroo/src/ml_switcheroo/frameworks/definitions/jax.json
```

**Output:**
```text
Compliance Report
=================

FrameworkAdapter Compliance:
Target                Compliance %    Reqs Met    Status
--------------------  --------------  ----------  --------
Multiple Targets (2)  100%            4/4         PASS

DIALECT Compliance (ONNX Superset, 204 total ops):
Target                Compliance %    Ops Implemented    Status      Missing Ops
--------------------  --------------  -----------------  --------  -------------
Multiple Targets (2)  50.0%           102/204            FAIL                102
```

### Checking an External Project

You can run compliance checks on a completely unrelated project with its own build system (`pyproject.toml`), passing the entire source directory to get a single unified score:

```bash
ml-switcheroo-ir compliance ~/repos/ml-switcheroo/onnx9000/packages/python/onnx9000-core/src/onnx9000
```

**Output:**
```text
Compliance Report
=================

DIALECT Compliance (ONNX Superset, 204 total ops):
Target                Compliance %    Ops Implemented    Status      Missing Ops
--------------------  --------------  -----------------  --------  -------------
onnx9000 (Directory)  11.8%           24/204             FAIL                180
```
