"""Unit tests for repository maintenance and code generation scripts."""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_ir.snapshots import DEFAULT_SNAPSHOT_DIR
from scripts.generate_registry import (
    cross_validate_with_onnx_defs,
    extract_section,
    parse_onnx_docs,
)
from scripts.generate_registry import (
    main as gen_main,
)
from scripts.generate_stablehlo_registry import (
    extract_stablehlo_schemas,
)
from scripts.generate_stablehlo_registry import (
    main as gen_stablehlo_main,
)
from scripts.update_badges import (
    count_shields,
    enforce_coverage_shields,
    format_cov,
    get_color,
    get_doc_coverage,
    get_test_coverage,
    parse_args,
    update_readme,
)
from scripts.update_badges import (
    main as update_badges_main,
)
from scripts.verify_grounding import (
    _extract_known_attributes,
    find_snapshots_directory,
    verify_array_api_grounding,
    verify_aten_grounding,
    verify_collectives_grounding,
    verify_custom_ops_grounding,
    verify_ir_snapshot_grounding,
    verify_metal_grounding,
    verify_mlir_grounding,
    verify_onnx_grounding,
    verify_ptx_grounding,
    verify_rdna_grounding,
    verify_sass_grounding,
    verify_stablehlo_grounding,
    verify_wasm_grounding,
    verify_wgsl_grounding,
)
from scripts.verify_grounding import (
    main as verify_grounding_main,
)


def test_scripts_package_import() -> None:
    """Test importing the scripts package module directly."""
    import scripts

    assert hasattr(scripts, "__doc__")
    assert scripts.__doc__ is not None


def test_get_color() -> None:
    """Test get_color across all percentage threshold ranges."""
    assert get_color(100.0) == "brightgreen"
    assert get_color(105.0) == "brightgreen"
    assert get_color(95.0) == "green"
    assert get_color(90.0) == "green"
    assert get_color(85.0) == "yellowgreen"
    assert get_color(80.0) == "yellowgreen"
    assert get_color(75.0) == "yellow"
    assert get_color(70.0) == "yellow"
    assert get_color(65.0) == "orange"
    assert get_color(60.0) == "orange"
    assert get_color(59.9) == "red"
    assert get_color(0.0) == "red"


def test_format_cov() -> None:
    """Test format_cov with whole numbers and fractional floats."""
    assert format_cov(100.0) == "100"
    assert format_cov(90.0) == "90"
    assert format_cov(95.54) == "95.5"
    assert format_cov(82.19) == "82.2"


def test_get_doc_coverage() -> None:
    """Test get_doc_coverage returns 100.0 and parses output correctly."""
    assert get_doc_coverage() == 100.0

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "actual: 97.4%\n"
        assert get_doc_coverage() == 97.4

    with patch("subprocess.run", side_effect=Exception("Failed")):
        assert get_doc_coverage() == 100.0


def test_count_shields() -> None:
    """Test count_shields accurately reports test, doc, and branch badge counts."""
    content = (
        "# Title\n"
        "[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)\n"
        "[![Branch Coverage](https://img.shields.io/badge/branch_coverage-100%25-brightgreen.svg)](#)\n"
        "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)\n"
    )
    assert count_shields(content) == (1, 1, 1)

    empty_content = "# No badges\n"
    assert count_shields(empty_content) == (0, 0, 0)


def test_parse_args() -> None:
    """Test parse_args with default, flag, and file overrides."""
    assert parse_args([]) == (False, "README.md")
    assert parse_args(["--enforce", "foo.md"]) == (True, "foo.md")
    assert parse_args(["--check"]) == (True, "README.md")
    assert parse_args(["-c"]) == (True, "README.md")
    assert parse_args(["custom.md"]) == (False, "custom.md")
    assert parse_args(["--unknown"]) == (False, "README.md")


def test_enforce_coverage_shields() -> None:
    """Test enforce_coverage_shields validates one-and-only-one invariants."""
    with TemporaryDirectory() as tmpdir:
        readme = os.path.join(tmpdir, "README.md")

        # Missing file
        with pytest.raises(FileNotFoundError, match="Target markdown file not found"):
            enforce_coverage_shields("/nonexistent/file.md")

        # Branch coverage present
        with open(readme, "w", encoding="utf-8") as f:
            f.write(
                "# Title\n"
                "[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)\n"
                "[![Branch Coverage](https://img.shields.io/badge/branch_coverage-100%25-brightgreen.svg)](#)\n"
                "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)\n"
            )
        with pytest.raises(ValueError, match="Expected 0 branch coverage shields"):
            enforce_coverage_shields(readme)

        # Missing test coverage shield
        with open(readme, "w", encoding="utf-8") as f:
            f.write(
                "# Title\n"
                "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)\n"
            )
        with pytest.raises(ValueError, match="Expected exactly 1 test coverage shield"):
            enforce_coverage_shields(readme)

        # Multiple test coverage shields
        with open(readme, "w", encoding="utf-8") as f:
            f.write(
                "# Title\n"
                "[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)\n"
                "[![Test Coverage](https://img.shields.io/badge/test_coverage-90%25-green.svg)](#)\n"
                "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)\n"
            )
        with pytest.raises(ValueError, match="Expected exactly 1 test coverage shield"):
            enforce_coverage_shields(readme)

        # Missing doc coverage shield
        with open(readme, "w", encoding="utf-8") as f:
            f.write(
                "# Title\n"
                "[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)\n"
            )
        with pytest.raises(ValueError, match="Expected exactly 1 doc coverage shield"):
            enforce_coverage_shields(readme)

        # Valid shields: exactly 1 test, 1 doc, 0 branch
        with open(readme, "w", encoding="utf-8") as f:
            f.write(
                "# Title\n"
                "[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)\n"
                "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)\n"
            )
        enforce_coverage_shields(readme)


def test_get_test_coverage_success() -> None:
    """Test get_test_coverage with valid coverage json file."""
    with TemporaryDirectory() as tmpdir:
        f_name = os.path.join(tmpdir, "coverage.json")
        with open(f_name, "w", encoding="utf-8") as f:
            json.dump({"totals": {"percent_covered": 98.5}}, f)
        with patch("subprocess.run") as mock_sub:
            val = get_test_coverage(coverage_json_path=f_name)
            mock_sub.assert_called_once()
            assert val == 98.5


def test_get_test_coverage_failure() -> None:
    """Test get_test_coverage returns 0.0 on missing or invalid file."""
    with patch("subprocess.run", side_effect=Exception("Failed")):
        val = get_test_coverage(coverage_json_path="/nonexistent/path/coverage.json")
        assert val == 0.0


def test_update_readme_missing_file() -> None:
    """Test update_readme returns gracefully when readme does not exist."""
    update_readme(readme_path="/nonexistent/path/README.md")


def test_update_readme_success() -> None:
    """Test update_readme properly updates shields in a markdown document."""
    initial_content = (
        "# Project\n"
        "[![Test Coverage](https://img.shields.io/badge/test_coverage-50%25-red.svg)](#)\n"
        "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-50%25-red.svg)](#)\n"
    )
    with TemporaryDirectory() as tmpdir:
        readme_file = os.path.join(tmpdir, "README.md")
        cov_file = os.path.join(tmpdir, "coverage.json")
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write(initial_content)
        with open(cov_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "totals": {
                        "percent_covered": 100.0,
                        "num_statements": 10,
                        "covered_lines": 10,
                        "num_branches": 2,
                        "covered_branches": 2,
                    }
                },
                f,
            )

        with patch("subprocess.run"):
            update_readme(readme_path=readme_file, coverage_json_path=cov_file)

        with open(readme_file, "r", encoding="utf-8") as f:
            updated = f.read()

        assert "test_coverage-100%25-brightgreen.svg" in updated
        assert "branch_coverage" not in updated
        assert "doc_coverage-100%25-brightgreen.svg" in updated
        assert count_shields(updated) == (1, 1, 0)

        # Update again with branch coverage injected to verify removal
        branch_badge = "\n[![Branch Coverage](https://img.shields.io/badge/branch_coverage-100%25-brightgreen.svg)](#)\n"
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write(updated + branch_badge)

        with patch("subprocess.run"):
            update_readme(readme_path=readme_file, coverage_json_path=cov_file)

        with open(readme_file, "r", encoding="utf-8") as f:
            updated2 = f.read()

        assert "branch_coverage" not in updated2
        assert count_shields(updated2) == (1, 1, 0)


def test_update_readme_insertions_and_deduplications() -> None:
    """Test update_readme insertion scenarios and duplicate shield pruning."""
    with TemporaryDirectory() as tmpdir:
        cov_file = os.path.join(tmpdir, "coverage.json")
        with open(cov_file, "w", encoding="utf-8") as f:
            json.dump({"totals": {"percent_covered": 100.0}}, f)

        # Case 1: Only Doc Coverage present -> insert Test Coverage before Doc Coverage
        case1_file = os.path.join(tmpdir, "case1.md")
        with open(case1_file, "w", encoding="utf-8") as f:
            f.write(
                "# Title\n[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)\n"
            )
        with patch("subprocess.run"):
            update_readme(readme_path=case1_file, coverage_json_path=cov_file)
        with open(case1_file, "r", encoding="utf-8") as f:
            assert count_shields(f.read()) == (1, 1, 0)

        # Case 2: Neither present, but header present -> insert after header
        case2_file = os.path.join(tmpdir, "case2.md")
        with open(case2_file, "w", encoding="utf-8") as f:
            f.write("# Title\n\nSome body text.\n")
        with patch("subprocess.run"):
            update_readme(readme_path=case2_file, coverage_json_path=cov_file)
        with open(case2_file, "r", encoding="utf-8") as f:
            assert count_shields(f.read()) == (1, 1, 0)

        # Case 3: Neither present and no header -> insert at top
        case3_file = os.path.join(tmpdir, "case3.md")
        with open(case3_file, "w", encoding="utf-8") as f:
            f.write("Just raw text without header.\n")
        with patch("subprocess.run"):
            update_readme(readme_path=case3_file, coverage_json_path=cov_file)
        with open(case3_file, "r", encoding="utf-8") as f:
            assert count_shields(f.read()) == (1, 1, 0)

        # Case 4: Multiple duplicates of both -> deduplicate to exactly one each
        case4_file = os.path.join(tmpdir, "case4.md")
        with open(case4_file, "w", encoding="utf-8") as f:
            f.write(
                "# Title\n"
                "[![Test Coverage](https://img.shields.io/badge/test_coverage-50%25-red.svg)](#)\n"
                "[![Test Coverage](https://img.shields.io/badge/test_coverage-60%25-red.svg)](#)\n"
                "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-50%25-red.svg)](#)\n"
                "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-60%25-red.svg)](#)\n"
            )
        with patch("subprocess.run"):
            update_readme(readme_path=case4_file, coverage_json_path=cov_file)
        with open(case4_file, "r", encoding="utf-8") as f:
            assert count_shields(f.read()) == (1, 1, 0)

        # Case 5: Enforcement failure after update raises ValueError
        case5_file = os.path.join(tmpdir, "case5.md")
        with open(case5_file, "w", encoding="utf-8") as f:
            f.write("# Title\n")
        with patch(
            "scripts.update_badges.count_shields", return_value=(2, 1, 0)
        ), pytest.raises(ValueError, match="Enforcement failed after update"):
            update_readme(readme_path=case5_file, coverage_json_path=cov_file)


def test_update_badges_main() -> None:
    """Test executing update_badges_main with various arguments."""
    with TemporaryDirectory() as tmpdir:
        readme_file = os.path.join(tmpdir, "README.md")
        cov_file = os.path.join(tmpdir, "coverage.json")
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write(
                "# Test\n"
                "[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)\n"
                "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)\n"
            )
        with open(cov_file, "w", encoding="utf-8") as f:
            json.dump({"totals": {"percent_covered": 100.0}}, f)

        # Normal run
        with patch("subprocess.run"):
            assert update_badges_main([readme_file]) == 0

        # Enforce-only run success
        assert update_badges_main(["--enforce", readme_file]) == 0

        # Enforce-only failure on invalid file
        invalid_file = os.path.join(tmpdir, "invalid.md")
        with open(invalid_file, "w", encoding="utf-8") as f:
            f.write("# No shields here\n")
        assert update_badges_main(["--enforce", invalid_file]) == 1

        # Enforce-only failure on missing file
        assert update_badges_main(["--enforce", "/nonexistent/file.md"]) == 1


def test_update_badges_runpy_main() -> None:
    """Test executing scripts.update_badges as __main__ module."""
    orig_badges_mod = sys.modules["scripts.update_badges"]
    try:
        with TemporaryDirectory() as tmpdir:
            readme_file = os.path.join(tmpdir, "README.md")
            cov_file = os.path.join(tmpdir, "coverage.json")
            with open(readme_file, "w", encoding="utf-8") as f:
                f.write(
                    "# Test\n"
                    "[![Test Coverage](https://img.shields.io/badge/test_coverage-50%25-red.svg)](#)\n"
                    "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-50%25-red.svg)](#)\n"
                )
            with open(cov_file, "w", encoding="utf-8") as f:
                json.dump({"totals": {"percent_covered": 100.0}}, f)

            # Success case (exit_code == 0)
            sys.modules.pop("scripts.update_badges", None)
            with patch("sys.argv", ["update_badges.py", readme_file]), patch(
                "subprocess.run"
            ):
                runpy.run_module("scripts.update_badges", run_name="__main__")

            # Failure case (exit_code != 0) invokes sys.exit
            sys.modules.pop("scripts.update_badges", None)
            with patch(
                "sys.argv", ["update_badges.py", "--enforce", "/nonexistent.md"]
            ), patch("sys.exit") as mock_exit:
                runpy.run_module("scripts.update_badges", run_name="__main__")
                mock_exit.assert_called_once_with(1)
    finally:
        sys.modules["scripts.update_badges"] = orig_badges_mod


SAMPLE_ONNX_DOCS = """
# ONNX Operators

### <a name="Abs"></a><a name="abs">**Abs**</a>
has been available since version 13.
#### Attributes
<dl>
<dt><tt>auto_pad</tt> : string (default is NOTSET)</dt>
<dd>Padding</dd>
<dt><tt>kernel_shape</tt> : list of ints (required)</dt>
<dd>Kernel</dd>
<dt><tt>rates</tt> : list of floats (default is [1.0, 2.0])</dt>
<dd>Rates</dd>
<dt><tt>strides</tt> : list of strings (default is ['a', 'b'])</dt>
<dd>Strides</dd>
<dt><tt>epsilon</tt> : float (default is 1e-5)</dt>
<dd>Eps</dd>
<dt><tt>count</tt> : int (default is 42)</dt>
<dd>Count</dd>
<dt><tt>axes</tt> : list of ints (default is [])</dt>
<dd>Axes</dd>
<dt><tt>mode</tt> : string (default is nearest)</dt>
<dd>Interpolation</dd>
<dt><tt>body</tt> : graph (required)</dt>
<dd>Subgraph</dd>
<dt><tt>target_type</tt> : type</dt>
<dd>Type attribute</dd>
<dt><tt>untyped_attr</tt> : custom_variant</dt>
<dd>Custom</dd>
<dt>Malformed dt entry without tt tag</dt>
</dl>
#### Inputs
<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input tensor</dd>
<dt>Malformed input without tt tag</dt>
</dl>
#### Outputs
<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output tensor</dd>
<dt>Malformed output without tt tag</dt>
</dl>

### <a name="FallbackOp"></a>**FallbackOp**
#### Attributes
#### Inputs
#### Outputs

### <a name="Cast"></a><a name="cast">**Cast**</a>
has been available since version 1.
#### Attributes
<dl>
<dt><tt>to</tt> : int (required)</dt>
<dd>Target type</dd>
</dl>
#### Inputs
<dl>
<dt><tt>input</tt></dt>
</dl>
#### Outputs
<dl>
<dt><tt>output</tt></dt>
</dl>

### <a name="NoMatch"></a>
Header without bold operator name.
"""


def test_parse_onnx_docs() -> None:
    """Test parse_onnx_docs with comprehensive operator markdown definitions."""
    with TemporaryDirectory() as tmpdir:
        f_name = os.path.join(tmpdir, "Operators.md")
        with open(f_name, "w", encoding="utf-8") as f:
            f.write(SAMPLE_ONNX_DOCS)
        ops = parse_onnx_docs(f_name)
        assert "Abs" in ops
        assert "FallbackOp" in ops
        assert "NoMatch" not in ops

        abs_op = ops["Abs"]
        assert abs_op["domain"] == "ai.onnx"
        assert abs_op["version"] == 13
        assert "auto_pad" in abs_op["attributes"]
        assert abs_op["attributes"]["auto_pad"]["default"] == "NOTSET"
        assert abs_op["attributes"]["auto_pad"]["type"] == "str"

        assert abs_op["attributes"]["kernel_shape"]["required"] is True
        assert abs_op["attributes"]["kernel_shape"]["type"] == "List[int]"

        assert abs_op["attributes"]["rates"]["default"] == [1.0, 2.0]
        assert abs_op["attributes"]["rates"]["type"] == "List[float]"

        assert abs_op["attributes"]["strides"]["type"] == "List[str]"
        assert abs_op["attributes"]["epsilon"]["type"] == "float"
        assert abs_op["attributes"]["count"]["type"] == "int"
        assert abs_op["attributes"]["axes"]["default"] == []
        assert abs_op["attributes"]["mode"]["default"] == "nearest"
        assert abs_op["attributes"]["body"]["type"] == "Any"
        assert abs_op["attributes"]["target_type"]["type"] == "str"
        assert abs_op["attributes"]["untyped_attr"]["type"] == "Any"

        assert abs_op["inputs"] == ["X"]
        assert abs_op["outputs"] == ["Y"]

        fallback_op = ops["FallbackOp"]
        assert fallback_op["version"] == 1
        assert fallback_op["attributes"] == {}
        assert fallback_op["inputs"] == []
        assert fallback_op["outputs"] == []


def test_generate_registry_main() -> None:
    """Test generate_registry main function writes json and python registry files."""
    with TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "Operators.md")
        json_path = os.path.join(tmpdir, "onnx_ops.json")
        py_path = os.path.join(tmpdir, "onnx_registry.py")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_ONNX_DOCS)

        gen_main(md_file=md_path, json_path=json_path, registry_path=py_path)

        assert os.path.exists(json_path)
        assert os.path.exists(py_path)

        with open(json_path, "r", encoding="utf-8") as f:
            loaded_json = json.load(f)
            assert "Abs" in loaded_json

        with open(py_path, "r", encoding="utf-8") as f:
            loaded_py = f.read()
            assert "ONNX_REGISTRY" in loaded_py
            assert "load_onnx_schemas" in loaded_py

        # Test onnx_dir resolution
        onnx_dir = os.path.join(tmpdir, "fake_onnx")
        docs_dir = os.path.join(onnx_dir, "docs")
        os.makedirs(docs_dir, exist_ok=True)
        with open(os.path.join(docs_dir, "Operators.md"), "w", encoding="utf-8") as f:
            f.write(SAMPLE_ONNX_DOCS)
        gen_main(onnx_dir=onnx_dir, json_path=json_path, registry_path=py_path)
        assert os.path.exists(json_path)

        # Test onnx_dir with Operators.md at root
        onnx_root_dir = os.path.join(tmpdir, "fake_onnx_root")
        os.makedirs(onnx_root_dir, exist_ok=True)
        with open(
            os.path.join(onnx_root_dir, "Operators.md"), "w", encoding="utf-8"
        ) as f:
            f.write(SAMPLE_ONNX_DOCS)
        with patch("sys.argv", ["generate_registry.py", "--onnx-dir", onnx_root_dir]):
            gen_main(json_path=json_path, registry_path=py_path)
        assert os.path.exists(json_path)

        # Test default md path fallback when no args provided
        with patch("sys.argv", ["generate_registry.py"]), patch(
            "scripts.generate_registry.parse_onnx_docs", return_value={}
        ):
            gen_main(json_path=json_path, registry_path=py_path)
        assert os.path.exists(json_path)


def test_generate_registry_module_main() -> None:
    """Test executing scripts.generate_registry as __main__ module."""
    orig_registry_mod = sys.modules["scripts.generate_registry"]
    try:
        with TemporaryDirectory() as tmpdir:
            tmp_md = os.path.join(tmpdir, "Operators.md")
            tmp_json = os.path.join(tmpdir, "onnx_ops.json")
            tmp_py = os.path.join(tmpdir, "onnx_registry.py")
            with open(tmp_md, "w", encoding="utf-8") as f:
                f.write(SAMPLE_ONNX_DOCS)
            sys.modules.pop("scripts.generate_registry", None)
            with patch("sys.argv", ["generate_registry.py", tmp_md, tmp_json, tmp_py]):
                runpy.run_module("scripts.generate_registry", run_name="__main__")
            assert os.path.exists(tmp_json)
            assert os.path.exists(tmp_py)
    finally:
        sys.modules["scripts.generate_registry"] = orig_registry_mod


def test_generate_registry_cross_validate_branches() -> None:
    """Test cross_validate_with_onnx_defs edge cases and ai.onnx prefixed operator handling."""
    # Test extract_section helper
    block = "#### SectionA\nContent A\n#### SectionB\nContent B"
    assert extract_section(block, "SectionA") == "Content A"
    assert extract_section(block, "NonExistent") == ""

    # Test parse_onnx_docs with ai.onnx. prefixed op and cross_validate=True
    md_content = """# ONNX Operators
### <a name="ai.onnx.preview.training.Adagrad"></a><a name="adagrad">**ai.onnx.preview.training.Adagrad**</a>
has been available since version 1.
#### Inputs
<dl>
<dt><tt>R</tt></dt>
</dl>
#### Outputs
<dl>
<dt><tt>output</tt></dt>
</dl>
"""
    with TemporaryDirectory() as tmpdir:
        md_file = os.path.join(tmpdir, "Operators.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        ops = parse_onnx_docs(md_file, cross_validate=True)
        assert "ai.onnx.preview.training.Adagrad" in ops
        assert (
            ops["ai.onnx.preview.training.Adagrad"]["domain"]
            == "ai.onnx.preview.training"
        )

    # Test cross_validate_with_onnx_defs with empty domain schema and non-empty domain
    class FakeSchema:
        """Mock ONNX operator schema definition for unit testing."""

        def __init__(self, domain: str = "") -> None:
            """Initialize mock schema.

            Args:
                domain: Domain string.
            """
            self.name = "FakeOp"
            self.since_version = 1
            self.attributes: dict[str, Any] = {}
            self.inputs: list[Any] = []
            self.outputs: list[Any] = []
            self.domain = domain

    with patch("onnx.defs.get_all_schemas", return_value=[FakeSchema("")]):
        test_ops = {
            "FakeOp": {
                "domain": "ai.onnx",
                "version": 1,
                "attributes": {"bad": {}},
                "inputs": ["x"],
                "outputs": [],
            }
        }
        cleaned = cross_validate_with_onnx_defs(test_ops)
        assert cleaned["FakeOp"]["attributes"] == {}
        assert cleaned["FakeOp"]["inputs"] == []
        assert cleaned["FakeOp"]["domain"] == "ai.onnx"

    # Test schema with explicit domain and version comparison false branch
    s_v10 = FakeSchema("ai.onnx.custom")
    s_v10.since_version = 10
    s_v5 = FakeSchema("ai.onnx.custom")
    s_v5.since_version = 5

    with patch("onnx.defs.get_all_schemas", return_value=[s_v10, s_v5]):
        test_ops2 = {
            "FakeOp": {
                "domain": "ai.onnx",
                "version": 1,
                "attributes": {},
                "inputs": [],
                "outputs": [],
            }
        }
        cleaned2 = cross_validate_with_onnx_defs(test_ops2)
        assert cleaned2["FakeOp"]["domain"] == "ai.onnx.custom"

    # Test ImportError branch in cross_validate_with_onnx_defs
    with patch.dict("sys.modules", {"onnx.defs": None}):
        assert cross_validate_with_onnx_defs({"Op": {}}) == {"Op": {}}


def test_generate_stablehlo_registry(tmp_path: Path) -> None:
    """Test extract_stablehlo_schemas and main in scripts.generate_stablehlo_registry.

    Args:
        tmp_path (Path): Temporary test directory.
    """
    manifest_data = {
        "operations": [
            {
                "api_path": "stablehlo.abs",
                "name": "AbsOp",
                "operands": [{"name": "operand"}],
                "returns": [{"name": "result"}],
                "attributes": {
                    "bool_attr": {"type": "DefaultValued<BoolAttr>"},
                    "int_attr": {"type": "IntegerAttr"},
                    "float_attr": {"type": "FloatAttr"},
                    "array_attr": {"type": "ArrayAttr"},
                    "dict_attr": {"type": "DictionaryAttr"},
                    "plain_attr": "StrAttr",
                    "": {"type": "invalid"},
                },
            },
            {
                "name": "ConvolutionOp",
                "api_path": "stablehlo.convolution",
                "operands": [{"name": "lhs"}, {"name": "rhs"}],
                "returns": [{"name": "result"}],
                "attributes": {
                    "batch_group_count": {"type": "IntegerAttr"},
                    "padding": {"type": "Array"},
                },
            },
            {
                "name": "CustomOp",
                "operands": [{"name": ""}, "not_dict"],
                "returns": [],
                "attributes": [
                    {
                        "name": "list_attr",
                        "type": "str",
                        "required": True,
                        "default": "x",
                    },
                    {"name": ""},
                ],
            },
        ]
    }
    manifest_file = tmp_path / "test_hlo.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    ops = extract_stablehlo_schemas(manifest_file)
    assert len(ops) == 3
    assert ops[0]["name"] == "abs"
    assert ops[1]["name"] == "convolution"
    assert ops[2]["name"] == "custom"

    # Test with raw list format
    list_file = tmp_path / "test_list_hlo.json"
    with open(list_file, "w", encoding="utf-8") as f:
        json.dump([{"name": "UnaryOp", "operands": [], "returns": []}], f)
    ops_list = extract_stablehlo_schemas(list_file)
    assert len(ops_list) == 1
    assert ops_list[0]["name"] == "unary"

    # Test categories format with util key, duplicates, and edge item structures
    cat_manifest = {
        "categories": {
            "util": [
                {
                    "name": "DotOp",
                    "operands": [{"name": None}, {"name": "valid_in"}, "not_a_dict"],
                    "returns": [{"name": None}, {"name": "valid_out"}, "not_a_dict"],
                    "attributes": [{}, {"name": "valid_attr"}, "not_a_dict"],
                },
                {"name": "DotOp"},  # duplicate name
                {"name": ""},  # empty name
            ],
            "other": [],
        }
    }
    cat_file = tmp_path / "cat_hlo.json"
    with open(cat_file, "w", encoding="utf-8") as f:
        json.dump(cat_manifest, f)
    ops_cat = extract_stablehlo_schemas(cat_file)
    assert len(ops_cat) == 1
    assert ops_cat[0]["name"] == "dot"

    # Test dictionary without categories or operations
    dict_file = tmp_path / "other_dict.json"
    with open(dict_file, "w", encoding="utf-8") as f:
        json.dump({"random_key": 1}, f)
    assert extract_stablehlo_schemas(dict_file) == []

    # Test categories dict without util
    no_util_file = tmp_path / "no_util.json"
    with open(no_util_file, "w", encoding="utf-8") as f:
        json.dump({"categories": {"other": []}}, f)
    assert extract_stablehlo_schemas(no_util_file) == []

    # Test non-dict non-list scalar JSON
    scalar_file = tmp_path / "scalar.json"
    with open(scalar_file, "w", encoding="utf-8") as f:
        json.dump("scalar_string", f)
    assert extract_stablehlo_schemas(scalar_file) == []

    # Test main() execution
    out_json = tmp_path / "stablehlo_ops.json"
    gen_stablehlo_main(snapshot_path=str(manifest_file), output_path=str(out_json))
    assert out_json.exists()

    # Test run_module as __main__
    sys.modules.pop("scripts.generate_stablehlo_registry", None)
    with patch(
        "sys.argv",
        ["generate_stablehlo_registry.py", str(manifest_file), str(out_json)],
    ):
        runpy.run_module("scripts.generate_stablehlo_registry", run_name="__main__")
    assert out_json.exists()


def test_verify_grounding_find_snapshots_directory() -> None:
    """Test find_snapshots_directory resolution and fallbacks."""
    with TemporaryDirectory() as tmpdir:
        # Override path that is a dir
        assert find_snapshots_directory(tmpdir) is not None

        # Override path that is not a dir
        non_existent = os.path.join(tmpdir, "does_not_exist")
        assert find_snapshots_directory(non_existent) is None

        # ML_ECOSYSTEM_SNAPSHOTS_DIR env branch
        with patch.dict(os.environ, {"ML_ECOSYSTEM_SNAPSHOTS_DIR": tmpdir}):
            assert find_snapshots_directory() == Path(tmpdir).resolve()

        # ML_ECOSYSTEM_SNAPSHOTS_DIR non-existent branch
        with patch.dict(
            os.environ, {"ML_ECOSYSTEM_SNAPSHOTS_DIR": "/nonexistent/path"}
        ):
            assert find_snapshots_directory() is not None

        # ML_FRAMEWORK_SNAPSHOTS_DIR env branch
        with patch.dict(
            os.environ,
            {"ML_FRAMEWORK_SNAPSHOTS_DIR": tmpdir, "ML_ECOSYSTEM_SNAPSHOTS_DIR": ""},
        ):
            assert find_snapshots_directory() == Path(tmpdir).resolve()

        # Without override, should find existing DEFAULT_SNAPSHOT_DIR or fallback
        with patch.dict(os.environ, {}, clear=True), patch(
            "scripts.verify_grounding.DEFAULT_SNAPSHOT_DIR", tmpdir
        ):
            assert find_snapshots_directory() == Path(tmpdir).resolve()

    # Test cache and fallback branches
    with patch.dict(os.environ, {}, clear=True), patch(
        "scripts.verify_grounding.DEFAULT_SNAPSHOT_DIR", "/nonexistent/dir"
    ):
        with patch("pathlib.Path.is_dir", side_effect=[False, True]):
            assert find_snapshots_directory() is not None

        with patch("pathlib.Path.is_dir", side_effect=[False, False, True]):
            assert find_snapshots_directory() is not None

        with patch("pathlib.Path.is_dir", side_effect=[False, False, False, True]):
            assert find_snapshots_directory() is not None

        with patch(
            "pathlib.Path.is_dir", side_effect=[False, False, False, False, True]
        ):
            assert find_snapshots_directory() is not None

        with patch(
            "pathlib.Path.is_dir", side_effect=[False, False, False, False, False, True]
        ):
            assert find_snapshots_directory() is not None

        with patch(
            "pathlib.Path.is_dir",
            side_effect=[False, False, False, False, False, False, True],
        ):
            assert find_snapshots_directory() is not None

        with patch(
            "pathlib.Path.is_dir",
            side_effect=[False, False, False, False, False, False, False, True],
        ):
            assert find_snapshots_directory() is not None

        # All is_dir return False
        with patch("pathlib.Path.is_dir", return_value=False):
            assert find_snapshots_directory() is None


def test_verify_grounding_stablehlo() -> None:
    """Test verify_stablehlo_grounding with missing file, invalid op, and valid snapshot."""
    from pathlib import Path

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Snapshot file missing
        errs = verify_stablehlo_grounding(tmp_path)
        assert any("file not found" in e for e in errs)

        # Snapshot file with missing op, custom_call attribute handling, and invalid attribute
        fake_snapshot = {
            "categories": {
                "stablehlo_op": [
                    {
                        "name": "dot_general",
                        "api_path": "stablehlo.dot_general",
                        "params": [{"name": "unknown_param"}, "not_dict", {}],
                        "attributes": [{"name": "unknown_attr"}, "string_attr", {}],
                    },
                    {
                        "name": "custom_call",
                        "api_path": "stablehlo.custom_call",
                        "params": [{"name": "call_target_name"}],
                        "attributes": [{"name": "call_target_name"}],
                    },
                ]
            }
        }
        snap_file = tmp_path / "stablehlo_v1.0.0.json"
        with open(snap_file, "w", encoding="utf-8") as f:
            json.dump(fake_snapshot, f)

        errs_missing = verify_stablehlo_grounding(tmp_path)
        assert len(errs_missing) > 0
        snap_file.unlink()

        # Test candidate file fallback when stablehlo_v1.0.0.json is not present
        candidate_file = tmp_path / "stablehlo_v1.9.0.json"
        with open(candidate_file, "w", encoding="utf-8") as f:
            json.dump(fake_snapshot, f)
        errs_candidate = verify_stablehlo_grounding(tmp_path)
        assert len(errs_candidate) > 0
        candidate_file.unlink()


def test_verify_grounding_custom_and_onnx() -> None:
    """Test verify_custom_ops_grounding and verify_onnx_grounding under normal and defective conditions."""
    # Healthy cases
    assert verify_custom_ops_grounding() == []
    assert verify_onnx_grounding() == []

    # Defective custom ops
    from ml_switcheroo_ir.schema.onnx_registry import OpSchema

    fake_custom = {
        "bad1": OpSchema(
            name="",
            domain="ml.switcheroo.custom",
            version=1,
            attributes={},
            inputs=[],
            outputs=["out"],
        ),
        "bad2": OpSchema(
            name="bad2",
            domain="wrong.domain",
            version=1,
            attributes={},
            inputs=[],
            outputs=["out"],
        ),
        "bad3": OpSchema(
            name="bad3",
            domain="ml.switcheroo.custom",
            version=1,
            attributes={},
            inputs=[],
            outputs=[],
        ),
    }
    with patch("scripts.verify_grounding.CUSTOM_OPS_REGISTRY", fake_custom):
        custom_errs = verify_custom_ops_grounding()
        assert len(custom_errs) == 3

    # Defective ONNX registry
    with patch("scripts.verify_grounding.ONNX_REGISTRY", {}):
        assert len(verify_onnx_grounding()) == 1

    fake_onnx = {
        "Mismatch": OpSchema(
            name="OtherName",
            domain="ai.onnx",
            version=1,
            attributes={},
            inputs=[],
            outputs=[],
        ),
        "WrongDomain": OpSchema(
            name="WrongDomain",
            domain="not.onnx",
            version=1,
            attributes={},
            inputs=[],
            outputs=[],
        ),
    }
    with patch("scripts.verify_grounding.ONNX_REGISTRY", fake_onnx):
        onnx_errs = verify_onnx_grounding()
        assert len(onnx_errs) == 2


def test_verify_grounding_main_paths() -> None:
    """Test verify_grounding main function under various scenarios."""
    # When snapshots directory not found but schemas are valid
    with patch("scripts.verify_grounding.find_snapshots_directory", return_value=None):
        assert verify_grounding_main([]) == 0

    # When snapshots directory not found and schemas are invalid
    with patch(
        "scripts.verify_grounding.find_snapshots_directory", return_value=None
    ), patch(
        "scripts.verify_grounding.verify_custom_ops_grounding", return_value=["error"]
    ):
        assert verify_grounding_main([]) == 1

    # When snapshots directory is empty and no --snapshots-dir argument given
    with TemporaryDirectory() as empty_tmpdir, patch(
        "scripts.verify_grounding.find_snapshots_directory",
        return_value=Path(empty_tmpdir),
    ):
        assert verify_grounding_main([]) == 0

    # When snapshots directory found with snapshots and no --snapshots-dir argument given
    with TemporaryDirectory() as populated_tmpdir:
        dummy_file = Path(populated_tmpdir) / "snapshot.json"
        dummy_file.write_text("{}", encoding="utf-8")
        with patch(
            "scripts.verify_grounding.find_snapshots_directory",
            return_value=Path(populated_tmpdir),
        ), patch(
            "scripts.verify_grounding.verify_stablehlo_grounding", return_value=[]
        ), patch(
            "scripts.verify_grounding.verify_mlir_grounding", return_value=[]
        ), patch(
            "scripts.verify_grounding.verify_ir_snapshot_grounding", return_value=[]
        ), patch(
            "scripts.verify_grounding.verify_custom_ops_grounding", return_value=[]
        ), patch("scripts.verify_grounding.verify_onnx_grounding", return_value=[]):
            assert verify_grounding_main([]) == 0

    # When snapshots directory found and all schemas valid
    with TemporaryDirectory() as tmpdir:
        with patch(
            "scripts.verify_grounding.find_snapshots_directory",
            return_value=Path(tmpdir),
        ), patch(
            "scripts.verify_grounding.verify_stablehlo_grounding", return_value=[]
        ), patch(
            "scripts.verify_grounding.verify_mlir_grounding", return_value=[]
        ), patch(
            "scripts.verify_grounding.verify_ir_snapshot_grounding", return_value=[]
        ), patch(
            "scripts.verify_grounding.verify_custom_ops_grounding", return_value=[]
        ), patch("scripts.verify_grounding.verify_onnx_grounding", return_value=[]):
            assert verify_grounding_main(["--snapshots-dir", tmpdir]) == 0

        # When snapshots directory found but errors detected
        with patch(
            "scripts.verify_grounding.find_snapshots_directory",
            return_value=Path(tmpdir),
        ), patch(
            "scripts.verify_grounding.verify_stablehlo_grounding",
            return_value=["stablehlo error"],
        ):
            assert verify_grounding_main(["--snapshots-dir", tmpdir]) == 1


def test_verify_grounding_runpy_main() -> None:
    """Test executing scripts.verify_grounding as __main__ module."""
    orig_mod = sys.modules["scripts.verify_grounding"]
    sys.modules.pop("scripts.verify_grounding", None)
    try:
        with TemporaryDirectory() as empty_tmpdir, patch.dict(
            os.environ, {"ML_FRAMEWORK_SNAPSHOTS_DIR": empty_tmpdir}
        ), patch("sys.argv", ["verify_grounding.py"]), patch("sys.exit") as mock_exit:
            runpy.run_module("scripts.verify_grounding", run_name="__main__")
            mock_exit.assert_called_once_with(0)
    finally:
        sys.modules["scripts.verify_grounding"] = orig_mod


def test_find_snapshots_directory_env_and_override() -> None:
    """Test find_snapshots_directory with environment variable and explicit override."""
    with TemporaryDirectory() as tmpdir:
        # Explicit override
        assert find_snapshots_directory(tmpdir) == Path(tmpdir).resolve()
        assert find_snapshots_directory("/non/existent/path") is None

        # Environment variable valid and invalid
        with patch.dict(os.environ, {"ML_FRAMEWORK_SNAPSHOTS_DIR": tmpdir}):
            assert find_snapshots_directory() == Path(tmpdir).resolve()

        with patch.dict(
            os.environ, {"ML_FRAMEWORK_SNAPSHOTS_DIR": "/invalid/nonexistent/dir"}
        ), patch("scripts.verify_grounding.DEFAULT_SNAPSHOT_DIR", tmpdir):
            # Should fall back to default_path or relative
            assert find_snapshots_directory() == Path(tmpdir).resolve()


def test_verify_mlir_grounding_paths() -> None:
    """Test verify_mlir_grounding with missing file, valid snapshot, and missing dialects."""
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # Missing file returns []
        assert verify_mlir_grounding(tmppath) == []

        # Candidate fallback when mlir_v0.4.30.json is not present
        candidate_mlir = tmppath / "mlir_v19.1.0.json"
        with open(candidate_mlir, "w", encoding="utf-8") as f:
            json.dump({"categories": {"util": [{"api_path": "other_dialect.op"}]}}, f)
        assert len(verify_mlir_grounding(tmppath)) > 0
        candidate_mlir.unlink()

        # File with missing dialects and edge items (non-list, non-dict, dict without api_path)
        mlir_json = tmppath / "mlir_v0.4.30.json"
        with open(mlir_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categories": {
                        "util": [
                            {"api_path": "arith.addf", "attributes": ["lhs", "rhs"]},
                            {
                                "api_path": "tensor.empty",
                                "attributes": [
                                    {"name": "staticSizes"},
                                    {"name": "other_attr"},
                                    "string_attr",
                                    {},
                                ],
                            },
                            "not_a_dict",
                            {},
                        ],
                        "non_list": "scalar_val",
                    }
                },
                f,
            )
        errs = verify_mlir_grounding(tmppath)
        assert len(errs) > 0


def test_verify_ir_snapshot_grounding_paths() -> None:
    """Test verify_ir_snapshot_grounding with missing file, valid snapshot, and defective entries."""
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # Missing file and no candidates returns []
        assert verify_ir_snapshot_grounding(tmppath) == []

        # Test candidate fallback with ir_v0.1.0.json when ir_v0.0.3.json is missing
        candidate_json = tmppath / "ir_v0.1.0.json"
        with open(candidate_json, "w", encoding="utf-8") as f:
            json.dump({"categories": {"classes": [], "functions": []}}, f)
        assert verify_ir_snapshot_grounding(tmppath) == []

        # File with unresolvable classes and functions
        ir_json = tmppath / "ir_v0.0.3.json"
        with open(ir_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categories": {
                        "classes": [
                            {"api_path": ""},
                            {"api_path": "nonexistent_module.BadClass"},
                            {"api_path": "ml_switcheroo_ir.NonExistentClass"},
                            {"api_path": "ml_switcheroo_ir.LogicalNode.bad_method"},
                        ],
                        "functions": [
                            {"api_path": ""},
                            {"api_path": "ml_switcheroo_ir.topological_sort"},
                            {"api_path": "ml_switcheroo_ir.nonexistent_fn"},
                            {"api_path": "nonexistent_module.bad_fn"},
                        ],
                    }
                },
                f,
            )
        errs = verify_ir_snapshot_grounding(tmppath)
        assert len(errs) >= 5


def test_extract_known_attributes() -> None:
    """Test _extract_known_attributes with various attribute and parameter representations."""
    # 1. Dict attributes and valid params
    rec1 = {
        "params": [{"name": "p1"}, {"name": ""}, "not_dict"],
        "attributes": {"attr_d1": {}, "attr_d2": {}},
    }
    assert _extract_known_attributes(rec1) == {"p1", "attr_d1", "attr_d2"}

    # 2. List attributes with dicts and strings
    rec2 = {
        "params": [],
        "attributes": [
            {"name": "attr_l1"},
            {"name": None},
            "attr_str",
            "not_dict_or_str",
        ],
    }
    assert _extract_known_attributes(rec2) == {"attr_l1", "attr_str", "not_dict_or_str"}

    # 3. Empty record
    assert _extract_known_attributes({}) == set()


def test_verify_grounding_rdna() -> None:
    """Test verify_rdna_grounding with missing file, invalid entries, and valid instructions."""
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # 1. Missing file returns empty list
        assert verify_rdna_grounding(tmppath) == []

        # 2. Defective snapshot
        rdna_file = tmppath / "amd_rdna_snapshot.json"
        with open(rdna_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categories": {
                        "instructions": [
                            "not_a_dict",
                            {"name": ""},
                            {"name": "V_BAD_SLOT", "vopd_slot": "INVALID"},
                            {"name": "V_GOOD_OP", "vopd_slot": "X"},
                        ]
                    }
                },
                f,
            )
        errs = verify_rdna_grounding(tmppath)
        assert len(errs) == 2
        assert "missing name/mnemonic" in errs[0]
        assert "invalid vopd_slot" in errs[1]


def test_verify_grounding_sass() -> None:
    """Test verify_sass_grounding with missing file, invalid entries, and valid instructions."""
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # 1. Missing file returns empty list
        assert verify_sass_grounding(tmppath) == []

        # 2. Defective snapshot
        sass_file = tmppath / "nvidia_sass_snapshot.json"
        with open(sass_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categories": {
                        "instructions": [
                            "not_a_dict",
                            {"name": ""},
                            {"name": "FFMA_BAD", "execution_latency": "bad_latency"},
                            {"name": "FFMA_GOOD", "execution_latency": 4},
                        ]
                    }
                },
                f,
            )
        errs = verify_sass_grounding(tmppath)
        assert len(errs) == 2
        assert "missing name/mnemonic" in errs[0]
        assert "invalid execution_latency" in errs[1]


def test_verify_grounding_wgsl() -> None:
    """Test verify_wgsl_grounding with missing file, defective ops, and valid schema."""
    from unittest.mock import mock_open

    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # 1. Valid real schema
        real_errs = verify_wgsl_grounding(tmppath)
        assert real_errs == []

        # 2. Missing file branch
        with patch("pathlib.Path.is_file", return_value=False):
            assert verify_wgsl_grounding(tmppath) == []

        # 3. Defective content
        fake_data = {
            "ops": [
                {"name": ""},
                {"name": "bad_dom_op", "domain": "invalid_domain"},
                {"name": "good_op", "domain": "wgsl"},
            ]
        }
        with patch("builtins.open", mock_open(read_data=json.dumps(fake_data))):
            errs = verify_wgsl_grounding(tmppath)
            assert len(errs) == 2
            assert "missing name" in errs[0]
            assert "invalid domain" in errs[1]


def test_verify_grounding_ptx() -> None:
    """Test verify_ptx_grounding with missing file, invalid items, and valid items."""
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # 1. Missing file
        assert verify_ptx_grounding(tmppath) == []

        # 2. Defective snapshot
        ptx_file = tmppath / "nvidia_ptx_snapshot.json"
        with open(ptx_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categories": {
                        "metadata": "non_list",
                        "instructions": [
                            "not_a_dict",
                            {"name": ""},
                            {"api_path": "ptx.add"},
                            {"mnemonic": "sub.s32"},
                            {"name": "add.s32"},
                        ],
                    }
                },
                f,
            )
        errs = verify_ptx_grounding(tmppath)
        assert len(errs) == 1
        assert "missing identifier" in errs[0]


def test_verify_grounding_metal() -> None:
    """Test verify_metal_grounding with missing file, invalid items, and valid items."""
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # 1. Missing file
        assert verify_metal_grounding(tmppath) == []

        # 2. Defective snapshot
        metal_file = tmppath / "metal_snapshot.json"
        with open(metal_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categories": {
                        "metadata": "non_list",
                        "functions": [
                            "not_a_dict",
                            {"name": ""},
                            {"api_path": "metal.threadgroup_barrier"},
                            {"name": "threadgroup_barrier"},
                        ],
                    }
                },
                f,
            )
        errs = verify_metal_grounding(tmppath)
        assert len(errs) == 1
        assert "missing identifier" in errs[0]


def test_verify_grounding_wasm() -> None:
    """Test verify_wasm_grounding with missing file, invalid items, and valid items."""
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # 1. Missing file
        assert verify_wasm_grounding(tmppath) == []

        # 2. Defective snapshot
        wasm_file = tmppath / "wasm_snapshot.json"
        with open(wasm_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categories": {
                        "metadata": "non_list",
                        "instructions": [
                            "not_a_dict",
                            {"name": ""},
                            {"api_path": "wasm.i32x4_add"},
                            {"name": "i32x4.add"},
                        ],
                    }
                },
                f,
            )
        errs = verify_wasm_grounding(tmppath)
        assert len(errs) == 1
        assert "missing identifier" in errs[0]


def test_verify_grounding_array_api() -> None:
    """Test verify_array_api_grounding with missing file, defective ops, and valid snapshots."""
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # 1. Missing file returns empty list
        assert verify_array_api_grounding(tmppath) == []

        # 2. Defective snapshot with missing op, wrong input, and unrecognized attribute
        array_api_file = tmppath / "array_api_v2024.12.json"
        with open(array_api_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categories": {
                        "array": [
                            {
                                "name": "abs",
                                "api_path": "array_api.abs",
                                "params": [
                                    {"name": "wrong_input", "kind": "POSITIONAL_ONLY"}
                                ],
                                "kwargs": [],
                            },
                            {
                                "name": "sum",
                                "api_path": "array_api.sum",
                                "params": [
                                    {"name": "x", "kind": "POSITIONAL_ONLY"},
                                    {"name": "axis", "kind": "KEYWORD_ONLY"},
                                ],
                                "kwargs": ["axis"],
                            },
                        ]
                    }
                },
                f,
            )
        errs = verify_array_api_grounding(tmppath)
        assert len(errs) > 0
        assert any("is not grounded" in e for e in errs)
        assert any("does not match snapshot parameter" in e for e in errs)
        assert any("attribute" in e and "is not recognized" in e for e in errs)

        # 3. Valid snapshot with fixture
        fixtures_dir = Path(DEFAULT_SNAPSHOT_DIR)
        if (fixtures_dir / "array_api_v2024.12.json").is_file():
            assert verify_array_api_grounding(fixtures_dir) == []

        # 4. Schema with empty inputs, empty kw_params, only name, only api_path, empty dict
        empty_schema = MagicMock()
        empty_schema.inputs = []
        empty_schema.attributes = ["axis"]
        empty_both = MagicMock()
        empty_both.inputs = []
        empty_both.attributes = []
        with open(array_api_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categories": {
                        "array": "not_a_list",
                        "util": [
                            "not_a_dict",
                            {},
                            {"name": "abs", "params": []},
                            {"api_path": "array_api.sum"},
                        ],
                    }
                },
                f,
            )
        with patch.dict(
            "scripts.verify_grounding.ARRAY_API_REGISTRY",
            {"abs": empty_schema, "sum": empty_both},
            clear=True,
        ):
            assert verify_array_api_grounding(tmppath) == []


def test_verify_grounding_aten() -> None:
    """Test verify_aten_grounding with missing file, defective ops, and valid snapshots."""
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # 1. Missing file returns empty list
        assert verify_aten_grounding(tmppath) == []

        # 2. Defective snapshot with missing op and missing attribute
        aten_file = tmppath / "aten_v2.8.0.json"
        with open(aten_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categories": {
                        "array": [
                            {
                                "name": "add",
                                "api_path": "aten.add",
                                "params": [
                                    {"name": "self", "kind": "POSITIONAL_ONLY"},
                                    {"name": "other", "kind": "POSITIONAL_ONLY"},
                                ],
                            }
                        ]
                    }
                },
                f,
            )
        errs = verify_aten_grounding(tmppath)
        assert len(errs) > 0
        assert any("is not grounded" in e for e in errs)
        assert any("attribute 'alpha' is not recognized" in e for e in errs)

        # 3. Snapshot with non-list category, non-dict item, only api_path, only name
        with open(aten_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categories": {
                        "array": "not_a_list",
                        "util": [
                            "not_a_dict",
                            {"name": "add"},
                            {"api_path": "aten.matmul"},
                        ],
                    }
                },
                f,
            )
        errs_cat = verify_aten_grounding(tmppath)
        assert len(errs_cat) > 0

        # 4. Valid snapshot with fixture
        fixtures_dir = Path(DEFAULT_SNAPSHOT_DIR)
        if (fixtures_dir / "aten_v2.8.0.json").is_file():
            assert verify_aten_grounding(fixtures_dir) == []

        # 5. ATen schema with empty known_params and empty dict
        empty_aten_schema = MagicMock()
        empty_aten_schema.attributes = ["alpha"]
        with open(aten_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categories": {
                        "array": [
                            {},
                            {"name": "add", "params": []},
                            {"api_path": "aten.matmul"},
                        ],
                    }
                },
                f,
            )
        with patch.dict(
            "scripts.verify_grounding.ATEN_REGISTRY",
            {"add": empty_aten_schema, "matmul": empty_aten_schema},
            clear=True,
        ):
            assert verify_aten_grounding(tmppath) == []


def test_verify_grounding_collectives() -> None:
    """Test verify_collectives_grounding with missing file, defective ops, and valid snapshots."""
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # 1. Missing file returns empty list
        assert verify_collectives_grounding(tmppath) == []

        # 2. Defective snapshot missing collective ops
        nccl_file = tmppath / "nccl_v2.21.json"
        with open(nccl_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categories": {
                        "array": [{"name": "broadcast", "api_path": "nccl.broadcast"}]
                    }
                },
                f,
            )
        errs = verify_collectives_grounding(tmppath)
        assert len(errs) >= 4
        assert any("is not grounded" in e for e in errs)

        # 3. Missing collective from registry
        with patch.dict(
            "scripts.verify_grounding.COLLECTIVE_OPS_REGISTRY",
            {},
            clear=True,
        ):
            errs_missing_reg = verify_collectives_grounding(tmppath)
            assert any(
                "missing from COLLECTIVE_OPS_REGISTRY" in e for e in errs_missing_reg
            )

        # 4. Snapshot with non-list category, non-dict item, only api_path, only name
        with open(nccl_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categories": {
                        "array": "not_a_list",
                        "util": [
                            "not_a_dict",
                            {},
                            {"name": "all_reduce"},
                            {"api_path": "nccl.all_gather"},
                            {"name": "reduce_scatter"},
                            {"api_path": "nccl.all_to_all"},
                        ],
                    }
                },
                f,
            )
        fake_all_reduce_no_red = MagicMock()
        fake_all_reduce_no_red.attributes = ["comm"]
        fake_all_reduce_no_comm = MagicMock()
        fake_all_reduce_no_comm.attributes = ["reduction_op"]

        with patch.dict(
            "scripts.verify_grounding.COLLECTIVE_OPS_REGISTRY",
            {
                "all_reduce": fake_all_reduce_no_red,
                "all_gather": fake_all_reduce_no_red,
                "reduce_scatter": fake_all_reduce_no_red,
                "all_to_all": fake_all_reduce_no_red,
            },
        ):
            errs_no_red = verify_collectives_grounding(tmppath)
            assert any(
                "schema missing 'reduction_op' attribute" in e for e in errs_no_red
            )

        with patch.dict(
            "scripts.verify_grounding.COLLECTIVE_OPS_REGISTRY",
            {
                "all_reduce": fake_all_reduce_no_comm,
                "all_gather": fake_all_reduce_no_comm,
                "reduce_scatter": fake_all_reduce_no_comm,
                "all_to_all": fake_all_reduce_no_comm,
            },
        ):
            errs_no_comm = verify_collectives_grounding(tmppath)
            assert any(
                "schema missing 'comm' communicator attribute" in e
                for e in errs_no_comm
            )

        # 5. Non-reduction collective schema with all communicator params valid
        fake_coll_schema = MagicMock()
        fake_coll_schema.attributes = ["comm", "stream", "datatype"]
        fake_coll_schema_red = MagicMock()
        fake_coll_schema_red.attributes = ["reduction_op", "comm", "stream", "datatype"]
        with patch.dict(
            "scripts.verify_grounding.COLLECTIVE_OPS_REGISTRY",
            {
                "all_reduce": fake_coll_schema_red,
                "all_gather": fake_coll_schema,
                "reduce_scatter": fake_coll_schema_red,
                "all_to_all": fake_coll_schema,
            },
            clear=True,
        ):
            assert verify_collectives_grounding(tmppath) == []

        # 6. Valid snapshot with fixture
        fixtures_dir = Path(DEFAULT_SNAPSHOT_DIR)
        if (fixtures_dir / "nccl_v2.21.json").is_file():
            assert verify_collectives_grounding(fixtures_dir) == []


def test_verify_grounding_custom_attention() -> None:
    """Test verify_custom_ops_grounding with flash_attention snapshot checks and edge cases."""
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        flash_file = tmppath / "flash_attention_v2.6.3.json"

        # 1. Defective JSON in flash_attention file
        flash_file.write_text("{invalid_json", encoding="utf-8")
        errs_bad_json = verify_custom_ops_grounding(tmppath)
        assert any(
            "Failed parsing flash_attention snapshot" in e for e in errs_bad_json
        )

        # 2. Non-list category and non-dict items
        flash_file.write_text(
            json.dumps(
                {
                    "categories": {
                        "neural_ops": ["not_a_dict"],
                        "util": "not_a_list",
                    }
                }
            ),
            encoding="utf-8",
        )
        assert verify_custom_ops_grounding(tmppath) == []

        # 3. Attention op inputs mismatch
        flash_file.write_text(
            json.dumps(
                {
                    "categories": {
                        "neural_ops": [
                            {
                                "name": "FlashAttention",
                                "api_path": "ml.switcheroo.custom.FlashAttention",
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        fake_schema = MagicMock()
        fake_schema.inputs = ["wrong_q", "wrong_k"]
        fake_schema.outputs = ["out"]
        fake_schema.name = "FlashAttention"
        fake_schema.domain = "ml.switcheroo.custom"
        with patch.dict(
            "scripts.verify_grounding.CUSTOM_OPS_REGISTRY",
            {"FlashAttention": fake_schema},
        ):
            errs_mismatch = verify_custom_ops_grounding(tmppath)
            assert any(
                "inputs ['wrong_q', 'wrong_k'] do not match expected" in e
                for e in errs_mismatch
            )

        # 4. Valid flash_attention snapshot matching CUSTOM_OPS_REGISTRY
        fixtures_dir = Path(DEFAULT_SNAPSHOT_DIR)
        if (fixtures_dir / "flash_attention_v2.6.3.json").is_file():
            assert verify_custom_ops_grounding(fixtures_dir) == []

        # 5. Snapshot item with empty dict and api_path but no name
        flash_file.write_text(
            json.dumps(
                {
                    "categories": {
                        "neural_ops": [
                            {},
                            {
                                "api_path": "ml.switcheroo.custom.FlashAttention",
                            },
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        assert verify_custom_ops_grounding(tmppath) == []

        # 6. Missing attention op in registry
        with patch.dict(
            "scripts.verify_grounding.CUSTOM_OPS_REGISTRY",
            {},
            clear=True,
        ):
            errs_missing = verify_custom_ops_grounding(fixtures_dir)
            assert any("missing from CUSTOM_OPS_REGISTRY" in e for e in errs_missing)


def test_verify_grounding_main_ecosystem_and_strict() -> None:
    """Test verify_grounding main function with --ecosystem-snapshots-dir and --strict flags."""
    fixtures_dir = str(DEFAULT_SNAPSHOT_DIR)

    # Strict mode with non-existent directory returns 1
    assert (
        verify_grounding_main(["--strict", "--snapshots-dir", "/nonexistent/dir"]) == 1
    )

    # Strict mode with empty directory returns 1
    with TemporaryDirectory() as empty_dir:
        assert verify_grounding_main(["--strict", "--snapshots-dir", empty_dir]) == 1

        # Strict mode when target_dir is None and empty snapshots_dir
        with patch(
            "scripts.verify_grounding.find_snapshots_directory",
            return_value=Path(empty_dir),
        ):
            assert verify_grounding_main(["--strict"]) == 1
            assert verify_grounding_main([]) == 0

    # Passing explicit --ecosystem-snapshots-dir with valid fixtures
    assert (
        verify_grounding_main(["--strict", "--ecosystem-snapshots-dir", fixtures_dir])
        == 0
    )
