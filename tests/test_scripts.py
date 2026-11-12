"""Unit tests for repository maintenance and code generation scripts."""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

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
    get_coverage_metrics,
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

    assert scripts is not None


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
    """Test get_doc_coverage parses output correctly and raises ValueError on failure."""
    assert get_doc_coverage() == 100.0

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "actual: 97.4%\n"
        assert get_doc_coverage() == 97.4

    with (
        patch("subprocess.run") as mock_run_zero,
        pytest.raises(
            ValueError,
            match="Failed to determine documentation coverage from interrogate",
        ),
    ):
        mock_run_zero.return_value.stdout = "actual: 0.0%\n"
        get_doc_coverage()

    with (
        patch("subprocess.run") as mock_run_no_match,
        pytest.raises(
            ValueError,
            match="Failed to determine documentation coverage from interrogate",
        ),
    ):
        mock_run_no_match.return_value.stdout = "no match in output\n"
        get_doc_coverage()

    with (
        patch("subprocess.run", side_effect=RuntimeError("Failed")),
        pytest.raises(
            ValueError,
            match="Failed to determine documentation coverage from interrogate",
        ),
    ):
        get_doc_coverage()


def test_count_shields() -> None:
    """Test count_shields accurately reports test, doc, and branch badge counts."""
    content = """# Title
[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)
[![Branch Coverage](https://img.shields.io/badge/branch_coverage-100%25-brightgreen.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)
"""
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
    with pytest.raises(SystemExit):
        parse_args(["--unknown"])


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
                """# Title
[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)
[![Branch Coverage](https://img.shields.io/badge/branch_coverage-100%25-brightgreen.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)
"""
            )
        with pytest.raises(ValueError, match="Expected 0 branch coverage shields"):
            enforce_coverage_shields(readme)

        # Missing test coverage shield
        with open(readme, "w", encoding="utf-8") as f:
            f.write("# Title\n[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)\n")
        with pytest.raises(ValueError, match="Expected exactly 1 test coverage shield"):
            enforce_coverage_shields(readme)

        # Multiple test coverage shields
        with open(readme, "w", encoding="utf-8") as f:
            f.write(
                """# Title
[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)
[![Test Coverage](https://img.shields.io/badge/test_coverage-90%25-green.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)
"""
            )
        with pytest.raises(ValueError, match="Expected exactly 1 test coverage shield"):
            enforce_coverage_shields(readme)

        # Missing doc coverage shield
        with open(readme, "w", encoding="utf-8") as f:
            f.write(
                "# Title\n[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)\n"
            )
        with pytest.raises(ValueError, match="Expected exactly 1 doc coverage shield"):
            enforce_coverage_shields(readme)

        # 0% coverage shield rejected
        with open(readme, "w", encoding="utf-8") as f:
            f.write(
                """# Title
[![Test Coverage](https://img.shields.io/badge/test_coverage-0%25-red.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)
"""
            )
        with pytest.raises(ValueError, match="Coverage shield cannot be 0%"):
            enforce_coverage_shields(readme)

        # Valid shields: exactly 1 test, 1 doc, 0 branch
        with open(readme, "w", encoding="utf-8") as f:
            f.write(
                """# Title
[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)
"""
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
    with patch("subprocess.run", side_effect=RuntimeError("Failed")):
        val = get_test_coverage(coverage_json_path="/nonexistent/path/coverage.json")
        assert val == 0.0


def test_get_coverage_metrics_temp_file_handling() -> None:
    """Test get_coverage_metrics temporary file generation and cleanup."""

    def fake_subprocess_run(cmd: list[str], **kwargs: bool | str | int | None) -> MagicMock:
        """Fake coverage json execution by dumping valid json.

        Args:
            cmd (list[str]): Command list.
            **kwargs (bool | str | int | None): Extra arguments.

        Returns:
            MagicMock: Mock subprocess completed process.
        """
        del kwargs
        if "-o" in cmd:
            out_idx = cmd.index("-o") + 1
            out_path = cmd[out_idx]
            with open(out_path, "w", encoding="utf-8") as f:
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
            return MagicMock(returncode=0)
        return MagicMock(returncode=0, stdout="actual: 100.0%\n")

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        overall, stmt, branch = get_coverage_metrics(None)
        assert overall == 100.0
        assert stmt == 100.0
        assert branch == 100.0

    def fail_remove(p: str) -> None:
        """Raise OSError on remove.

        Args:
            p: Path to remove.

        Raises:
            OSError: Always raised to simulate permission error.
        """
        del p
        raise OSError("Permission denied")

    with (
        patch("subprocess.run", side_effect=fake_subprocess_run),
        patch("os.remove", side_effect=fail_remove),
    ):
        overall, _, _ = get_coverage_metrics(None)
        assert overall == 100.0


def test_update_readme_without_coverage_path_or_sibling() -> None:
    """Test update_readme when coverage_json_path is None and sibling coverage.json does not exist."""

    def fake_subprocess_run(cmd: list[str], **kwargs: bool | str | int | None) -> MagicMock:
        """Fake coverage json execution.

        Args:
            cmd (list[str]): Command list.
            **kwargs (bool | str | int | None): Extra arguments.

        Returns:
            MagicMock: Mock subprocess completed process.
        """
        del kwargs
        if "-o" in cmd:
            out_idx = cmd.index("-o") + 1
            out_path = cmd[out_idx]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "totals": {
                            "percent_covered": 100.0,
                            "num_statements": 5,
                            "covered_lines": 5,
                            "num_branches": 0,
                            "covered_branches": 0,
                        }
                    },
                    f,
                )
            return MagicMock(returncode=0)
        return MagicMock(returncode=0, stdout="actual: 100.0%\n")

    with TemporaryDirectory() as tmpdir:
        readme_file = os.path.join(tmpdir, "README.md")
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write(
                "# Title\n[![Test Coverage](https://img.shields.io/badge/test_coverage-50%25-red.svg)](#)\n[![Doc Coverage](https://img.shields.io/badge/doc_coverage-50%25-red.svg)](#)\n"
            )
        with patch("subprocess.run", side_effect=fake_subprocess_run):
            update_readme(readme_path=readme_file, coverage_json_path=None)
        with open(readme_file, encoding="utf-8") as f:
            content = f.read()
        assert "test_coverage-100%25" in content

        # Error when coverage subprocess fails (returns 0.0 -> raises ValueError)
        with (
            patch("subprocess.run", side_effect=RuntimeError("No coverage")),
            pytest.raises(
                ValueError,
                match="Failed to determine test coverage or test coverage is 0%",
            ),
        ):
            update_readme(readme_path=readme_file, coverage_json_path=None)


def test_update_readme_missing_file() -> None:
    """Test update_readme returns gracefully when readme does not exist."""
    missing_path = "/nonexistent/path/README.md"
    update_readme(readme_path=missing_path)
    assert not os.path.exists(missing_path)


def test_update_readme_success() -> None:
    """Test update_readme properly updates shields in a markdown document."""
    initial_content = """# Project
[![Test Coverage](https://img.shields.io/badge/test_coverage-50%25-red.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-50%25-red.svg)](#)
"""
    mock_res = MagicMock(stdout="actual: 100.0%\n")
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

        with patch("subprocess.run", return_value=mock_res):
            update_readme(readme_path=readme_file, coverage_json_path=cov_file)

        with open(readme_file, encoding="utf-8") as f:
            updated = f.read()

        assert "test_coverage-100%25-brightgreen.svg" in updated
        assert "branch_coverage" not in updated
        assert "doc_coverage-100%25-brightgreen.svg" in updated
        assert count_shields(updated) == (1, 1, 0)

        # Update again with branch coverage injected to verify removal
        branch_badge = (
            "\n[![Branch Coverage](https://img.shields.io/badge/branch_coverage-100%25-brightgreen.svg)](#)\n"
        )
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write(updated + branch_badge)

        with patch("subprocess.run", return_value=mock_res):
            update_readme(readme_path=readme_file, coverage_json_path=cov_file)

        with open(readme_file, encoding="utf-8") as f:
            updated2 = f.read()

        assert "branch_coverage" not in updated2
        assert count_shields(updated2) == (1, 1, 0)


def test_update_readme_insertions_and_deduplications() -> None:
    """Test update_readme insertion scenarios and duplicate shield pruning."""
    mock_res = MagicMock(stdout="actual: 100.0%\n")
    with TemporaryDirectory() as tmpdir:
        cov_file = os.path.join(tmpdir, "coverage.json")
        with open(cov_file, "w", encoding="utf-8") as f:
            json.dump({"totals": {"percent_covered": 100.0}}, f)

        # Case 1: Only Doc Coverage present -> insert Test Coverage before Doc Coverage
        case1_file = os.path.join(tmpdir, "case1.md")
        with open(case1_file, "w", encoding="utf-8") as f:
            f.write("# Title\n[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)\n")
        with patch("subprocess.run", return_value=mock_res):
            update_readme(readme_path=case1_file, coverage_json_path=cov_file)
        with open(case1_file, encoding="utf-8") as f:
            assert count_shields(f.read()) == (1, 1, 0)

        # Case 2: Neither present, but header present -> insert after header
        case2_file = os.path.join(tmpdir, "case2.md")
        with open(case2_file, "w", encoding="utf-8") as f:
            f.write("# Title\n\nSome body text.\n")
        with patch("subprocess.run", return_value=mock_res):
            update_readme(readme_path=case2_file, coverage_json_path=cov_file)
        with open(case2_file, encoding="utf-8") as f:
            assert count_shields(f.read()) == (1, 1, 0)

        # Case 3: Neither present and no header -> insert at top
        case3_file = os.path.join(tmpdir, "case3.md")
        with open(case3_file, "w", encoding="utf-8") as f:
            f.write("Just raw text without header.\n")
        with patch("subprocess.run", return_value=mock_res):
            update_readme(readme_path=case3_file, coverage_json_path=cov_file)
        with open(case3_file, encoding="utf-8") as f:
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
        with patch("subprocess.run", return_value=mock_res):
            update_readme(readme_path=case4_file, coverage_json_path=cov_file)
        with open(case4_file, encoding="utf-8") as f:
            assert count_shields(f.read()) == (1, 1, 0)

        # Case 5: Enforcement failure after update raises ValueError
        case5_file = os.path.join(tmpdir, "case5.md")
        with open(case5_file, "w", encoding="utf-8") as f:
            f.write("# Title\n")
        with (
            patch("subprocess.run", return_value=mock_res),
            patch("scripts.update_badges.count_shields", return_value=(2, 1, 0)),
            pytest.raises(ValueError, match="Enforcement failed after update"),
        ):
            update_readme(readme_path=case5_file, coverage_json_path=cov_file)


def test_update_badges_main() -> None:
    """Test executing update_badges_main with various arguments."""
    mock_res = MagicMock(stdout="actual: 100.0%\n")
    with TemporaryDirectory() as tmpdir:
        readme_file = os.path.join(tmpdir, "README.md")
        cov_file = os.path.join(tmpdir, "coverage.json")
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write(
                """# Test
[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)
"""
            )
        with open(cov_file, "w", encoding="utf-8") as f:
            json.dump({"totals": {"percent_covered": 100.0}}, f)

        # Normal run
        with patch("subprocess.run", return_value=mock_res):
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
    mock_res = MagicMock(stdout="actual: 100.0%\n")
    try:
        with TemporaryDirectory() as tmpdir:
            readme_file = os.path.join(tmpdir, "README.md")
            cov_file = os.path.join(tmpdir, "coverage.json")
            with open(readme_file, "w", encoding="utf-8") as f:
                f.write(
                    "# Test\n[![Test Coverage](https://img.shields.io/badge/test_coverage-50%25-red.svg)](#)\n[![Doc Coverage](https://img.shields.io/badge/doc_coverage-50%25-red.svg)](#)\n"
                )
            with open(cov_file, "w", encoding="utf-8") as f:
                json.dump({"totals": {"percent_covered": 100.0}}, f)

            # Success case (exit_code == 0)
            sys.modules.pop("scripts.update_badges", None)
            with (
                patch("sys.argv", ["update_badges.py", readme_file]),
                patch("subprocess.run", return_value=mock_res),
            ):
                runpy.run_module("scripts.update_badges", run_name="__main__")

            # Failure case (exit_code != 0) invokes sys.exit
            sys.modules.pop("scripts.update_badges", None)
            with (
                patch("sys.argv", ["update_badges.py", "--enforce", "/nonexistent.md"]),
                patch("sys.exit") as mock_exit,
            ):
                runpy.run_module("scripts.update_badges", run_name="__main__")
                mock_exit.assert_called_once_with(1)
                assert mock_exit.call_count == 1
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
        abs_attrs = abs_op["attributes"]
        assert isinstance(abs_attrs, dict)
        assert "auto_pad" in abs_attrs
        assert abs_attrs["auto_pad"]["default"] == "NOTSET"
        assert abs_attrs["auto_pad"]["type"] == "str"

        assert abs_attrs["kernel_shape"]["required"] is True
        assert abs_attrs["kernel_shape"]["type"] == "List[int]"

        assert abs_attrs["rates"]["default"] == [1.0, 2.0]
        assert abs_attrs["rates"]["type"] == "List[float]"

        assert abs_attrs["strides"]["type"] == "List[str]"
        assert abs_attrs["epsilon"]["type"] == "float"
        assert abs_attrs["count"]["type"] == "int"
        assert abs_attrs["axes"]["default"] == []
        assert abs_attrs["mode"]["default"] == "nearest"
        assert abs_attrs["body"]["type"] == "Graph"
        assert abs_attrs["target_type"]["type"] == "DType"
        assert abs_attrs["untyped_attr"]["type"] == "Any"

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

        with open(json_path, encoding="utf-8") as f:
            loaded_json = json.load(f)
            assert "Abs" in loaded_json

        with open(py_path, encoding="utf-8") as f:
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
        with open(os.path.join(onnx_root_dir, "Operators.md"), "w", encoding="utf-8") as f:
            f.write(SAMPLE_ONNX_DOCS)
        with patch("sys.argv", ["generate_registry.py", "--onnx-dir", onnx_root_dir]):
            gen_main(json_path=json_path, registry_path=py_path)
        assert os.path.exists(json_path)

        # Test default md path fallback when no args provided
        with (
            patch("sys.argv", ["generate_registry.py"]),
            patch("scripts.generate_registry.parse_onnx_docs", return_value={}),
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
        assert ops["ai.onnx.preview.training.Adagrad"]["domain"] == "ai.onnx.preview.training"

    from ml_switcheroo_ir.types import JsonValue
    from scripts.generate_registry import (
        OnnxOpEntryValue,
        OnnxParamProtocol,
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
            self.attributes: dict[str, JsonValue] = {}
            self.inputs: list[OnnxParamProtocol] = []
            self.outputs: list[OnnxParamProtocol] = []
            self.domain = domain

    fake_onnx_defs = MagicMock()
    fake_onnx_defs.get_all_schemas = MagicMock(return_value=[FakeSchema("")])
    fake_onnx_pkg = MagicMock()
    fake_onnx_pkg.defs = fake_onnx_defs

    with patch.dict("sys.modules", {"onnx": fake_onnx_pkg, "onnx.defs": fake_onnx_defs}):
        test_ops: dict[str, dict[str, OnnxOpEntryValue]] = {
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

    fake_onnx_defs.get_all_schemas = MagicMock(return_value=[s_v10, s_v5])
    with patch.dict("sys.modules", {"onnx": fake_onnx_pkg, "onnx.defs": fake_onnx_defs}):
        test_ops2: dict[str, dict[str, OnnxOpEntryValue]] = {
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
        assert find_snapshots_directory(tmpdir) == Path(tmpdir).resolve()

        # Override path that is not a dir
        non_existent = os.path.join(tmpdir, "does_not_exist")
        assert find_snapshots_directory(non_existent) is None

        # ML_ECOSYSTEM_SNAPSHOTS_DIR env branch
        with patch.dict(os.environ, {"ML_ECOSYSTEM_SNAPSHOTS_DIR": tmpdir}):
            assert find_snapshots_directory() == Path(tmpdir).resolve()

        # ML_ECOSYSTEM_SNAPSHOTS_DIR non-existent branch
        with patch.dict(os.environ, {"ML_ECOSYSTEM_SNAPSHOTS_DIR": "/nonexistent/path"}):
            assert find_snapshots_directory() is not None

        # Without override, should find existing DEFAULT_SNAPSHOT_DIR
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("scripts.verify_grounding.DEFAULT_SNAPSHOT_DIR", tmpdir),
        ):
            assert find_snapshots_directory() == Path(tmpdir).resolve()

        # When DEFAULT_SNAPSHOT_DIR is non-existent, falls back to get_default_snapshots_dir()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("scripts.verify_grounding.DEFAULT_SNAPSHOT_DIR", "/nonexistent/dir"),
            patch(
                "ml_switcheroo_ir.snapshots.get_default_snapshots_dir",
                return_value=tmpdir,
            ),
        ):
            assert find_snapshots_directory() == Path(tmpdir).resolve()

        # When both DEFAULT_SNAPSHOT_DIR and get_default_snapshots_dir() are non-existent
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("scripts.verify_grounding.DEFAULT_SNAPSHOT_DIR", "/nonexistent/dir"),
            patch(
                "ml_switcheroo_ir.snapshots.get_default_snapshots_dir",
                return_value="/nonexistent/fallback",
            ),
        ):
            assert find_snapshots_directory() is None


def test_verify_grounding_stablehlo() -> None:
    """Test verify_stablehlo_grounding with missing file, invalid op, and valid snapshot."""
    from pathlib import Path

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Snapshot file missing
        errs = verify_stablehlo_grounding(tmp_path)
        assert len(errs) == 0

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
    with (
        patch("scripts.verify_grounding.find_snapshots_directory", return_value=None),
        patch(
            "scripts.verify_grounding.verify_custom_ops_grounding",
            return_value=["error"],
        ),
    ):
        assert verify_grounding_main([]) == 1

    # When snapshots directory is empty and no --snapshots-dir argument given
    with (
        TemporaryDirectory() as empty_tmpdir,
        patch(
            "scripts.verify_grounding.find_snapshots_directory",
            return_value=Path(empty_tmpdir),
        ),
    ):
        assert verify_grounding_main([]) == 0

    # When snapshots directory found with snapshots and no --snapshots-dir argument given
    with TemporaryDirectory() as populated_tmpdir:
        dummy_file = Path(populated_tmpdir) / "snapshot.json"
        dummy_file.write_text("{}", encoding="utf-8")
        with (
            patch(
                "scripts.verify_grounding.find_snapshots_directory",
                return_value=Path(populated_tmpdir),
            ),
            patch("scripts.verify_grounding.verify_stablehlo_grounding", return_value=[]),
            patch("scripts.verify_grounding.verify_mlir_grounding", return_value=[]),
            patch("scripts.verify_grounding.verify_ir_snapshot_grounding", return_value=[]),
            patch("scripts.verify_grounding.verify_custom_ops_grounding", return_value=[]),
            patch("scripts.verify_grounding.verify_onnx_grounding", return_value=[]),
        ):
            assert verify_grounding_main([]) == 0

    # When snapshots directory found and all schemas valid
    with TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "snapshot.json").write_text("{}", encoding="utf-8")
        with (
            patch(
                "scripts.verify_grounding.find_snapshots_directory",
                return_value=Path(tmpdir),
            ),
            patch("scripts.verify_grounding.verify_stablehlo_grounding", return_value=[]),
            patch("scripts.verify_grounding.verify_mlir_grounding", return_value=[]),
            patch("scripts.verify_grounding.verify_ir_snapshot_grounding", return_value=[]),
            patch("scripts.verify_grounding.verify_custom_ops_grounding", return_value=[]),
            patch("scripts.verify_grounding.verify_onnx_grounding", return_value=[]),
        ):
            assert verify_grounding_main(["--snapshots-dir", tmpdir]) == 0

        # When snapshots directory found but errors detected
        with (
            patch(
                "scripts.verify_grounding.find_snapshots_directory",
                return_value=Path(tmpdir),
            ),
            patch(
                "scripts.verify_grounding.verify_stablehlo_grounding",
                return_value=["stablehlo error"],
            ),
        ):
            assert verify_grounding_main(["--snapshots-dir", tmpdir]) == 1


def test_verify_grounding_runpy_main() -> None:
    """Test executing scripts.verify_grounding as __main__ module."""
    orig_mod = sys.modules["scripts.verify_grounding"]
    sys.modules.pop("scripts.verify_grounding", None)
    try:
        with (
            TemporaryDirectory() as empty_tmpdir,
            patch.dict(os.environ, {"ML_ECOSYSTEM_SNAPSHOTS_DIR": empty_tmpdir}),
            patch("sys.argv", ["verify_grounding.py"]),
            patch("sys.exit") as mock_exit,
        ):
            runpy.run_module("scripts.verify_grounding", run_name="__main__")
            mock_exit.assert_called_once_with(0)
            assert mock_exit.call_count == 1
    finally:
        sys.modules["scripts.verify_grounding"] = orig_mod


def test_find_snapshots_directory_env_and_override() -> None:
    """Test find_snapshots_directory with environment variable and explicit override."""
    with TemporaryDirectory() as tmpdir:
        # Explicit override
        assert find_snapshots_directory(tmpdir) == Path(tmpdir).resolve()
        assert find_snapshots_directory("/non/existent/path") is None

        # Repo-root override with nested src/ml_ecosystem_snapshots/snapshots
        eco_root = Path(tmpdir) / "eco_repo"
        nested_eco = eco_root / "src" / "ml_ecosystem_snapshots" / "snapshots"
        nested_eco.mkdir(parents=True)
        assert find_snapshots_directory(str(eco_root)) == nested_eco.resolve()

        # Sibling directory when non-existent
        empty_ws = Path(tmpdir) / "empty_ws" / "ml-ecosystem-snapshots"
        assert find_snapshots_directory(str(empty_ws)) is None

        # Environment variable valid and invalid
        with patch.dict(os.environ, {"ML_ECOSYSTEM_SNAPSHOTS_DIR": tmpdir}):
            assert find_snapshots_directory() == Path(tmpdir).resolve()
        # Env var set to nonexistent dir
        with (
            patch.dict(os.environ, {"ML_ECOSYSTEM_SNAPSHOTS_DIR": "/invalid/nonexistent/dir"}),
            patch("scripts.verify_grounding.DEFAULT_SNAPSHOT_DIR", tmpdir),
        ):
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

        from ml_switcheroo_ir.schema.onnx_registry import OpSchema

        bad_mlir_op = OpSchema("arith.ungrounded_op", "mlir.arith", 1, {}, [], [])
        with patch.dict("scripts.verify_grounding.MLIR_REGISTRY", {"arith.ungrounded_op": bad_mlir_op}):
            errs_ungrounded = verify_mlir_grounding(tmppath)
            assert any("is not grounded in mlir_v0.4.30.json" in e for e in errs_ungrounded)


def test_verify_ir_snapshot_grounding_paths() -> None:
    """Test verify_ir_snapshot_grounding with missing file, valid snapshot, and defective entries."""
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # Missing file and no candidates returns []
        assert verify_ir_snapshot_grounding(tmppath) == []

        # Test candidate fallback with ir_v0.1.0.json when ir_v0.0.4.json is missing
        candidate_json = tmppath / "ir_v0.1.0.json"
        with open(candidate_json, "w", encoding="utf-8") as f:
            json.dump({"categories": {"classes": [], "functions": []}}, f)
        assert verify_ir_snapshot_grounding(tmppath) == []

        # File with unresolvable classes and functions
        ir_json = tmppath / "ir_v0.0.4.json"
        with open(ir_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categories": {
                        "classes": [
                            {"api_path": ""},
                            {"api_path": "ml_switcheroo_ir.LogicalNode"},
                            {"api_path": "ml_switcheroo_ir.LogicalNode.clone"},
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
    from ml_switcheroo_ir.validation.grounding import SnapshotManifestNode

    rec2: dict[str, SnapshotManifestNode] = {
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
    """Test verify_rdna_grounding with missing file, strict mode, invalid entries, and VOPD mismatches."""
    from ml_switcheroo_ir.snapshots import get_default_snapshots_dir

    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        assert verify_rdna_grounding(tmppath) == []
        assert any("Missing required RDNA snapshot" in e for e in verify_rdna_grounding(tmppath, strict=True))

        rdna_file = tmppath / "amd_rdna_snapshot.json"
        with open(rdna_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categories": {
                        "instructions": [
                            "not_a_dict",
                            {"name": ""},
                            {"name": "V_BAD_SLOT", "vopd_slot": "INVALID"},
                            {"name": "V_DUAL_FMAC_F32", "vopd_slot": "X"},
                            {"name": "V_FMAC_F32", "vopd_target": "V_DUAL_ADD_F32"},
                        ]
                    }
                },
                f,
            )
        errs = verify_rdna_grounding(tmppath)
        assert any("missing name/mnemonic" in e for e in errs)
        assert any("invalid vopd_slot" in e for e in errs)
        assert any("VOPD slot mismatch" in e for e in errs)
        assert any("VOPD target mismatch" in e for e in errs)
        assert any("is not grounded" in e for e in errs)

    live_dir = Path(get_default_snapshots_dir())
    assert verify_rdna_grounding(live_dir, strict=True) == []


def test_verify_grounding_sass() -> None:
    """Test verify_sass_grounding with missing file, strict mode, invalid entries, and latency mismatches."""
    from ml_switcheroo_ir.snapshots import get_default_snapshots_dir

    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        assert verify_sass_grounding(tmppath) == []
        assert any("Missing required SASS snapshot" in e for e in verify_sass_grounding(tmppath, strict=True))

        sass_file = tmppath / "nvidia_sass_snapshot.json"
        with open(sass_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categories": {
                        "instructions": [
                            "not_a_dict",
                            {"name": ""},
                            {"name": "FFMA_BAD", "execution_latency": "bad_latency"},
                            {"name": "HMMA", "execution_latency": 999},
                        ]
                    }
                },
                f,
            )
        errs = verify_sass_grounding(tmppath)
        assert any("missing name/mnemonic" in e for e in errs)
        assert any("invalid execution_latency" in e for e in errs)
        assert any("SASS latency mismatch for 'HMMA'" in e for e in errs)
        assert any("is not grounded" in e for e in errs)

    live_dir = Path(get_default_snapshots_dir())
    assert verify_sass_grounding(live_dir, strict=True) == []


def test_verify_grounding_wgsl() -> None:
    """Test verify_wgsl_grounding with missing file, strict mode, defective ops, and live snapshot."""
    from ml_switcheroo_ir.snapshots import get_default_snapshots_dir

    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        assert verify_wgsl_grounding(tmppath) == []
        assert any("Missing required WGSL snapshot" in e for e in verify_wgsl_grounding(tmppath, strict=True))

        wgsl_file = tmppath / "wgsl_ops.json"
        wgsl_file.write_text(
            json.dumps(
                {
                    "ops": [
                        {"name": ""},
                        {"name": "bad_dom_op", "domain": "invalid_domain"},
                        {"name": "workgroupBarrier", "domain": "wgsl"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        errs = verify_wgsl_grounding(tmppath)
        assert any("missing name" in e for e in errs)
        assert any("invalid domain" in e for e in errs)
        assert any("is not grounded" in e for e in errs)

    live_dir = Path(get_default_snapshots_dir())
    assert verify_wgsl_grounding(live_dir, strict=True) == []


def test_verify_grounding_ptx() -> None:
    """Test verify_ptx_grounding with missing file, strict mode, invalid items, and live snapshot."""
    from ml_switcheroo_ir.snapshots import get_default_snapshots_dir

    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        assert verify_ptx_grounding(tmppath) == []
        assert any("Missing required PTX snapshot" in e for e in verify_ptx_grounding(tmppath, strict=True))

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
                        ],
                    }
                },
                f,
            )
        errs = verify_ptx_grounding(tmppath)
        assert any("missing identifier" in e for e in errs)
        assert any("PTX op 'fma' is not grounded" in e for e in errs)

    live_dir = Path(get_default_snapshots_dir())
    assert verify_ptx_grounding(live_dir, strict=True) == []


def test_verify_grounding_metal() -> None:
    """Test verify_metal_grounding with missing file, strict mode, invalid items, and live snapshot."""
    from ml_switcheroo_ir.snapshots import get_default_snapshots_dir

    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        assert verify_metal_grounding(tmppath) == []
        assert any("Missing required Metal snapshot" in e for e in verify_metal_grounding(tmppath, strict=True))

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
                        ],
                    }
                },
                f,
            )
        errs = verify_metal_grounding(tmppath)
        assert any("missing identifier" in e for e in errs)
        assert any("Metal op 'simdgroup_multiply_accumulate' is not grounded" in e for e in errs)

    live_dir = Path(get_default_snapshots_dir())
    assert verify_metal_grounding(live_dir, strict=True) == []


def test_verify_grounding_wasm() -> None:
    """Test verify_wasm_grounding with missing file, strict mode, invalid items, and live snapshot."""
    from ml_switcheroo_ir.snapshots import get_default_snapshots_dir

    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        assert verify_wasm_grounding(tmppath) == []
        assert any("Missing required WASM snapshot" in e for e in verify_wasm_grounding(tmppath, strict=True))

        wasm_file = tmppath / "wasm_snapshot.json"
        with open(wasm_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categories": {
                        "metadata": "non_list",
                        "instructions": [
                            "not_a_dict",
                            {"name": ""},
                            {"api_path": "wasm.i32x4.add"},
                        ],
                    }
                },
                f,
            )
        errs = verify_wasm_grounding(tmppath)
        assert any("missing identifier" in e for e in errs)
        assert any("WASM op 'f32x4.add' is not grounded" in e for e in errs)

    live_dir = Path(get_default_snapshots_dir())
    assert verify_wasm_grounding(live_dir, strict=True) == []


def test_verify_grounding_array_api() -> None:
    """Test verify_array_api_grounding with missing file, defective ops, and unconditional live snapshot."""
    from ml_switcheroo_ir.schema.base import OpAttribute, OpSchema
    from ml_switcheroo_ir.snapshots import get_default_snapshots_dir

    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        assert verify_array_api_grounding(tmppath) == []

        array_api_file = tmppath / "array_api_v2024.12.json"
        with open(array_api_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "categories": {
                        "array": [
                            {
                                "name": "abs",
                                "api_path": "array_api.abs",
                                "params": [{"name": "wrong_input", "kind": "POSITIONAL_ONLY"}],
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
        assert any("is not grounded" in e for e in errs)
        assert any("does not match snapshot parameter" in e for e in errs)
        assert any("attribute" in e and "is not recognized" in e for e in errs)

        live_dir = Path(get_default_snapshots_dir())
        assert verify_array_api_grounding(live_dir, strict=True) == []

        abs_schema = OpSchema(
            name="abs",
            domain="array_api",
            attributes={"axis": OpAttribute(name="axis", type="int", required=False)},
            inputs=[],
            outputs=["out"],
        )
        sum_schema = OpSchema(name="sum", domain="array_api", attributes={}, inputs=[], outputs=["out"])
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
            {"abs": abs_schema, "sum": sum_schema},
            clear=True,
        ):
            assert verify_array_api_grounding(tmppath) == []


def test_verify_grounding_aten() -> None:
    """Test verify_aten_grounding with missing file, defective ops, and unconditional live snapshot."""
    from ml_switcheroo_ir.schema.base import OpAttribute, OpSchema
    from ml_switcheroo_ir.snapshots import get_default_snapshots_dir

    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        assert verify_aten_grounding(tmppath) == []

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
        assert any("is not grounded" in e for e in errs)
        assert any("attribute 'alpha' is not recognized" in e for e in errs)

        live_dir = Path(get_default_snapshots_dir())
        assert verify_aten_grounding(live_dir, strict=True) == []

        add_schema = OpSchema(
            name="add",
            domain="aten",
            attributes={"alpha": OpAttribute(name="alpha", type="float", required=False)},
            inputs=[],
            outputs=["out"],
        )
        mm_schema = OpSchema(name="matmul", domain="aten", attributes={}, inputs=[], outputs=["out"])
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
            {"add": add_schema, "matmul": mm_schema},
            clear=True,
        ):
            assert verify_aten_grounding(tmppath) == []


def test_verify_grounding_collectives() -> None:
    """Test verify_collectives_grounding with missing file, defective ops, and unconditional live snapshot."""
    from ml_switcheroo_ir.schema.base import OpAttribute, OpSchema
    from ml_switcheroo_ir.snapshots import get_default_snapshots_dir

    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        assert verify_collectives_grounding(tmppath) == []

        nccl_file = tmppath / "nccl_v2.21.json"
        with open(nccl_file, "w", encoding="utf-8") as f:
            json.dump(
                {"categories": {"array": [{"name": "broadcast", "api_path": "nccl.broadcast"}]}},
                f,
            )
        errs = verify_collectives_grounding(tmppath)
        assert len(errs) >= 4
        assert any("is not grounded" in e for e in errs)

        with patch.dict("scripts.verify_grounding.COLLECTIVE_OPS_REGISTRY", {}, clear=True):
            errs_missing_reg = verify_collectives_grounding(tmppath)
            assert any("COLLECTIVE_OPS_REGISTRY is empty." in e for e in errs_missing_reg)

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
        no_red_ar = OpSchema(
            name="collective.all_reduce",
            domain="collective",
            attributes={"comm": OpAttribute(name="comm", type="str", required=False)},
            aliases=("all_reduce",),
        )
        no_comm_ar = OpSchema(
            name="collective.all_reduce",
            domain="collective",
            attributes={"reduction_op": OpAttribute(name="reduction_op", type="str", required=True)},
            aliases=("all_reduce",),
        )
        with patch.dict("scripts.verify_grounding.COLLECTIVE_OPS_REGISTRY", {"collective.all_reduce": no_red_ar}):
            errs_no_red = verify_collectives_grounding(tmppath)
            assert any("schema missing 'reduction_op' attribute" in e for e in errs_no_red)

        with patch.dict("scripts.verify_grounding.COLLECTIVE_OPS_REGISTRY", {"collective.all_reduce": no_comm_ar}):
            errs_no_comm = verify_collectives_grounding(tmppath)
            assert any("schema missing 'comm' communicator attribute" in e for e in errs_no_comm)

        live_dir = Path(get_default_snapshots_dir())
        assert verify_collectives_grounding(live_dir, strict=True) == []


def test_verify_grounding_custom_attention() -> None:
    """Test verify_custom_ops_grounding with bound flash_attention symbol signatures and edge cases."""
    from ml_switcheroo_ir.schema.base import OpAttribute, OpSchema
    from ml_switcheroo_ir.snapshots import get_default_snapshots_dir

    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        flash_file = tmppath / "flash_attention_v2.6.3.json"

        flash_file.write_text("{invalid_json", encoding="utf-8")
        errs_bad_json = verify_custom_ops_grounding(tmppath)
        assert any("Failed parsing flash_attention snapshot" in e for e in errs_bad_json)

        flash_file.write_text(
            json.dumps({"categories": {"neural_ops": ["not_a_dict"], "util": "not_a_list"}}),
            encoding="utf-8",
        )
        errs_empty_syms = verify_custom_ops_grounding(tmppath)
        assert any("contains no symbols" in e for e in errs_empty_syms)

        flash_file.write_text(
            json.dumps(
                {
                    "categories": {
                        "neural_ops": [
                            {
                                "name": "FlashAttention",
                                "api_path": "ml.switcheroo.custom.FlashAttention",
                                "params": [
                                    {"name": "Q", "kind": "POSITIONAL_ONLY"},
                                    {"name": "K", "kind": "POSITIONAL_ONLY"},
                                    {"name": "V", "kind": "POSITIONAL_ONLY"},
                                    {"name": "causal", "kind": "KEYWORD_ONLY"},
                                ],
                                "kwargs": ["causal"],
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        bad_fa = OpSchema(
            name="FlashAttention",
            domain="ml.switcheroo.custom",
            inputs=["wrong_q", "wrong_k"],
            outputs=["out"],
            attributes={"bad_attr": OpAttribute(name="bad_attr", type="bool", required=False)},
        )
        with patch.dict("scripts.verify_grounding.CUSTOM_OPS_REGISTRY", {"FlashAttention": bad_fa}):
            errs_mismatch = verify_custom_ops_grounding(tmppath)
            assert any("inputs ['wrong_q', 'wrong_k'] do not match" in e for e in errs_mismatch)
            assert any("attribute 'bad_attr' is not grounded" in e for e in errs_mismatch)

        live_dir = Path(get_default_snapshots_dir())
        assert verify_custom_ops_grounding(live_dir, strict=True) == []

        with patch.dict("scripts.verify_grounding.CUSTOM_OPS_REGISTRY", {}, clear=True):
            errs_missing = verify_custom_ops_grounding(tmppath)
            assert any("CUSTOM_OPS_REGISTRY is empty." in e for e in errs_missing)


def test_verify_grounding_main_ecosystem_and_strict() -> None:
    """Test verify_grounding main function with --ecosystem-snapshots-dir, --require-snapshots-dir, and --strict."""
    from ml_switcheroo_ir.snapshots import get_default_snapshots_dir

    live_dir = get_default_snapshots_dir()
    assert live_dir != ""

    assert verify_grounding_main(["--strict", "--snapshots-dir", "/nonexistent/dir"]) == 1
    assert verify_grounding_main(["--require-snapshots-dir", "--snapshots-dir", "/nonexistent/dir"]) == 1

    with TemporaryDirectory() as empty_dir:
        assert verify_grounding_main(["--strict", "--snapshots-dir", empty_dir]) == 1
        with patch(
            "scripts.verify_grounding.find_snapshots_directory",
            return_value=Path(empty_dir),
        ):
            assert verify_grounding_main(["--strict"]) == 1
            assert verify_grounding_main([]) == 0

    assert verify_grounding_main(["--strict", "--ecosystem-snapshots-dir", live_dir]) == 0
    assert (
        verify_grounding_main(
            ["--strict", "--framework-snapshots-dir", "/nonexistent", "--ecosystem-snapshots-dir", live_dir]
        )
        == 0
    )
    assert (
        verify_grounding_main(
            [
                "--strict",
                "--snapshots-dir",
                "../ml-framework-snapshots",
                "--ecosystem-snapshots-dir",
                "../ml-ecosystem-snapshots",
            ]
        )
        == 0
    )


def test_verify_grounding_strict_fault_injection_all_21_registries() -> None:
    """Prove verify_grounding --strict catches injected ungrounded ops in every single dialect registry."""
    from ml_switcheroo_ir.schema import (
        ARRAY_API_REGISTRY,
        ATEN_REGISTRY,
        CHEX_REGISTRY,
        CUSTOM_OPS_REGISTRY,
        GRAIN_REGISTRY,
        METAL_REGISTRY,
        MLIR_REGISTRY,
        ODL_CATALOG,
        ONNX_REGISTRY,
        OPTAX_REGISTRY,
        ORBAX_REGISTRY,
        PTX_REGISTRY,
        RDNA_REGISTRY,
        SASS_REGISTRY,
        STABLEHLO_REGISTRY,
        STATE_OPS_REGISTRY,
        WASM_REGISTRY,
        WEBGL_REGISTRY,
        WGSL_REGISTRY,
        override_registry,
    )
    from ml_switcheroo_ir.schema.base import OpSchema
    from ml_switcheroo_ir.schema.custom_ops import COLLECTIVE_OPS_REGISTRY, QUANTIZATION_OPS_REGISTRY
    from ml_switcheroo_ir.snapshots import get_default_snapshots_dir
    from scripts.verify_grounding import (
        _run_all_snapshot_verifiers,
        verify_array_api_grounding,
        verify_aten_grounding,
        verify_chex_grounding,
        verify_collectives_grounding,
        verify_custom_ops_grounding,
        verify_grain_grounding,
        verify_metal_grounding,
        verify_mlir_grounding,
        verify_odl_grounding,
        verify_onnx_grounding,
        verify_optax_grounding,
        verify_orbax_grounding,
        verify_ptx_grounding,
        verify_quantization_grounding,
        verify_rdna_grounding,
        verify_sass_grounding,
        verify_stablehlo_grounding,
        verify_state_ops_grounding,
        verify_translation_grounding,
        verify_wasm_grounding,
        verify_webgl_grounding,
        verify_wgsl_grounding,
    )

    live_dir = Path(get_default_snapshots_dir())
    assert _run_all_snapshot_verifiers(live_dir, strict=True) == []

    all_21_targets = (
        (ONNX_REGISTRY, "HallucinatedOnnxOp", "ai.onnx", verify_onnx_grounding),
        (STABLEHLO_REGISTRY, "hallucinated_shlo_op", "stablehlo", verify_stablehlo_grounding),
        (MLIR_REGISTRY, "arith.hallucinated_mlir_op", "mlir.arith", verify_mlir_grounding),
        (CUSTOM_OPS_REGISTRY, "HallucinatedCustomOp", "ml.switcheroo.custom", verify_custom_ops_grounding),
        (COLLECTIVE_OPS_REGISTRY, "collective.hallucinated_coll", "collective", verify_collectives_grounding),
        (QUANTIZATION_OPS_REGISTRY, "quantization.hallucinated_quant", "quantization", verify_quantization_grounding),
        (STATE_OPS_REGISTRY, "HallucinatedStateOp", "ml.switcheroo.state", verify_state_ops_grounding),
        (ATEN_REGISTRY, "hallucinated_aten_op", "aten", verify_aten_grounding),
        (ARRAY_API_REGISTRY, "hallucinated_arr_op", "array_api", verify_array_api_grounding),
        (CHEX_REGISTRY, "hallucinated_chex_op", "chex", verify_chex_grounding),
        (GRAIN_REGISTRY, "hallucinated_grain_op", "grain", verify_grain_grounding),
        (ORBAX_REGISTRY, "HallucinatedOrbaxOp", "orbax", verify_orbax_grounding),
        (OPTAX_REGISTRY, "hallucinated_optax_op", "optax", verify_optax_grounding),
        (ODL_CATALOG, "hallucinated_odl_op", "odl", verify_odl_grounding),
        (RDNA_REGISTRY, "V_HALLUCINATED_RDNA_F32", "amd_rdna", verify_rdna_grounding),
        (SASS_REGISTRY, "HALLUCINATED_SASS_OP", "nvidia_sass", verify_sass_grounding),
        (WGSL_REGISTRY, "hallucinatedWgslOp", "wgsl", verify_wgsl_grounding),
        (PTX_REGISTRY, "hallucinated_ptx_op", "nvidia_ptx", verify_ptx_grounding),
        (METAL_REGISTRY, "hallucinated_metal_op", "metal_msl", verify_metal_grounding),
        (WASM_REGISTRY, "f32x4.hallucinated_wasm", "wasm_simd", verify_wasm_grounding),
        (WEBGL_REGISTRY, "hallucinatedWebglOp", "webgl", verify_webgl_grounding),
    )
    assert len(all_21_targets) == 21

    for target_reg, bad_name, dom, verifier_fn in all_21_targets:
        injected = OpSchema(name=bad_name, domain=dom, inputs=["x"], outputs=["y"])
        with override_registry(target_reg, overrides={bad_name: injected}):
            errs = verifier_fn(live_dir, strict=True)
            assert errs, f"Fault injection into {dom} ({bad_name}) was NOT caught!"
            assert any(bad_name in e for e in errs), f"Expected '{bad_name}' in {errs}"

    with override_registry(ONNX_REGISTRY, overrides={"HallucinatedTopOp": OpSchema("HallucinatedTopOp", "ai.onnx")}):
        top_errs = _run_all_snapshot_verifiers(live_dir, strict=True)
        assert any("HallucinatedTopOp" in e for e in top_errs)
    assert verify_translation_grounding(live_dir, strict=True) == []


def test_verify_stablehlo_grounding_exhaustive_paths(tmp_path: Path) -> None:
    """Test exhaustive stablehlo paths in verify_stablehlo_grounding.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    from ml_switcheroo_ir.schema.onnx_registry import OpSchema
    from scripts.verify_grounding import verify_stablehlo_grounding

    dummy_shlo = {"add": OpSchema("add", "stablehlo", 1, {}, ["lhs", "rhs"], ["result"])}
    snap_payload = json.dumps({"operations": [{"name": "add", "api_path": "stablehlo.add"}]})
    with patch("scripts.verify_grounding.STABLEHLO_REGISTRY", dummy_shlo):
        (tmp_path / "stablehlo_exhaustive.json").write_text(snap_payload, encoding="utf-8")
        assert verify_stablehlo_grounding(tmp_path) == []

        (tmp_path / "stablehlo_exhaustive.json").unlink()
        fw_dir = tmp_path.parent / "frameworks"
        fw_dir.mkdir(parents=True, exist_ok=True)
        (fw_dir / "stablehlo_exhaustive.json").write_text(snap_payload, encoding="utf-8")
        assert verify_stablehlo_grounding(tmp_path) == []

        import importlib.util

        eco_snap_dir = tmp_path / "ml-ecosystem-snapshots" / "snapshots"
        eco_snap_dir.mkdir(parents=True, exist_ok=True)
        spec_ok = importlib.util.spec_from_file_location(
            "ml_ecosystem_snapshots.frameworks", str(fw_dir / "__init__.py")
        )
        assert spec_ok is not None
        fake_eco_fw = importlib.util.module_from_spec(spec_ok)
        fake_eco_root = MagicMock(frameworks=fake_eco_fw)
        with patch.dict(
            "sys.modules",
            {"ml_ecosystem_snapshots": fake_eco_root, "ml_ecosystem_snapshots.frameworks": fake_eco_fw},
        ):
            assert verify_stablehlo_grounding(eco_snap_dir) == []
        (fw_dir / "stablehlo_exhaustive.json").unlink()
        with patch.dict("sys.modules", {"ml_ecosystem_snapshots": None, "ml_ecosystem_snapshots.frameworks": None}):
            assert verify_stablehlo_grounding(eco_snap_dir) == []
            assert any("Missing required StableHLO" in e for e in verify_stablehlo_grounding(eco_snap_dir, strict=True))
        spec_bad = importlib.util.spec_from_file_location(
            "ml_ecosystem_snapshots.frameworks", str(tmp_path / "nonexistent_pkg" / "__init__.py")
        )
        assert spec_bad is not None
        fake_eco_fw_bad = importlib.util.module_from_spec(spec_bad)
        fake_eco_root_bad = MagicMock(frameworks=fake_eco_fw_bad)
        with patch.dict(
            "sys.modules",
            {"ml_ecosystem_snapshots": fake_eco_root_bad, "ml_ecosystem_snapshots.frameworks": fake_eco_fw_bad},
        ):
            assert verify_stablehlo_grounding(eco_snap_dir) == []

        from ml_switcheroo_ir.snapshots import resolve_snapshots_path

        eco_repo = tmp_path / "eco_repo_populated"
        pop_eco = eco_repo / "src" / "ml_ecosystem_snapshots" / "snapshots"
        pop_eco.mkdir(parents=True, exist_ok=True)
        (pop_eco / "onnx.json").write_text("{}", encoding="utf-8")
        assert resolve_snapshots_path(eco_repo) == pop_eco.resolve()

        eco_empty = tmp_path / "eco_repo_empty"
        empty_eco = eco_empty / "src" / "ml_ecosystem_snapshots" / "snapshots"
        empty_eco.mkdir(parents=True, exist_ok=True)
        assert resolve_snapshots_path(eco_empty) == empty_eco.resolve()

        assert resolve_snapshots_path(tmp_path / "nonexistent_dir") is None


def test_verify_quantization_state_and_odl_empty_and_mismatch() -> None:
    """Test empty and mismatched registry diagnostics in quantization, state, and ODL verifiers."""
    from ml_switcheroo_ir.schema.onnx_registry import OpSchema
    from scripts.verify_grounding import (
        verify_odl_grounding,
        verify_quantization_grounding,
        verify_state_ops_grounding,
    )

    bad_quant = OpSchema("wrong_q", "bad_dom", 1, {}, [], [])
    with patch("scripts.verify_grounding.QUANTIZATION_OPS_REGISTRY", {}):
        assert "QUANTIZATION_OPS_REGISTRY is empty." in verify_quantization_grounding()
    with patch(
        "scripts.verify_grounding.QUANTIZATION_OPS_REGISTRY",
        {"block_quantize": bad_quant},
    ):
        q_errs = verify_quantization_grounding()
        assert len(q_errs) == 2

    bad_state = OpSchema("wrong_s", "bad_dom", 1, {}, [], [])
    with patch("scripts.verify_grounding.STATE_OPS_REGISTRY", {}):
        assert "STATE_OPS_REGISTRY is empty." in verify_state_ops_grounding()
    with patch(
        "scripts.verify_grounding.STATE_OPS_REGISTRY",
        {"ReadVariable": bad_state},
    ):
        s_errs = verify_state_ops_grounding()
        assert len(s_errs) == 2

    bad_odl = OpSchema("wrong_o", "bad_dom", 1, {}, [], [])
    with patch("scripts.verify_grounding.ODL_CATALOG", {}):
        assert "ODL_CATALOG is empty." in verify_odl_grounding()
    with patch(
        "scripts.verify_grounding.ODL_CATALOG",
        {"add": bad_odl},
    ):
        o_errs = verify_odl_grounding()
        assert len(o_errs) == 3


def test_verify_domain_registries_grounding(tmp_path: Path) -> None:
    """Test Chex, Grain, Orbax, and Optax grounding verification paths.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    from ml_switcheroo_ir.schema.onnx_registry import OpAttribute, OpSchema
    from scripts.verify_grounding import (
        verify_chex_grounding,
        verify_grain_grounding,
        verify_optax_grounding,
        verify_orbax_grounding,
    )

    with patch("scripts.verify_grounding.CHEX_REGISTRY", {}):
        assert "CHEX_REGISTRY is empty." in verify_chex_grounding()
    with patch("scripts.verify_grounding.GRAIN_REGISTRY", {}):
        assert "GRAIN_REGISTRY is empty." in verify_grain_grounding()
    with patch("scripts.verify_grounding.ORBAX_REGISTRY", {}):
        assert "ORBAX_REGISTRY is empty." in verify_orbax_grounding()
    with patch("scripts.verify_grounding.OPTAX_REGISTRY", {}):
        assert "OPTAX_REGISTRY is empty." in verify_optax_grounding()

    snap_sub = tmp_path / "domain_snaps"
    snap_sub.mkdir(exist_ok=True)
    (snap_sub / "optax_v0.2.8.json").write_text(
        json.dumps(
            {
                "categories": {
                    "functions": [
                        {
                            "name": "adam",
                            "api_path": "optax.adam",
                            "params": [{"name": "learning_rate", "kind": "POSITIONAL_OR_KEYWORD"}],
                        },
                        {
                            "name": "no_params_op",
                            "api_path": "optax.no_params_op",
                            "params": [],
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    bad_optax = {
        "adam": OpSchema(
            "wrong_name",
            "bad_dom",
            1,
            {
                "bad_attr": OpAttribute("bad_attr", "float", False),
                "learning_rate": OpAttribute("learning_rate", "float", True),
            },
            [],
            [],
        ),
        "no_params_op": OpSchema(
            "no_params_op",
            "optax",
            1,
            {"ignored_attr": OpAttribute("ignored_attr", "float", False)},
            [],
            [],
        ),
        "missing_op": OpSchema("missing_op", "optax", 1, {}, [], []),
    }
    with patch("scripts.verify_grounding.OPTAX_REGISTRY", bad_optax):
        optax_errs = verify_optax_grounding(snap_sub)
        assert any("op key 'adam' != schema.name 'wrong_name'" in e for e in optax_errs)
        assert any("invalid domain 'bad_dom'" in e for e in optax_errs)
        assert any("attribute 'bad_attr' is not recognized" in e for e in optax_errs)
        assert any("op 'missing_op' is not grounded" in e for e in optax_errs)


def test_lint_typing_exhaustive(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test scripts/lint_typing.py across clean and violating Python constructs.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    from scripts.lint_typing import check_source, collect_python_files, main

    clean_src = (
        "from typing import Protocol\n"
        "from builtins import int\n"
        "import os\n"
        "import typing\n"
        "import builtins\n"
        "x: typing.Sequence[int] = [1, 2]\n"
        "y = builtins.len(x)\n"
        "z = repr(x)\n"
        "p = os.path.join('a', 'b')\n"
        "class C:\n"
        "  object = 'object'\n"
        "  def __init__(self) -> None:\n"
        "    super().__init__()\n"
        "  def __eq__(self, other: int | str) -> bool:\n"
        "    return False\n"
        "# normal comment\n"
    )
    assert check_source(clean_src, Path("clean.py")) == []

    bad_src = (
        "from typing import Any, cast, Sequence\n"
        "from builtins import getattr, len\n"
        "from operator import attrgetter\n"
        "import typing\n"
        "import builtins\n"
        "import operator\n"
        "a: typing.Any = 1\n"
        "b = builtins.setattr\n"
        "c = object.__setattr__\n"
        "d = getattr(a, 'x', None)\n"
        "e: Any = 2\n"
        "g = a.__repr__()\n"
        "h = operator.itemgetter('x')\n"
        "i = core_schema.any_schema()\n"
        "f = 3  # " + "type: ignore[assignment]\n"
    )
    violations = check_source(bad_src, Path("bad.py"))
    assert len(violations) == 15
    assert any("Forbidden import 'typing.Any'" in v for v in violations)
    assert any("Forbidden import 'typing.cast'" in v for v in violations)
    assert any("Forbidden import 'builtins.getattr'" in v for v in violations)
    assert any("Forbidden import 'operator.attrgetter'" in v for v in violations)
    assert any("Forbidden reference 'typing.Any'" in v for v in violations)
    assert any("Forbidden builtin reference 'builtins.setattr'" in v for v in violations)
    assert any("Forbidden dunder attribute access '.__setattr__'" in v for v in violations)
    assert any("Forbidden dunder attribute access '.__repr__'" in v for v in violations)
    assert any("Forbidden reflection/cast call 'getattr()'" in v for v in violations)
    assert any("Forbidden reference to 'Any'" in v for v in violations)
    assert any("Forbidden reference to 'object'" in v for v in violations)
    assert any("Forbidden direct dunder method call '.__repr__()'" in v for v in violations)
    assert any("Forbidden operator helper 'operator.itemgetter'" in v for v in violations)
    assert any("Forbidden untyped schema reference '.any_schema'" in v for v in violations)
    assert any("Forbidden '# type: ignore' comment" in v for v in violations)

    sub_dir = tmp_path / "pkg"
    sub_dir.mkdir()
    clean_file = sub_dir / "ok.py"
    clean_file.write_text("x: int = 1\n", encoding="utf-8")
    txt_file = sub_dir / "notes.txt"
    txt_file.write_text("ignored\n", encoding="utf-8")

    collected = collect_python_files([clean_file, sub_dir, txt_file, tmp_path / "missing.py"])
    assert collected == [clean_file]

    assert main([str(clean_file)]) == 0

    bad_file = sub_dir / "bad.py"
    bad_file.write_text("x = getattr(object(), '__doc__', None)\n", encoding="utf-8")
    assert main([str(sub_dir)]) == 1
    captured = capsys.readouterr()
    assert "Forbidden reflection/cast call 'getattr()'" in captured.err

    with patch("sys.argv", ["lint_typing.py"]):
        assert main(None) == 0


def test_check_docs_exhaustive(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test scripts/check_docs.py across all AST docstring verification branches.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    import ast as _ast

    from scripts.check_docs import _has_non_none_return, check_directory, check_file, main

    assert _has_non_none_return(None) is False
    assert _has_non_none_return(_ast.Name(id="None", ctx=_ast.Load())) is False
    assert _has_non_none_return(_ast.Constant(value=None)) is False

    assert check_directory(str(tmp_path / "nonexistent")) == []

    txt_file = tmp_path / "ignore.txt"
    txt_file.write_text("not python\n", encoding="utf-8")
    assert check_directory(str(txt_file)) == []

    bad_py = tmp_path / "bad.py"
    bad_py.write_text(
        "class MissingClassDoc:\n"
        "    pass\n"
        "def missing_func_doc():\n"
        "    pass\n"
        "from typing import overload\n"
        "import typing\n"
        "@overload\n"
        "def ov1(x: int) -> int: ...\n"
        "@typing.overload\n"
        "def ov2(x: str) -> str: ...\n"
        "@staticmethod\n"
        "@typing.final\n"
        "def bad_doc(pos_only: int, /, a: int, *args: int, kw_only: int = 1, **kwargs: int) -> int:\n"
        '    """Doc with (' + "Any) and (" + 'object) but no sections."""\n'
        "    return 1\n"
        "class WithMethod:\n"
        '    """Class doc."""\n'
        "    def method(self, x: int) -> None:\n"
        '        """Method doc.\n\n'
        "        Args:\n"
        "            x (int): Value.\n"
        '        """\n'
        "        pass\n"
        "    async def async_fn(cls, y: int) -> str:\n"
        '        """Async doc.\n\n'
        "        Args:\n"
        "            y (int): Value.\n\n"
        "        Returns:\n"
        "            str: Output.\n"
        '        """\n'
        "        return 'ok'\n",
        encoding="utf-8",
    )
    violations = check_file(str(bad_py))
    assert any("missing module docstring" in v for v in violations)
    assert any("missing class docstring on 'MissingClassDoc'" in v for v in violations)
    assert any("missing docstring on 'missing_func_doc'" in v for v in violations)
    assert any("prohibited Any/object docstring type in 'bad_doc'" in v for v in violations)
    assert any("missing 'Args:' section in 'bad_doc'" in v for v in violations)
    assert any("missing arg 'pos_only' in 'bad_doc'" in v for v in violations)
    assert any("missing arg 'args' in 'bad_doc'" in v for v in violations)
    assert any("missing arg 'kw_only' in 'bad_doc'" in v for v in violations)
    assert any("missing arg 'kwargs' in 'bad_doc'" in v for v in violations)
    assert any("missing 'Returns:'/'Yields:' section in 'bad_doc'" in v for v in violations)

    assert check_directory(str(bad_py)) == violations
    assert main([str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "missing module docstring" in captured.err

    clean_dir = tmp_path / "clean_pkg"
    clean_dir.mkdir()
    clean_py = clean_dir / "clean.py"
    clean_py.write_text(
        '"""Clean module docstring."""\n\n'
        "def ok(x: int) -> int:\n"
        '    """Compute value.\n\n'
        "    Args:\n"
        "        x (int): Input.\n\n"
        "    Returns:\n"
        "        int: Result.\n"
        '    """\n'
        "    return x\n",
        encoding="utf-8",
    )
    assert main([str(clean_dir)]) == 0
    with patch("sys.argv", ["check_docs.py", str(clean_dir)]):
        assert main(None) == 0


def test_check_docs_advanced_sections_and_main(tmp_path: Path) -> None:
    """Test check_docs.py __main__, multiline Args, stale Args, Yields, Raises, Attributes, and Enums.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    from scripts.check_docs import check_file

    clean_dir = tmp_path / "clean_runpy_pkg"
    clean_dir.mkdir()
    (clean_dir / "ok.py").write_text('"""Clean module."""\n', encoding="utf-8")

    # Test check_docs __main__, multiline/stale Args, Yields, Raises, Attributes, Field, inline docs, Enums
    advanced_py = tmp_path / "advanced_docs.py"
    advanced_py.write_text(
        '"""Module doc."""\n'
        "import enum\n"
        "from enum import Enum\n"
        "import pydantic\n\n"
        "class BadEnum(Enum):\n"
        '    """Enum missing member docs."""\n'
        "    _ignore = 0\n"
        "    A = 1\n\n"
        "class QualEnum(enum.Enum):\n"
        '    """Qualified enum with inline doc."""\n'
        "    A = 1\n"
        '    """Inline member doc."""\n\n'
        "class BadFields(pydantic.BaseModel):\n"
        '    """Class missing field docs."""\n'
        "    _priv: int = 0\n"
        "    undoc: int = pydantic.Field(default=1)\n"
        '    empty_desc: int = pydantic.Field(default=1, description="   ")\n'
        "    computed_desc: int = pydantic.Field(default=1, description=str(1))\n"
        "    inline_ok: int = 2\n"
        '    """Inline field doc."""\n'
        "    trailing_undoc: int\n\n"
        "def gen_missing_yields(x: int):\n"
        '    """Summary mentions x only.\n\n'
        "    Args:\n"
        "         Preamble note without colon\n"
        "        x (int):\n"
        "            Multiline description for x.\n"
        "        Trailing unindented note without colon\n"
        "        stale_param (int): No longer in signature.\n"
        '    """\n'
        "    def _nested() -> None:\n"
        '        """Nested helper."""\n'
        "        raise ValueError('nested')\n"
        "    if x < 0:\n"
        "        raise ValueError('bad x')\n"
        "    yield x\n",
        encoding="utf-8",
    )
    adv_v = check_file(str(advanced_py))
    assert any("undocumented enum member 'A' on Enum 'BadEnum'" in v for v in adv_v)
    assert not any("GoodEnum" in v for v in adv_v)
    assert any("undocumented attribute 'undoc' on class 'BadFields'" in v for v in adv_v)
    assert any("undocumented attribute 'empty_desc' on class 'BadFields'" in v for v in adv_v)
    assert not any("computed_desc" in v for v in adv_v)
    assert not any("inline_ok" in v for v in adv_v)
    assert any("stale documented arg 'stale_param'" in v for v in adv_v)
    assert any("missing 'Yields:' section" in v for v in adv_v)
    assert any("missing 'Raises:' section" in v for v in adv_v)

    orig_cd = sys.modules.pop("scripts.check_docs", None)
    try:
        with patch("sys.argv", ["check_docs.py", str(clean_dir)]), pytest.raises(SystemExit) as cd_exc:
            runpy.run_module("scripts.check_docs", run_name="__main__")
        assert cd_exc.value.code == 0
    finally:
        if orig_cd is not None:
            sys.modules["scripts.check_docs"] = orig_cd


def test_lint_typing_hardened_rules_and_main(tmp_path: Path) -> None:
    """Test new anti-type-erasure, anti-assert, and LSP __eq__ rules in scripts/lint_typing.py.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    from scripts.lint_typing import check_source

    eq_src = "class K:\n    def __eq__(self, other: object) -> bool:\n        return False\n"
    assert check_source(eq_src, Path("src/ml_switcheroo_ir/ok.py")) == []

    erasure_src = (
        "import sys\n"
        "import typing\n"
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    T = int\n"
        "else:\n"
        "    T = str\n"
        "if typing.TYPE_CHECKING:\n"
        "    U = int\n"
        "else:\n"
        "    U = str\n"
        "assert True\n"
        "m = sys.modules['os']\n"
        "class Ann:\n"
        "    def __get_pydantic_json_schema__(self) -> dict[str, int]:\n"
        "        return {}\n"
    )
    v_src = check_source(erasure_src, Path("src/ml_switcheroo_ir/bad_prod.py"))
    assert sum("if TYPE_CHECKING: ... else:" in v for v in v_src) == 2
    assert any("Forbidden bare 'assert'" in v for v in v_src)
    assert any("Forbidden cross-module 'sys.modules'" in v for v in v_src)
    assert any("Forbidden unconstrained empty '{}' return" in v for v in v_src)

    clean_f = tmp_path / "clean_lt.py"
    clean_f.write_text('"""Doc."""\nx: int = 1\n', encoding="utf-8")
    orig_lt = sys.modules.pop("scripts.lint_typing", None)
    try:
        with patch("sys.argv", ["lint_typing.py", str(clean_f)]), pytest.raises(SystemExit) as lt_exc:
            runpy.run_module("scripts.lint_typing", run_name="__main__")
        assert lt_exc.value.code == 0
    finally:
        if orig_lt is not None:
            sys.modules["scripts.lint_typing"] = orig_lt


def test_check_function_coverage_exhaustive(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test scripts/check_function_coverage.py across 100% covered and defective coverage payloads.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    from scripts.check_function_coverage import (
        main as cov_main,
    )
    from scripts.check_function_coverage import (
        verify_coverage_data,
    )

    sample_mod = tmp_path / "mod.py"
    sample_mod.write_text(
        '"""Sample module."""\n'
        "import enum\n"
        "import typing\n"
        "from typing import Protocol, overload\n\n"
        "class NormalCls(enum.Enum):\n"
        '    """Non-protocol class with Name and Attribute bases."""\n'
        "    X = 1\n\n"
        "class SubNormal(NormalCls):\n"
        '    """Non-protocol Name base."""\n\n'
        "class ProtoA(Protocol):\n"
        '    """Protocol A."""\n'
        "    def p1(self) -> int:\n"
        '        """Doc."""\n'
        "        ...\n\n"
        "class ProtoB(typing.Protocol):\n"
        '    """Protocol B."""\n'
        "    def p2(self) -> int:\n"
        '        """Doc."""\n\n'
        "@overload\n"
        "def fn(x: int) -> int: ...\n"
        "@typing.overload\n"
        "def fn(x: str) -> str: ...\n"
        "@typing.final\n"
        "@staticmethod\n"
        "def fn(x: int | str) -> int | str:\n"
        '    """Concrete fn."""\n'
        "    return x\n\n"
        "def uncovered_fn() -> int:\n"
        '    """Uncovered fn."""\n'
        "    return 42\n",
        encoding="utf-8",
    )

    assert verify_coverage_data({}) == ["No measured files found in coverage report."]

    from ml_switcheroo_ir.types import JsonValue

    good_cov: dict[str, JsonValue] = {
        "files": {
            "mod.py": {
                "executed_lines": list(range(1, 45)),
                "missing_lines": "not_a_list",
                "missing_branches": [],
            },
            "deleted.py": {
                "executed_lines": [1],
                "missing_lines": [],
                "missing_branches": [],
            },
            "bad_rec": "ignored",
        },
        "totals": {"percent_covered": 100.0},
    }
    assert verify_coverage_data(good_cov, repo_root=tmp_path) == []
    assert verify_coverage_data({"files": good_cov["files"]}, repo_root=tmp_path) == []

    bad_cov: dict[str, JsonValue] = {
        "files": {
            str(sample_mod): {
                "executed_lines": [32],
                "missing_lines": [36],
                "missing_branches": [[32, 36]],
            }
        },
        "totals": {"percent_covered": 85.0, "bad_list": "nope"},
    }
    errs = verify_coverage_data(bad_cov, repo_root=tmp_path)
    assert any("missing lines [36]" in e for e in errs)
    assert any("missing branches" in e for e in errs)
    assert any("uncovered function 'uncovered_fn'" in e for e in errs)
    assert any("85.00% is below required 100.00%" in e for e in errs)

    cov_json_file = tmp_path / "cov.json"
    cov_json_file.write_text(json.dumps(good_cov), encoding="utf-8")
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        assert cov_main(["--coverage-json", str(cov_json_file)]) == 0

    bad_cov_file = tmp_path / "bad_cov.json"
    bad_cov_file.write_text(json.dumps(bad_cov), encoding="utf-8")
    assert cov_main(["--coverage-json", str(bad_cov_file)]) == 1
    assert "[COVERAGE FAILURE]" in capsys.readouterr().err

    list_cov_file = tmp_path / "list_cov.json"
    list_cov_file.write_text("[1, 2]", encoding="utf-8")
    assert cov_main(["--coverage-json", str(list_cov_file)]) == 1


def test_check_function_coverage_subprocess_and_main(tmp_path: Path) -> None:
    """Test check_function_coverage.py subprocess coverage JSON export and __main__ entrypoint.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    from ml_switcheroo_ir.types import JsonValue
    from scripts.check_function_coverage import (
        _load_or_generate_coverage_json,
    )
    from scripts.check_function_coverage import (
        main as cov_main,
    )

    good_cov: dict[str, JsonValue] = {
        "files": {"deleted.py": {"executed_lines": [1], "missing_lines": [], "missing_branches": []}},
        "totals": {"percent_covered": 100.0},
    }
    cov_json_file = tmp_path / "cov_sub.json"
    cov_json_file.write_text(json.dumps(good_cov), encoding="utf-8")

    # Test subprocess coverage json export path (success, non-dict, and failure)
    def _fake_cov_ok(cmd: list[str], **kwargs: bool | str | int | None) -> MagicMock:
        """Write valid coverage JSON on subprocess run.

        Args:
            cmd (list[str]): Command arguments.
            **kwargs (bool | str | int | None): Subprocess options.

        Returns:
            MagicMock: Completed process mock.
        """
        del kwargs
        out_p = cmd[cmd.index("-o") + 1]
        Path(out_p).write_text(json.dumps(good_cov), encoding="utf-8")
        m = MagicMock()
        m.returncode = 0
        return m

    with patch("subprocess.run", side_effect=_fake_cov_ok):
        loaded = _load_or_generate_coverage_json(None)
        assert "files" in loaded

    def _fake_cov_list(cmd: list[str], **kwargs: bool | str | int | None) -> MagicMock:
        """Write invalid list JSON on subprocess run.

        Args:
            cmd (list[str]): Command arguments.
            **kwargs (bool | str | int | None): Subprocess options.

        Returns:
            MagicMock: Completed process mock.
        """
        del kwargs
        out_p = cmd[cmd.index("-o") + 1]
        Path(out_p).write_text("[]", encoding="utf-8")
        m = MagicMock()
        m.returncode = 0
        return m

    with patch("subprocess.run", side_effect=_fake_cov_list), pytest.raises(RuntimeError, match="must be an object"):
        _load_or_generate_coverage_json(None)

    fail_proc = MagicMock()
    fail_proc.returncode = 1
    fail_proc.stderr = "no data"
    with patch("subprocess.run", return_value=fail_proc):
        assert cov_main([]) == 1

    orig_cfc = sys.modules.pop("scripts.check_function_coverage", None)
    try:
        with (
            patch("pathlib.Path.cwd", return_value=tmp_path),
            patch("sys.argv", ["check_function_coverage.py", "--coverage-json", str(cov_json_file)]),
            pytest.raises(SystemExit) as cfc_exc,
        ):
            runpy.run_module("scripts.check_function_coverage", run_name="__main__")
        assert cfc_exc.value.code == 0
    finally:
        if orig_cfc is not None:
            sys.modules["scripts.check_function_coverage"] = orig_cfc


def test_snapshots_and_verify_grounding_ast_edge_branches(tmp_path: Path) -> None:
    """Test require_framework_snapshots_dir, bundled catalogs, and verify_grounding AST edge branches.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    import ml_switcheroo_ir.snapshots as snap_mod
    from ml_switcheroo_ir.snapshots import (
        _resolve_fallback_snapshot_dir,
        find_schema_file,
        get_effective_snapshot_dir,
        require_framework_snapshots_dir,
        resolve_snapshots_path,
    )
    from scripts.verify_grounding import (
        _class_defines_member_ast,
        _resolve_ast_symbol_in_module,
        _resolve_ir_symbol,
        find_snapshots_directory,
    )
    from scripts.verify_grounding import (
        main as vg_main,
    )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        require_framework_snapshots_dir(tmp_path / "missing_fw_snaps")

    empty_fw = tmp_path / "empty_fw_snaps"
    empty_fw.mkdir()
    with pytest.raises(FileNotFoundError, match="contains no .json manifests"):
        require_framework_snapshots_dir(empty_fw)

    pop_fw = tmp_path / "pop_fw_snaps"
    pop_fw.mkdir()
    (pop_fw / "ok.json").write_text("{}", encoding="utf-8")
    assert require_framework_snapshots_dir(pop_fw) == pop_fw.resolve()

    # Fallback snapshot dir when ml-framework-snapshots exists on disk
    fake_parent = tmp_path / "repos_root"
    (fake_parent / "ml-framework-snapshots").mkdir(parents=True)
    assert "ml-framework-snapshots" in _resolve_fallback_snapshot_dir(fake_parent)

    # DEFAULT_SNAPSHOT_DIR override on snapshots module and verify_grounding
    with patch.object(snap_mod, "DEFAULT_SNAPSHOT_DIR", str(pop_fw)):
        assert get_effective_snapshot_dir() == str(pop_fw)

    with patch("scripts.verify_grounding.DEFAULT_SNAPSHOT_DIR", str(pop_fw)), patch.dict(os.environ, {}, clear=True):
        assert find_snapshots_directory(None) == pop_fw.resolve()

    # Multi-candidate sequence in find_snapshots_directory: missing + empty nested fallback
    shell_repo = tmp_path / "shell_repo"
    (shell_repo / "snapshots").mkdir(parents=True)
    with patch("ml_switcheroo_ir.snapshots.get_effective_snapshot_dir", return_value=str(empty_fw)):
        assert resolve_snapshots_path(shell_repo, allow_default_fallback=True) == (shell_repo / "snapshots").resolve()
        assert (
            find_snapshots_directory([str(tmp_path / "nope"), str(shell_repo)]) == (shell_repo / "snapshots").resolve()
        )
        assert find_snapshots_directory([str(tmp_path / "nope1"), str(tmp_path / "nope2")]) is None

    # Bundled catalogs lookup in find_schema_file
    fake_pkg = tmp_path / "fake_sw_pkg"
    cat_dir = fake_pkg / "schema" / "catalogs"
    cat_dir.mkdir(parents=True)
    bundled_f = cat_dir / "bundled_ops.json"
    bundled_f.write_text("{}", encoding="utf-8")
    with (
        patch.object(snap_mod, "DEFAULT_SNAPSHOT_DIR", str(empty_fw)),
        patch.object(snap_mod, "__file__", str(fake_pkg / "snapshots.py")),
    ):
        assert find_schema_file("bundled_ops.json") == bundled_f

    # verify_grounding --framework-snapshots-dir CLI flag
    assert vg_main(["--strict", "--framework-snapshots-dir", str(tmp_path / "nonexistent_fw")]) == 1

    # verify_grounding AST symbol resolution edge cases (cycle, Pydantic base method, base traversal, non-class parent)
    assert (
        _resolve_ast_symbol_in_module("ml_switcheroo_ir", "LogicalNode", visited={("ml_switcheroo_ir", "LogicalNode")})
        is None
    )
    assert _resolve_ir_symbol("ml_switcheroo_ir.LogicalNode", "model_dump", allow_nested=True) is True
    assert _resolve_ir_symbol("ml_switcheroo_ir.ZeroTangent", "clone", allow_nested=True) is True
    assert _resolve_ir_symbol("ml_switcheroo_ir.ZeroTangent", "nonexistent_member", allow_nested=True) is False
    assert _resolve_ir_symbol("ml_switcheroo_ir.topological_sort", "not_a_class_member", allow_nested=True) is False

    import ast as _ast

    sub_cls = _ast.parse("class Sub(tuple[int, ...], UnknownBase):\n    pass\n").body[0]
    assert isinstance(sub_cls, _ast.ClassDef)
    assert _class_defines_member_ast("ml_switcheroo_ir.graph", sub_cls, "missing_m") is False


def test_registry_generators_and_verify_grounding_remaining_branches(tmp_path: Path) -> None:
    """Exercise remaining branches in generate_registry.py, generate_stablehlo_registry.py, and verify_grounding.py.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    from scripts.generate_registry import (
        _parse_proto_default_str,
        extract_grounded_onnx_catalog,
        extract_onnx_from_defs,
        extract_onnx_from_snapshot_json,
        map_onnx_attribute_type,
    )
    from scripts.generate_registry import main as gen_onnx_main

    # 1. generate_registry.py branches
    assert map_onnx_attribute_type("AttributeProto.INT") == "int"
    assert map_onnx_attribute_type("CompletelyUnknownType") == "Any"
    assert _parse_proto_default_str("[1, 2]") == [1, 2]
    assert _parse_proto_default_str("(1, 2)") == "(1, 2)"
    assert _parse_proto_default_str("{1, 2}") == "{1, 2}"
    assert _parse_proto_default_str("unparseable(") == "unparseable("
    assert extract_onnx_from_snapshot_json(tmp_path / "missing.json") == {}
    non_map_snap = tmp_path / "onnx_list.json"
    non_map_snap.write_text("[1, 2]", encoding="utf-8")
    assert extract_onnx_from_snapshot_json(non_map_snap) == {}
    ops_only_snap = tmp_path / "onnx_ops_only.json"
    ops_only_snap.write_text(json.dumps({"operations": [{"name": "OnlyOp"}]}), encoding="utf-8")
    assert "OnlyOp" in extract_onnx_from_snapshot_json(ops_only_snap)

    s_new = MagicMock(domain="", deprecated=False, since_version=14, attributes={}, inputs=[], outputs=[])
    s_new.name = "Add"
    s_old = MagicMock(domain="", deprecated=False, since_version=1, attributes={}, inputs=[], outputs=[])
    s_old.name = "Add"
    fake_defs = MagicMock(get_all_schemas_with_history=lambda: [s_new, s_old], get_all_schemas=lambda: [s_old])
    fake_onnx = MagicMock(defs=fake_defs)
    with patch.dict("sys.modules", {"onnx": fake_onnx, "onnx.defs": fake_defs}):
        assert extract_onnx_from_defs()["Add"]["version"] == 14

    snap_dir = tmp_path / "onnx_snaps"
    snap_dir.mkdir()
    (snap_dir / "onnx_ops.json").write_text("{}", encoding="utf-8")
    (snap_dir / "onnx_custom.json").write_text(
        json.dumps(
            {
                "categories": {
                    "nn": "not_list",
                    "math": [
                        "skip_str",
                        {
                            "name": "SnapExtraOp",
                            "domain_metadata": {"domain": "", "upstream_version": "bad"},
                            "operands": [{"name": "X"}],
                            "returns": [{"name": "Y"}],
                            "attributes": {"alpha": {"type": "float", "default": "f: 1.5"}, "bad": 5},
                        },
                    ],
                },
                "operations": ["skip_op_str", {"name": ""}, {"name": "SnapExtraOp"}, {"name": "SnapOp2"}],
            }
        ),
        encoding="utf-8",
    )
    cat = extract_grounded_onnx_catalog(snap_dir)
    assert "SnapExtraOp" in cat and "SnapOp2" in cat
    with patch("scripts.generate_registry.get_default_snapshots_dir", return_value=""):
        assert "Add" in extract_grounded_onnx_catalog(None)
    assert "Add" in extract_grounded_onnx_catalog(tmp_path / "not_a_dir")
    with patch.dict("sys.modules", {"onnx": None, "onnx.defs": None}):
        assert extract_onnx_from_defs() == {}
    with (
        patch("sys.argv", ["generate_registry.py"]),
        patch("scripts.generate_registry.extract_grounded_onnx_catalog", return_value={}),
        patch("os.path.exists", side_effect=lambda p: p == "docs/Operators.md"),
        patch("scripts.generate_registry.parse_onnx_docs", return_value={"Add": cat["SnapOp2"]}),
    ):
        gen_onnx_main(json_path=str(tmp_path / "out_onnx.json"), registry_path=str(tmp_path / "out_reg.py"))
    with (
        patch("sys.argv", ["generate_registry.py"]),
        patch("scripts.generate_registry.extract_grounded_onnx_catalog", return_value={}),
        patch("os.path.exists", return_value=False),
    ):
        gen_onnx_main(json_path=str(tmp_path / "out_onnx2.json"), registry_path=str(tmp_path / "out_reg2.py"))


def test_generate_stablehlo_registry_remaining_branches(tmp_path: Path) -> None:
    """Exercise remaining branches in scripts/generate_stablehlo_registry.py.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    from scripts.generate_stablehlo_registry import (
        _default_stablehlo_snapshot_path,
        _extract_stablehlo_attributes,
        extract_stablehlo_schemas,
    )

    conv_attrs = _extract_stablehlo_attributes(
        "convolution", [{"name": "dimension_numbers", "type": "str", "required": False}]
    )
    assert conv_attrs[0]["required"] is True and conv_attrs[0]["type"] == "dict"
    ret_only_file = tmp_path / "shlo_ret_scalar.json"
    ret_only_file.write_text(
        json.dumps([{"name": "add", "operands": [{"name": "a"}], "returns": "tensor"}]),
        encoding="utf-8",
    )
    assert extract_stablehlo_schemas(ret_only_file)[0]["outputs"] == ["result"]
    exh_dir = tmp_path / "shlo_exh_dir"
    exh_dir.mkdir()
    (exh_dir / "stablehlo_exhaustive.json").write_text("{}", encoding="utf-8")
    with patch("scripts.generate_stablehlo_registry.get_effective_snapshot_dir", return_value=str(exh_dir)):
        assert _default_stablehlo_snapshot_path().endswith("stablehlo_exhaustive.json")
    (exh_dir / "stablehlo_exhaustive.json").unlink()
    with patch("scripts.generate_stablehlo_registry.get_effective_snapshot_dir", return_value=str(exh_dir)):
        assert _default_stablehlo_snapshot_path().endswith("stablehlo_v1.9.0.json")


def test_verify_onnx_grounding_snapshot_mismatches(tmp_path: Path) -> None:
    """Verify ONNX snapshot cross-verification across dict and operations manifests.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    from ml_switcheroo_ir.schema import ONNX_REGISTRY, OpAttribute, OpSchema, override_registry
    from scripts.verify_grounding import (
        _compare_onnx_op_against_spec,
        _extract_onnx_attr_map,
        _load_onnx_ground_truth,
        _parse_onnx_snapshot_entry,
        _supplement_onnx_defs_specs,
        _verify_registry_structural_integrity,
        verify_onnx_grounding,
    )

    mismatched_schema = OpSchema("WrongKey", "wrong_dom", 1, {}, [], [])
    s_errs = _verify_registry_structural_integrity({"KeyA": mismatched_schema}, "TEST_REG", ("ai.onnx",), "Test")
    assert len(s_errs) == 2

    onnx_dir = tmp_path / "vg_onnx"
    onnx_dir.mkdir()
    assert verify_onnx_grounding(None) == []
    assert any("Missing required ONNX" in e for e in verify_onnx_grounding(onnx_dir, strict=True))
    (onnx_dir / "onnx_list.json").write_text("[1]", encoding="utf-8")
    assert any("is not grounded in onnx_list.json snapshot" in e for e in verify_onnx_grounding(onnx_dir, strict=True))
    (onnx_dir / "onnx_list.json").unlink()
    with patch.dict("sys.modules", {"onnx": None, "onnx.defs": None}):
        (onnx_dir / "onnx_dict.json").write_text(
            json.dumps(
                {
                    "SkipStr": "bad",
                    "MissingFromReg": {
                        "domain": "ai.onnx",
                        "version": 1,
                        "inputs": [],
                        "outputs": [],
                        "attributes": [],
                    },
                    "OpA": {
                        "domain": "ai.onnx",
                        "version": 14,
                        "inputs": ["X"],
                        "outputs": ["Y"],
                        "attributes": [
                            {"name": "req_a", "type": "int", "required": True},
                            {"name": "bad_t", "type": "BadType"},
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )
        bad_op_a = OpSchema(
            "OpA",
            "ai.onnx",
            10,
            {"bad_t": OpAttribute("bad_t", "float", False), "extra_a": OpAttribute("extra_a", "int", False)},
            ["W"],
            ["Z"],
        )
        with override_registry(ONNX_REGISTRY, {"OpA": bad_op_a}, clear=True):
            o_mismatch = verify_onnx_grounding(onnx_dir)
            assert any("MissingFromReg" in e for e in o_mismatch)
            assert any("version" in e for e in o_mismatch)
            assert any("inputs" in e for e in o_mismatch)
            assert any("outputs" in e for e in o_mismatch)
            assert any("missing grounded attribute 'req_a'" in e for e in o_mismatch)
            assert any("invalid snapshot type" in e for e in o_mismatch)
            assert any("ungrounded attribute 'extra_a'" in e for e in o_mismatch)

    assert _extract_onnx_attr_map([{"no_name": 1}]) == {}
    assert (
        _parse_onnx_snapshot_entry({"name": "OpD", "domain_metadata": {"domain": "", "upstream_version": "bad"}})
        is not None
    )
    empty_name_onnx = onnx_dir / "onnx_empty_name_dict.json"
    empty_name_onnx.write_text('{"EmptyNameOp": {"name": ""}}', encoding="utf-8")
    assert _load_onnx_ground_truth(empty_name_onnx) == ({}, False)
    empty_name_onnx.unlink()

    ops_onnx_file = onnx_dir / "onnx_ops_manifest.json"
    ops_onnx_file.write_text(
        json.dumps(
            {
                "operations": [
                    {"name": ""},
                    {
                        "name": "OpA",
                        "domain_metadata": {"domain": "ai.onnx.ml", "upstream_version": 3},
                        "operands": [{"name": "X"}],
                        "attributes": {"a1": {"type": "int", "required": True}, "skip_bad": 5},
                    },
                    {"name": "OpA", "domain": "dup_ignored"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (onnx_dir / "onnx_dict.json").unlink()
    with override_registry(
        ONNX_REGISTRY,
        {"OpA": OpSchema("OpA", "ai.onnx", 3, {"a1": OpAttribute("a1", "float", False)}, [], [])},
        clear=True,
    ):
        m2 = verify_onnx_grounding(onnx_dir)
        assert any("domain" in e for e in m2)
        assert any("required=" in e for e in m2)
        assert any("type 'float' != " in e for e in m2)

    assert (
        _compare_onnx_op_against_spec(
            "OpB",
            OpSchema("OpB", "ai.onnx", 1, {"a": OpAttribute("a", "int", False)}, [], []),
            {
                "domain": "ai.onnx",
                "version": 1,
                "exact_version": False,
                "inputs": [],
                "outputs": [],
                "attributes": "bad",
            },
            "src",
        )
        == []
    )
    assert (
        _compare_onnx_op_against_spec(
            "OpC",
            OpSchema("OpC", "ai.onnx", 1, {"k": OpAttribute("k", "int", False)}, [], []),
            {
                "domain": "ai.onnx",
                "version": 1,
                "exact_version": False,
                "inputs": [],
                "outputs": [],
                "attributes": {"k": "not_map"},
            },
            "src",
        )
        == []
    )
    with patch.object(sys, "path", []):
        with patch("scripts.generate_registry.extract_onnx_from_defs", return_value={"Already": {"version": 1}}):
            _ = _supplement_onnx_defs_specs({"Already": {"name": "Already"}}, is_partial=True)


def test_verify_concept_map_and_translation_grounding_branches(tmp_path: Path) -> None:
    """Verify Quantization, State, ODL, and ParameterTranslationEngine snapshot checks.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    from ml_switcheroo_ir.translation import ParameterTranslationEngine
    from scripts.verify_grounding import (
        verify_chex_grounding,
        verify_odl_grounding,
        verify_quantization_grounding,
        verify_state_ops_grounding,
        verify_translation_grounding,
    )

    direct_dir = tmp_path / "vg_direct"
    direct_dir.mkdir()
    assert verify_quantization_grounding(None) == []
    assert verify_state_ops_grounding(None) == []
    assert verify_odl_grounding(None) == []
    assert verify_translation_grounding(None) == []
    assert any("Missing required Quantization" in e for e in verify_quantization_grounding(direct_dir, strict=True))
    assert any("Missing required State" in e for e in verify_state_ops_grounding(direct_dir, strict=True))
    assert any("Missing required ODL" in e for e in verify_odl_grounding(direct_dir, strict=True))
    assert any(
        "Missing required ParameterTranslationEngine" in e
        for e in verify_translation_grounding(direct_dir, strict=True)
    )

    (direct_dir / "concept_map.json").write_text("{}", encoding="utf-8")
    assert any("not grounded in concept_map.json" in e for e in verify_quantization_grounding(direct_dir))
    assert any("not grounded in concept_map.json" in e for e in verify_state_ops_grounding(direct_dir))
    assert any("not grounded in concept_map.json" in e for e in verify_odl_grounding(direct_dir))
    assert any("missing '_parameter_translations'" in e for e in verify_translation_grounding(direct_dir, strict=True))

    (direct_dir / "concept_map.json").write_text("[1, 2]", encoding="utf-8")
    assert verify_quantization_grounding(direct_dir) == []
    assert verify_state_ops_grounding(direct_dir) == []
    assert any("root must be a JSON object" in e for e in verify_translation_grounding(direct_dir))

    canon_eng = ParameterTranslationEngine()
    ops_iter = iter(canon_eng.translations.operations)
    canon_op = next(ops_iter)
    canon_op_2 = next(ops_iter)
    canon_role = next(iter(canon_eng.translations[canon_op].roles))
    canon_role_2 = next(iter(canon_eng.translations[canon_op_2].roles))
    (direct_dir / "concept_map.json").write_text(
        json.dumps(
            {
                "_parameter_translations": {
                    "missing_op_in_engine": {"roles": {}},
                    canon_op: {
                        "roles": {
                            "missing_role": {},
                            canon_role: {"torch": ["missing_torch_param"], "bad_fw": "not_list"},
                        }
                    },
                    canon_op_2: {"roles": {canon_role_2: "not_a_map"}},
                }
            }
        ),
        encoding="utf-8",
    )
    t_errs = verify_translation_grounding(direct_dir, strict=True)
    assert any("missing_op_in_engine" in e for e in t_errs)
    assert any("missing_role" in e for e in t_errs)
    assert any("missing_torch_param" in e for e in t_errs)
    (direct_dir / "concept_map.json").write_text(
        json.dumps({"_parameter_translations": {canon_op: "not_a_mapping"}}),
        encoding="utf-8",
    )
    assert verify_translation_grounding(direct_dir, strict=True) == []
    (direct_dir / "concept_map.json").write_text('{"_parameter_translations": "not_map"}', encoding="utf-8")
    assert verify_translation_grounding(direct_dir, strict=False) == []

    empty_eng = MagicMock()
    empty_eng.translations = {}
    with patch("scripts.verify_grounding.ParameterTranslationEngine", return_value=empty_eng):
        assert verify_translation_grounding(None) == ["ParameterTranslationEngine translations table is empty."]

    # Direct snapshot files for Quantization, State, ODL, and Chex
    (direct_dir / "quantization_snap.json").write_text(
        '{"ops": [{"name": "quantization.block_quantize"}]}', encoding="utf-8"
    )
    (direct_dir / "state_snap.json").write_text('{"ops": [{"name": "ReadVariable"}]}', encoding="utf-8")
    (direct_dir / "odl_snap.json").write_text('{"ops": []}', encoding="utf-8")
    assert any("not grounded in quantization_snap.json" in e for e in verify_quantization_grounding(direct_dir))
    assert any("not grounded in state_snap.json" in e for e in verify_state_ops_grounding(direct_dir))
    assert any("not grounded in odl_snap.json" in e for e in verify_odl_grounding(direct_dir))
    (direct_dir / "chex_empty.json").write_text("{}", encoding="utf-8")
    assert verify_chex_grounding(direct_dir) == []


def test_verify_isa_and_helper_grounding_edge_branches(tmp_path: Path) -> None:
    """Verify AST symbol helper, attention helper, and RDNA/SASS/WGSL/PTX edge branches.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    from ml_switcheroo_ir.schema import (
        OpSchema,
    )
    from scripts.verify_grounding import (
        _check_single_attention_op,
        _collect_manifest_records,
        _find_prefixed_symbol,
        _resolve_ir_symbol,
        _validate_wgsl_ops_payload,
        find_snapshots_directory,
        verify_ptx_grounding,
        verify_rdna_grounding,
        verify_sass_grounding,
        verify_stablehlo_grounding,
        verify_wgsl_grounding,
    )

    with (
        patch("scripts.verify_grounding.DEFAULT_SNAPSHOT_DIR", ""),
        patch("ml_switcheroo_ir.snapshots.get_default_snapshots_dir", return_value=""),
    ):
        assert find_snapshots_directory() is None
    assert _collect_manifest_records([{"name": "a"}, "skip"]) == [{"name": "a"}]
    assert _collect_manifest_records(123) == []
    assert _resolve_ir_symbol("ml_switcheroo_ir.validator", "_extract_dst_vgpr", False)
    assert not _resolve_ir_symbol("ml_switcheroo_ir.validator", "TotallyMissingSymbol", False)
    assert _validate_wgsl_ops_payload({"ops": ["not_a_dict"]}) == []
    assert _find_prefixed_symbol("add", ("nvidia_ptx", "ptx"), {"ptx.add": {"name": "add"}}) == {"name": "add"}

    shlo_empty_sym = tmp_path / "shlo_empty_sym"
    shlo_empty_sym.mkdir()
    (shlo_empty_sym / "stablehlo_e.json").write_text('{"add": {}}', encoding="utf-8")
    with patch("scripts.verify_grounding.STABLEHLO_REGISTRY", {"add": OpSchema("add", "stablehlo")}):
        assert verify_stablehlo_grounding(shlo_empty_sym) == []

    pa_schema = OpSchema("PagedAttention", "ml.switcheroo.custom", 1, {}, ["bad_inp"], ["out"])
    pa_errs = _check_single_attention_op(
        "PagedAttention",
        pa_schema,
        {"paged_attention_v1": {"params": [{"name": "query"}]}},
        "flash.json",
    )
    assert any("bad_inp" in e for e in pa_errs)
    assert _check_single_attention_op("PagedAttention", pa_schema, {"paged_attention_v1": {}}, "flash.json") == []

    non_dict_isa = tmp_path / "vg_isa_edge"
    non_dict_isa.mkdir()
    (non_dict_isa / "amd_rdna_list.json").write_text("[1]", encoding="utf-8")
    (non_dict_isa / "nvidia_sass_list.json").write_text("[1]", encoding="utf-8")
    (non_dict_isa / "wgsl_list.json").write_text("[1]", encoding="utf-8")
    (non_dict_isa / "nvidia_ptx_list.json").write_text("[1]", encoding="utf-8")
    assert verify_rdna_grounding(non_dict_isa) == []
    assert verify_sass_grounding(non_dict_isa) == []
    assert verify_wgsl_grounding(non_dict_isa) == []
    assert verify_ptx_grounding(non_dict_isa) == []

    (non_dict_isa / "wgsl_list.json").write_text('{"ops": "not_list"}', encoding="utf-8")
    assert verify_wgsl_grounding(non_dict_isa) == []
    (non_dict_isa / "nvidia_ptx_list.json").write_text('{"categories": "not_dict"}', encoding="utf-8")
    assert verify_ptx_grounding(non_dict_isa) == []
    (non_dict_isa / "amd_rdna_list.json").write_text(
        json.dumps({"source_type": "tablegen", "instructions": [{"name": "V_DOT4C_I32_I8"}]}),
        encoding="utf-8",
    )
    _ = verify_rdna_grounding(non_dict_isa)

    assert verify_rdna_grounding(None) == []
    assert verify_sass_grounding(None) == []
    assert verify_wgsl_grounding(None) == []
    assert verify_ptx_grounding(None) == []


def test_verify_isa_structural_tables_and_ptx_attrs(tmp_path: Path) -> None:
    """Verify RDNA VOPD, SASS latency, WGSL primitive, and PTX attribute mismatch checks.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    from ml_switcheroo_ir.schema import (
        PTX_REGISTRY,
        RDNA_TO_VOPD_MAP,
        RDNA_VOPD_SLOTS,
        SASS_PIPELINE_LATENCIES,
        SASS_REGISTRY,
        WGSL_REGISTRY,
        OpAttribute,
        OpSchema,
        override_registry,
    )
    from scripts.verify_grounding import (
        verify_ptx_grounding,
        verify_rdna_grounding,
        verify_sass_grounding,
        verify_wgsl_grounding,
    )

    saved_vslots = RDNA_VOPD_SLOTS.copy()
    saved_vmap = RDNA_TO_VOPD_MAP.copy()
    saved_slat = SASS_PIPELINE_LATENCIES.copy()
    try:
        RDNA_VOPD_SLOTS["MISSING_VOPD"] = "BAD_SLOT"
        RDNA_TO_VOPD_MAP["MISSING_SRC"] = "MISSING_DST"
        r_tbl_errs = verify_rdna_grounding(None)
        assert any("MISSING_VOPD" in e for e in r_tbl_errs)
        assert any("BAD_SLOT" in e for e in r_tbl_errs)
        assert any("MISSING_SRC" in e for e in r_tbl_errs)
        assert any("MISSING_DST" in e for e in r_tbl_errs)

        SASS_PIPELINE_LATENCIES["MISSING_SASS_LAT"] = 0
        s_tbl_errs = verify_sass_grounding(None)
        assert any("MISSING_SASS_LAT" in e for e in s_tbl_errs)
    finally:
        RDNA_VOPD_SLOTS.clear()
        RDNA_VOPD_SLOTS.update(saved_vslots)
        RDNA_TO_VOPD_MAP.clear()
        RDNA_TO_VOPD_MAP.update(saved_vmap)
        SASS_PIPELINE_LATENCIES.clear()
        SASS_PIPELINE_LATENCIES.update(saved_slat)

    with override_registry(
        WGSL_REGISTRY, {"atomicAdd": OpSchema("atomicAdd", "wgsl", 1, {}, ["wrong"], [])}, clear=True
    ):
        w_sig_errs = verify_wgsl_grounding(None)
        assert any("signature mismatch" in e for e in w_sig_errs)
        assert any("missing from WGSL_REGISTRY" in e for e in w_sig_errs)

    sass_nolat_dir = tmp_path / "vg_sass_nolat"
    sass_nolat_dir.mkdir()
    (sass_nolat_dir / "nvidia_sass_nolat.json").write_text(
        json.dumps({"instructions": [{"name": "FFMA", "execution_latency": 4}, {"name": "FADD"}]}),
        encoding="utf-8",
    )
    with override_registry(
        SASS_REGISTRY,
        {
            "FFMA": OpSchema("FFMA", "nvidia_sass", 1, {}, ["a"], ["d"]),
            "FADD": OpSchema("FADD", "nvidia_sass", 1, {}, ["a"], ["d"]),
        },
        clear=True,
    ):
        saved_lat2 = SASS_PIPELINE_LATENCIES.copy()
        try:
            SASS_PIPELINE_LATENCIES.clear()
            SASS_PIPELINE_LATENCIES["FFMA"] = 4
            assert verify_sass_grounding(sass_nolat_dir) == []
        finally:
            SASS_PIPELINE_LATENCIES.clear()
            SASS_PIPELINE_LATENCIES.update(saved_lat2)

    ptx_attr_dir = tmp_path / "vg_ptx_attr"
    ptx_attr_dir.mkdir()
    (ptx_attr_dir / "nvidia_ptx_attr.json").write_text(
        json.dumps({"categories": {"instructions": [{"name": "fma", "params": [{"name": "known_p"}]}]}}),
        encoding="utf-8",
    )
    with override_registry(
        PTX_REGISTRY,
        {
            "fma": OpSchema(
                "fma", "nvidia_ptx", 1, {"ungrounded_attr": OpAttribute("ungrounded_attr", "str")}, ["a"], ["d"]
            )
        },
        clear=True,
    ):
        assert any("ungrounded_attr" in e for e in verify_ptx_grounding(ptx_attr_dir))
