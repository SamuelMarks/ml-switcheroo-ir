"""Unit tests for repository maintenance and code generation scripts."""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from scripts.generate_registry import main as gen_main
from scripts.generate_registry import parse_onnx_docs
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
    find_snapshots_directory,
    verify_custom_ops_grounding,
    verify_ir_snapshot_grounding,
    verify_mlir_grounding,
    verify_onnx_grounding,
    verify_stablehlo_grounding,
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
        with patch("sys.argv", ["update_badges.py", readme_file]), patch(
            "subprocess.run"
        ):
            runpy.run_module("scripts.update_badges", run_name="__main__")

        # Failure case (exit_code != 0) invokes sys.exit
        with patch(
            "sys.argv", ["update_badges.py", "--enforce", "/nonexistent.md"]
        ), patch("sys.exit") as mock_exit:
            runpy.run_module("scripts.update_badges", run_name="__main__")
            mock_exit.assert_called_once_with(1)


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
            assert '"Abs": OpSchema(' in loaded_py


def test_generate_registry_module_main() -> None:
    """Test executing scripts.generate_registry as __main__ module."""
    with TemporaryDirectory() as tmpdir:
        tmp_md = os.path.join(tmpdir, "Operators.md")
        tmp_json = os.path.join(tmpdir, "onnx_ops.json")
        tmp_py = os.path.join(tmpdir, "onnx_registry.py")
        with open(tmp_md, "w", encoding="utf-8") as f:
            f.write(SAMPLE_ONNX_DOCS)
        with patch("sys.argv", ["generate_registry.py", tmp_md, tmp_json, tmp_py]):
            runpy.run_module("scripts.generate_registry", run_name="__main__")
        assert os.path.exists(tmp_json)
        assert os.path.exists(tmp_py)


def test_verify_grounding_find_snapshots_directory() -> None:
    """Test find_snapshots_directory resolution and fallbacks."""
    with TemporaryDirectory() as tmpdir:
        # Override path that is a dir
        assert find_snapshots_directory(tmpdir) is not None

        # Override path that is not a dir
        non_existent = os.path.join(tmpdir, "does_not_exist")
        assert find_snapshots_directory(non_existent) is None

        # Without override, should find existing DEFAULT_SNAPSHOT_DIR or fallback
        with patch("scripts.verify_grounding.DEFAULT_SNAPSHOT_DIR", tmpdir):
            assert find_snapshots_directory() == Path(tmpdir).resolve()

    # Test fallback to script_relative when default_path does not exist
    with patch(
        "scripts.verify_grounding.DEFAULT_SNAPSHOT_DIR", "/nonexistent/dir"
    ), patch("pathlib.Path.is_dir", side_effect=[False, True]):
        assert find_snapshots_directory() is not None

    # Test when default path does not exist and script relative does not exist
    with patch(
        "scripts.verify_grounding.DEFAULT_SNAPSHOT_DIR", "/nonexistent/dir"
    ), patch("pathlib.Path.is_dir", return_value=False):
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
                        "attributes": [{"name": "unknown_attr"}, "not_dict", {}],
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
    with patch("sys.argv", ["verify_grounding.py"]), patch("sys.exit") as mock_exit:
        runpy.run_module("scripts.verify_grounding", run_name="__main__")
        mock_exit.assert_called_once_with(0)


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
                        "util": [{"api_path": "arith.addf"}, "not_a_dict", {}],
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
