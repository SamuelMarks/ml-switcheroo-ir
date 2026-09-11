"""Unit tests for repository maintenance and code generation scripts."""

from __future__ import annotations

import json
import os
import runpy
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.generate_registry import main as gen_main
from scripts.generate_registry import parse_onnx_docs
from scripts.update_badges import (
    format_cov,
    get_color,
    get_doc_coverage,
    get_test_coverage,
    update_readme,
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
    """Test get_doc_coverage returns 100.0."""
    assert get_doc_coverage() == 100.0


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
            json.dump({"totals": {"percent_covered": 100.0}}, f)

        with patch("subprocess.run"):
            update_readme(readme_path=readme_file, coverage_json_path=cov_file)

        with open(readme_file, "r", encoding="utf-8") as f:
            updated = f.read()

        assert "test_coverage-100%25-brightgreen.svg" in updated
        assert "doc_coverage-100%25-brightgreen.svg" in updated


def test_update_badges_main() -> None:
    """Test executing scripts.update_badges as __main__ module."""
    with TemporaryDirectory() as tmpdir:
        readme_file = os.path.join(tmpdir, "README.md")
        cov_file = os.path.join(tmpdir, "coverage.json")
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write(
                "# Test\n[![Test Coverage](https://img.shields.io/badge/test_coverage-50%25-red.svg)](#)\n"
            )
        with open(cov_file, "w", encoding="utf-8") as f:
            json.dump({"totals": {"percent_covered": 100.0}}, f)
        with patch("sys.argv", ["update_badges.py", readme_file]), patch(
            "subprocess.run"
        ):
            runpy.run_module("scripts.update_badges", run_name="__main__")


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
