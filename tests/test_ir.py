"""Tests for ml_switcheroo_ir package."""

import json
import pytest
import runpy
from tempfile import NamedTemporaryFile
from unittest.mock import patch

from ml_switcheroo_ir import (
    LogicalGraph,
    LogicalNode,
    LogicalEdge,
    LogicalMesh,
    LogicalAxis,
    PartitionSpec,
    topological_sort,
    CompilerBackend,
    GraphFrontend,
)
from ml_switcheroo_ir.cli import main as cli_main, _parse_graph_from_json


def test_topological_sort_linear():
    """Test standard linear topological sort."""
    n1 = LogicalNode("n1", "Input")
    n2 = LogicalNode("n2", "Linear")
    n3 = LogicalNode("n3", "Output")

    graph = LogicalGraph(
        nodes=[n3, n1, n2], edges=[LogicalEdge("n1", "n2"), LogicalEdge("n2", "n3")]
    )

    sorted_nodes = topological_sort(graph)
    assert [n.id for n in sorted_nodes] == ["n1", "n2", "n3"]


def test_topological_sort_disconnected():
    """Test disconnected graph topological sort."""
    n1 = LogicalNode("n1", "Input")
    n2 = LogicalNode("n2", "Output")
    n3 = LogicalNode("n3", "Floating")

    graph = LogicalGraph(nodes=[n1, n2, n3], edges=[LogicalEdge("n1", "n2")])

    sorted_nodes = topological_sort(graph)
    assert [n.id for n in sorted_nodes] == ["n1", "n3", "n2"]


def test_topological_sort_cycle():
    """Test cycle handling in topological sort."""
    n1 = LogicalNode("n1", "Node1")
    n2 = LogicalNode("n2", "Node2")
    n3 = LogicalNode("n3", "Node3")

    graph = LogicalGraph(
        nodes=[n1, n2, n3],
        edges=[
            LogicalEdge("n1", "n2"),
            LogicalEdge("n2", "n3"),
            LogicalEdge("n3", "n1"),
        ],
    )

    sorted_nodes = topological_sort(graph)
    assert len(sorted_nodes) == 3
    assert set(n.id for n in sorted_nodes) == {"n1", "n2", "n3"}


def test_topological_sort_cycle_with_root():
    """Test a cycle where another root node feeds into it."""
    n1 = LogicalNode("n1", "Root")
    n2 = LogicalNode("n2", "Cycle1")
    n3 = LogicalNode("n3", "Cycle2")

    graph = LogicalGraph(
        nodes=[n1, n2, n3],
        edges=[
            LogicalEdge("n1", "n2"),
            LogicalEdge("n2", "n3"),
            LogicalEdge("n3", "n2"),  # cycle
        ],
    )
    sorted_nodes = topological_sort(graph)
    assert len(sorted_nodes) == 3


def test_topological_sort_missing_nodes():
    """Test edge referencing non-existent nodes."""
    n1 = LogicalNode("n1", "Input")
    n2 = LogicalNode("n2", "Floating")

    graph = LogicalGraph(
        nodes=[n1, n2], edges=[LogicalEdge("n1", "n3"), LogicalEdge("n4", "n2")]
    )

    sorted_nodes = topological_sort(graph)
    assert set(n.id for n in sorted_nodes) == {"n1", "n2"}


def test_dataclasses_coverage():
    """Ensure dataclass instantiation logic works cleanly for coverage."""
    axis = LogicalAxis(name="batch", size=32)
    assert axis.name == "batch"
    assert axis.size == 32

    spec = PartitionSpec(axes=("data", None))
    assert spec.axes == ("data", None)

    mesh = LogicalMesh(shape={"data": 4})
    assert mesh.shape["data"] == 4

    node = LogicalNode(id="x", kind="Linear", sharding=spec)
    assert node.metadata == {}
    assert node.sharding == spec


def test_not_implemented_errors():
    """Test that abstract methods raise NotImplementedError."""

    class PartialBackend(CompilerBackend):
        """A partial implementation of CompilerBackend for testing."""

        def compile(self, graph):
            """Override compile to call super().

            Args:
                graph (LogicalGraph): The logical graph.

            Returns:
                Any: The compiled output.

            """
            return super().compile(graph)

    class PartialFrontend(GraphFrontend):
        """A partial implementation of GraphFrontend for testing."""

        def parse_to_graph(self, code):
            """Override parse_to_graph to call super().

            Args:
                code (str): The source code.

            Returns:
                LogicalGraph: The parsed graph.

            """
            return super().parse_to_graph(code)

    with pytest.raises(NotImplementedError):
        PartialBackend().compile(LogicalGraph())

    with pytest.raises(NotImplementedError):
        PartialFrontend().parse_to_graph("")


def test_cli_parse_json():
    """Test JSON parsing inside CLI."""
    json_data = json.dumps(
        {
            "name": "TestModel",
            "nodes": [
                {"id": "n1", "kind": "Input", "metadata": {"shape": "2"}},
                {"id": "n2", "kind": "Output"},
            ],
            "edges": [{"source": "n1", "target": "n2"}],
        }
    )

    graph = _parse_graph_from_json(json_data)
    assert graph.name == "TestModel"
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.nodes[0].metadata["shape"] == "2"

    graph_empty = _parse_graph_from_json("{}")
    assert graph_empty.name == "Model"
    assert len(graph_empty.nodes) == 0
    assert len(graph_empty.edges) == 0


def test_cli_main(capsys):
    """Test CLI main entrypoint.

    Args:
        capsys: Pytest fixture to capture stdout and stderr.

    """
    json_data = json.dumps(
        {
            "name": "TestModel",
            "nodes": [{"id": "n1", "kind": "Input"}, {"id": "n2", "kind": "Output"}],
            "edges": [{"source": "n1", "target": "n2"}],
        }
    )

    with NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(json_data)
        f_name = f.name

    cli_main(["toposort", f_name])

    captured = capsys.readouterr()
    assert "Topological Order:" in captured.out
    assert "n1 (Input)" in captured.out
    assert "n2 (Output)" in captured.out


def test_cli_main_sys_argv(monkeypatch, capsys):
    """Test CLI main using sys.argv.

    Args:
        monkeypatch: Pytest fixture to mock attributes.
        capsys: Pytest fixture to capture stdout and stderr.

    """
    json_data = json.dumps({"nodes": [], "edges": []})
    with NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(json_data)
        f_name = f.name

    monkeypatch.setattr("sys.argv", ["ml-switcheroo-ir", "toposort", f_name])
    cli_main()

    captured = capsys.readouterr()
    assert "Topological Order:" in captured.out


def test_runpy_main_module():
    """Execute __main__.py to get coverage."""
    with NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("{}")
        f_name = f.name
    with patch("sys.argv", ["ml-switcheroo-ir", "toposort", f_name]):
        runpy.run_module("ml_switcheroo_ir.__main__", run_name="__main__")


def test_runpy_cli_module():
    """Execute cli.py to get coverage on its __main__ block."""
    with NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("{}")
        f_name = f.name
    with patch("sys.argv", ["ml-switcheroo-ir", "toposort", f_name]):
        runpy.run_module("ml_switcheroo_ir.cli", run_name="__main__")


def test_cli_main_other_command(monkeypatch, capsys):
    """Test CLI main with another mock command.

    Args:
        monkeypatch: Pytest fixture to mock attributes.
        capsys: Pytest fixture to capture stdout and stderr.

    """

    class MockArgs:
        """A mock arguments class for testing."""

        command = "other"

    monkeypatch.setattr(
        "argparse.ArgumentParser.parse_args", lambda self, args: MockArgs()
    )
    cli_main(["other"])


def test_verify_backend_missing_file(capsys):
    """Test verify-backend with a missing file.

    Args:
        capsys: Pytest fixture to capture stdout and stderr.

    """
    cli_main(["verify-backend", "nonexistent_file_12345.py", "MyClass"])
    captured = capsys.readouterr()
    assert "not found" in captured.out
    assert "Compliance: 0%" in captured.out


def test_verify_backend_spec_none(capsys, monkeypatch):
    """Test verify-backend when importlib spec is None.

    Args:
        capsys: Pytest fixture to capture stdout and stderr.
        monkeypatch: Pytest fixture to mock attributes.

    """
    monkeypatch.setattr("importlib.util.spec_from_file_location", lambda n, p: None)
    with NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("")
        fname = f.name
    cli_main(["verify-backend", fname, "MyClass"])
    captured = capsys.readouterr()
    assert "Error: Could not load module" in captured.out


def test_verify_backend_execution_error(capsys):
    """Test verify-backend when module execution fails.

    Args:
        capsys: Pytest fixture to capture stdout and stderr.

    """
    with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("import nonexistent_module_123")
        fname = f.name
    cli_main(["verify-backend", fname, "MyClass"])
    captured = capsys.readouterr()
    assert "Error executing module" in captured.out
    assert "Compliance: 0%" in captured.out


def test_verify_backend_missing_class(capsys):
    """Test verify-backend when the specified class is missing.

    Args:
        capsys: Pytest fixture to capture stdout and stderr.

    """
    with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("x = 1")
        fname = f.name
    cli_main(["verify-backend", fname, "MyClass"])
    captured = capsys.readouterr()
    assert "not found in" in captured.out
    assert "Compliance: 20%" in captured.out


def test_verify_backend_not_a_class(capsys):
    """Test verify-backend when the specified name is not a class.

    Args:
        capsys: Pytest fixture to capture stdout and stderr.

    """
    with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("MyClass = 1")
        fname = f.name
    cli_main(["verify-backend", fname, "MyClass"])
    captured = capsys.readouterr()
    assert "Compliance: 40%" in captured.out


def test_verify_backend_no_graph_arg(capsys):
    """Test verify-backend when the compile method is missing the graph arg.

    Args:
        capsys: Pytest fixture to capture stdout and stderr.

    """
    with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            "from ml_switcheroo_ir import CompilerBackend\nclass MyClass(CompilerBackend):\n    def compile(self, wrong_arg):\n        pass"
        )
        fname = f.name
    cli_main(["verify-backend", fname, "MyClass"])
    captured = capsys.readouterr()
    assert "Compliance: 80%" in captured.out


def test_verify_backend_perfect(capsys):
    """Test verify-backend with a perfect implementation.

    Args:
        capsys: Pytest fixture to capture stdout and stderr.

    """
    with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            "from ml_switcheroo_ir import CompilerBackend\nclass MyClass(CompilerBackend):\n    def compile(self, graph):\n        pass"
        )
        fname = f.name
    cli_main(["verify-backend", fname, "MyClass"])
    captured = capsys.readouterr()
    assert "Compliance: 100%" in captured.out


def test_verify_backend_class_not_inheriting(capsys):
    """Test verify-backend when the class does not inherit CompilerBackend.

    Args:
        capsys: Pytest fixture to capture stdout and stderr.

    """
    with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("class MyClass:\n    pass")
        fname = f.name
    cli_main(["verify-backend", fname, "MyClass"])
    captured = capsys.readouterr()
    assert "Compliance: 40%" in captured.out


def test_cli_tabulate_fallback(monkeypatch, capsys):
    import sys

    monkeypatch.setitem(sys.modules, "tabulate", None)
    import importlib
    import ml_switcheroo_ir.cli

    importlib.reload(ml_switcheroo_ir.cli)
    from ml_switcheroo_ir.cli import tabulate

    res = tabulate([["a", "b"]], ["A", "B"])
    assert "A | B" in res


def test_cli_main_invalid_command():
    from ml_switcheroo_ir.cli import main as cli_main
    import pytest

    with pytest.raises(SystemExit):
        cli_main(["invalid_command"])


def test_cli_main_validate_valid(capsys, tmp_path):
    from ml_switcheroo_ir.cli import main as cli_main

    f = tmp_path / "graph.json"
    f.write_text('{"nodes": [], "edges": []}')
    with pytest.raises(SystemExit) as e:
        cli_main(["validate", str(f)])
    assert e.value.code == 0
    assert "Graph is valid." in capsys.readouterr().out


def test_cli_main_validate_invalid(capsys, tmp_path):
    from ml_switcheroo_ir.cli import main as cli_main

    f = tmp_path / "graph.json"
    f.write_text('{"nodes": [{"id": "1", "kind": "InvalidOp"}], "edges": []}')
    with pytest.raises(SystemExit) as e:
        cli_main(["validate", str(f)])
    assert e.value.code == 1
    assert "ERROR" in capsys.readouterr().out


def test_cli_main_validate_custom_ops(capsys, tmp_path):
    from ml_switcheroo_ir.cli import main as cli_main

    f = tmp_path / "graph.json"
    f.write_text(
        '{"nodes": [{"id": "1", "kind": "MyOp", "domain": "custom"}], "edges": []}'
    )
    c = tmp_path / "custom.json"
    c.write_text('{"MyOp": {"name": "MyOp", "domain": "custom", "attributes": {}}}')
    with pytest.raises(SystemExit) as e:
        cli_main(["validate", str(f), "--custom-ops", str(c)])
    assert e.value.code == 0


def test_cli_main_list_ops(capsys):
    from ml_switcheroo_ir.cli import main as cli_main

    cli_main(["list-ops", "--domain", "ai.onnx", "--search", "Abs"])
    out = capsys.readouterr().out
    assert "Abs" in out


def test_cli_compliance_not_found(capsys):
    from ml_switcheroo_ir.cli import main as cli_main
    import pytest

    with pytest.raises(SystemExit) as e:
        cli_main(["compliance", "nonexistent_path"])
    assert e.value.code == 1


def test_cli_compliance_file(tmp_path, capsys):
    from ml_switcheroo_ir.cli import main as cli_main

    f = tmp_path / "test_file.py"
    f.write_text("class Test:\n  def forward(self):\n    pass")
    cli_main(["compliance", str(f)])
    captured = capsys.readouterr()
    assert "Compliance Report" in captured.out


def test_cli_compliance_register_framework(tmp_path, capsys):
    from ml_switcheroo_ir.cli import main as cli_main

    f = tmp_path / "test_fw.py"
    f.write_text(
        "@register_framework('my_fw')\nclass MyAdapter:\n    def definitions(self):\n        return {'Add': StandardMap()}\n"
    )
    cli_main(["compliance", str(f)])
    captured = capsys.readouterr()
    assert "Compliance Report" in captured.out


def test_cli_compliance_backend_frontend(tmp_path, capsys):
    from ml_switcheroo_ir.cli import main as cli_main

    f = tmp_path / "test_be.py"
    f.write_text(
        "class MyBackend(CompilerBackend):\n    def compile(self, graph):\n        pass\nclass MyFrontend(GraphFrontend):\n    def parse_to_graph(self, code):\n        pass"
    )
    cli_main(["compliance", str(f)])
    captured = capsys.readouterr()
    assert "Compliance Report" in captured.out


def test_cli_compliance_verbose(tmp_path, capsys):
    from ml_switcheroo_ir.cli import main as cli_main

    f = tmp_path / "test_be.py"
    f.write_text('def my_func():\n    return {"Add": 1}\n')
    cli_main(["compliance", str(f), "-v"])
    captured = capsys.readouterr()
    assert "Verbose Missing Operations Report" in captured.out


def test_cli_compliance_no_targets(tmp_path, capsys):
    from ml_switcheroo_ir.cli import main as cli_main

    # Empty directory to avoid dialect ops
    d = tmp_path / "empty_dir"
    d.mkdir()
    cli_main(["compliance", str(d)])
    captured = capsys.readouterr()
    assert "No IR, FrameworkAdapter, or DIALECT targets detected" in captured.out


def test_cli_compliance_directory(tmp_path, capsys):
    from ml_switcheroo_ir.cli import main as cli_main

    (tmp_path / "node_modules").mkdir()
    (tmp_path / ".venv").mkdir()
    f = tmp_path / "test_be.py"
    f.write_text(
        "class MyBackend(CompilerBackend):\n    def compile(self, graph):\n        pass\n"
    )
    cli_main(["compliance", str(tmp_path)])
    captured = capsys.readouterr()
    assert "Compliance Report" in captured.out


def test_cli_main_validate_warnings_only(capsys, tmp_path):
    from ml_switcheroo_ir.cli import main as cli_main

    f = tmp_path / "graph.json"
    # We need a node that generates a warning.
    # In validator, if node.kind is in registry but some attribute is missing and not strict, is it an error or warning?
    # Actually, missing required attribute is ERROR.
    # Unknown attribute is WARNING.
    f.write_text(
        '{"nodes": [{"id": "1", "kind": "Add", "metadata": {"unknown_attr": "val"}}], "edges": []}'
    )
    import pytest

    with pytest.raises(SystemExit) as e:
        cli_main(["validate", str(f)])
    assert e.value.code == 0
    assert "WARNING" in capsys.readouterr().out


def test_cli_compliance_exception(monkeypatch, tmp_path, capsys):
    from ml_switcheroo_ir.cli import main as cli_main

    # Create an unparseable file to trigger exception
    f = tmp_path / "test_bad.py"
    f.write_text("class Test(:")
    cli_main(["compliance", str(f)])
    assert "Compliance Report" in capsys.readouterr().out


def test_cli_compliance_json_exception(tmp_path, capsys):
    from ml_switcheroo_ir.cli import main as cli_main

    f = tmp_path / "test_fw.py"
    f.write_text(
        "class MyAdapter:\n    def definitions(self):\n        return 'my_adapter'"
    )
    # Create invalid json file
    j = tmp_path / "my_adapter.json"
    j.write_text("invalid json")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    cli_main(["compliance", "test_fw.py"])
    assert "Compliance Report" in capsys.readouterr().out
