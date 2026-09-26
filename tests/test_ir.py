"""Tests for ml_switcheroo_ir package."""

import json
import pathlib
import runpy
from tempfile import NamedTemporaryFile
from typing import Any
from unittest.mock import patch

import pytest

from ml_switcheroo_ir import (
    CompilerBackend,
    DType,
    GraphFrontend,
    LogicalAxis,
    LogicalEdge,
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    PartitionSpec,
    TensorSpec,
    __version__,
    eliminate_dead_nodes,
    topological_sort,
)
from ml_switcheroo_ir.cli import _parse_graph_from_json
from ml_switcheroo_ir.cli import main as cli_main


def test_package_version() -> None:
    """Test package version exposure."""
    assert __version__ == "0.0.3"


def test_topological_sort_linear() -> None:
    """Test standard linear topological sort."""
    n1 = LogicalNode("n1", "Input")
    n2 = LogicalNode("n2", "Linear", inputs=["n1"])
    n3 = LogicalNode("n3", "Output", inputs=["n2"])

    graph = LogicalGraph(nodes={"n1": n1, "n2": n2, "n3": n3})

    sorted_nodes = topological_sort(graph)
    assert [n.id for n in sorted_nodes] == ["n1", "n2", "n3"]


def test_topological_sort_disconnected() -> None:
    """Test disconnected graph topological sort."""
    n1 = LogicalNode("n1", "Input")
    n2 = LogicalNode("n2", "Output", inputs=["n1"])
    n3 = LogicalNode("n3", "Floating")

    graph = LogicalGraph(nodes={"n1": n1, "n2": n2, "n3": n3})

    sorted_nodes = topological_sort(graph)
    assert [n.id for n in sorted_nodes] == ["n1", "n3", "n2"]


def test_topological_sort_cycle() -> None:
    """Test cycle handling in topological sort."""
    n1 = LogicalNode("n1", "Node1", inputs=["n3"])
    n2 = LogicalNode("n2", "Node2", inputs=["n1"])
    n3 = LogicalNode("n3", "Node3", inputs=["n2"])

    graph = LogicalGraph(nodes={"n1": n1, "n2": n2, "n3": n3})

    sorted_nodes = topological_sort(graph)
    assert len(sorted_nodes) == 3
    assert {n.id for n in sorted_nodes} == {"n1", "n2", "n3"}


def test_topological_sort_cycle_with_root() -> None:
    """Test a cycle where another root node feeds into it."""
    n1 = LogicalNode("n1", "Root")
    n2 = LogicalNode("n2", "Cycle1", inputs=["n1", "n3"])
    n3 = LogicalNode("n3", "Cycle2", inputs=["n2"])

    graph = LogicalGraph(nodes={"n1": n1, "n2": n2, "n3": n3})
    sorted_nodes = topological_sort(graph)
    assert len(sorted_nodes) == 3


def test_topological_sort_missing_nodes() -> None:
    """Test edge referencing non-existent nodes."""
    n1 = LogicalNode("n1", "Input")
    n2 = LogicalNode("n2", "Floating", inputs=["n4"])

    graph = LogicalGraph(nodes={"n1": n1, "n2": n2})

    sorted_nodes = topological_sort(graph)
    assert {n.id for n in sorted_nodes} == {"n1", "n2"}


def test_dataclasses_coverage() -> None:
    """Ensure dataclass instantiation logic works cleanly for coverage."""
    axis = LogicalAxis(name="batch", size=32)
    assert axis.name == "batch"
    assert axis.size == 32

    spec = PartitionSpec(axes=("data", None))
    assert spec.axes == ("data", None)

    mesh = LogicalMesh(shape={"data": 4})
    assert mesh.shape["data"] == 4

    node = LogicalNode(id="x", op_type="Linear", sharding=spec)
    assert node.attributes == {}
    assert node.sharding == spec


def test_not_implemented_errors() -> None:
    """Test that abstract methods raise NotImplementedError."""

    class PartialBackend(CompilerBackend):
        """A partial implementation of CompilerBackend for testing."""

        def compile(self, graph: LogicalGraph) -> object:
            """Override compile to call super().

            Args:
                graph (LogicalGraph): The logical graph.

            Returns:
                object: The compiled output.

            """
            return CompilerBackend.compile(self, graph)

    class PartialFrontend(GraphFrontend):
        """A partial implementation of GraphFrontend for testing."""

        def parse_to_graph(self, code: str) -> LogicalGraph:
            """Override parse_to_graph to call super().

            Args:
                code (str): The source code.

            Returns:
                LogicalGraph: The parsed graph.

            """
            return GraphFrontend.parse_to_graph(self, code)

    with pytest.raises(NotImplementedError):
        PartialBackend().compile(LogicalGraph())

    with pytest.raises(NotImplementedError):
        PartialFrontend().parse_to_graph("")


def test_cli_parse_json() -> None:
    """Test JSON parsing inside CLI."""
    json_data = json.dumps(
        {
            "name": "TestModel",
            "nodes": [
                {"id": "n1", "kind": "Input", "metadata": {"shape": "2"}},
                {"id": "n2", "kind": "Output", "inputs": ["n1"]},
            ],
        }
    )

    graph = _parse_graph_from_json(json_data)
    assert graph.name == "TestModel"
    assert len(graph.nodes) == 2
    assert len(graph.nodes["n2"].inputs) == 1
    assert graph.nodes["n1"].attributes["shape"] == "2"

    graph_empty = _parse_graph_from_json("{}")
    assert graph_empty.name == "Model"
    assert len(graph_empty.nodes) == 0
    assert len(graph_empty.nodes) == 0


def test_cli_main(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI main entrypoint.

    Args:
        capsys: Pytest fixture to capture stdout and stderr.

    """
    json_data = json.dumps(
        {
            "name": "TestModel",
            "nodes": [
                {"id": "n1", "kind": "Input"},
                {"id": "n2", "kind": "Output", "inputs": ["n1"]},
            ],
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


def test_cli_main_sys_argv(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
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


def test_runpy_main_module() -> None:
    """Execute __main__.py to get coverage."""
    import importlib
    import sys

    import ml_switcheroo_ir.__main__

    importlib.reload(ml_switcheroo_ir.__main__)

    with NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("{}")
        f_name = f.name
    sys.modules.pop("ml_switcheroo_ir.__main__", None)
    with patch("sys.argv", ["ml-switcheroo-ir", "toposort", f_name]):
        runpy.run_module("ml_switcheroo_ir.__main__", run_name="__main__")


def test_runpy_cli_module() -> None:
    """Execute cli.py to get coverage on its __main__ block."""
    import sys

    with NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("{}")
        f_name = f.name
    sys.modules.pop("ml_switcheroo_ir.cli", None)
    with patch("sys.argv", ["ml-switcheroo-ir", "toposort", f_name]):
        runpy.run_module("ml_switcheroo_ir.cli", run_name="__main__")


def test_cli_main_other_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
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


def test_verify_backend_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    """Test verify-backend with a missing file.

    Args:
        capsys: Pytest fixture to capture stdout and stderr.

    """
    cli_main(["verify-backend", "nonexistent_file_12345.py", "MyClass"])
    captured = capsys.readouterr()
    assert "not found" in captured.out
    assert "Compliance: 0%" in captured.out


def test_verify_backend_spec_none(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_verify_backend_execution_error(capsys: pytest.CaptureFixture[str]) -> None:
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


def test_verify_backend_missing_class(capsys: pytest.CaptureFixture[str]) -> None:
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


def test_verify_backend_not_a_class(capsys: pytest.CaptureFixture[str]) -> None:
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


def test_verify_backend_no_graph_arg(capsys: pytest.CaptureFixture[str]) -> None:
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


def test_verify_backend_perfect(capsys: pytest.CaptureFixture[str]) -> None:
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


def test_verify_backend_class_not_inheriting(
    capsys: pytest.CaptureFixture[str],
) -> None:
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


def test_cli_tabulate_fallback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test."""
    import sys

    monkeypatch.setitem(sys.modules, "tabulate", None)
    import importlib

    import ml_switcheroo_ir.cli

    importlib.reload(ml_switcheroo_ir.cli)
    from ml_switcheroo_ir.cli import tabulate

    res = tabulate([["a", "b"]], ["A", "B"])
    assert "A | B" in res


def test_cli_main_invalid_command() -> None:
    """Test."""
    import pytest

    from ml_switcheroo_ir.cli import main as cli_main

    with pytest.raises(SystemExit):
        cli_main(["invalid_command"])


def test_cli_main_validate_valid(
    capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    """Test."""
    from ml_switcheroo_ir.cli import main as cli_main

    f = tmp_path / "graph.json"
    f.write_text('{"nodes": []}')
    with pytest.raises(SystemExit) as e:
        cli_main(["validate", str(f)])
    assert e.value.code == 0
    assert "Graph is valid." in capsys.readouterr().out


def test_cli_main_validate_invalid(
    capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    """Test."""
    from ml_switcheroo_ir.cli import main as cli_main

    f = tmp_path / "graph.json"
    f.write_text('{"nodes": [{"id": "1", "kind": "InvalidOp"}]}')
    with pytest.raises(SystemExit) as e:
        cli_main(["validate", str(f)])
    assert e.value.code == 1
    assert "ERROR" in capsys.readouterr().out


def test_cli_main_validate_custom_ops(
    capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    """Test."""
    from ml_switcheroo_ir.cli import main as cli_main

    f = tmp_path / "graph.json"
    f.write_text('{"nodes": [{"id": "1", "kind": "MyOp", "domain": "custom"}]}')
    c = tmp_path / "custom.json"
    c.write_text('{"MyOp": {"name": "MyOp", "domain": "custom", "attributes": {}}}')
    with pytest.raises(SystemExit) as e:
        cli_main(["validate", str(f), "--custom-ops", str(c)])
    assert e.value.code == 0


def test_cli_main_list_ops(capsys: pytest.CaptureFixture[str]) -> None:
    """Test."""
    from ml_switcheroo_ir.cli import main as cli_main

    cli_main(["list-ops", "--domain", "ai.onnx", "--search", "Abs"])
    out = capsys.readouterr().out
    assert "Abs" in out


def test_cli_main_list_ops_no_filters(capsys: pytest.CaptureFixture[str]) -> None:
    """Test."""
    from ml_switcheroo_ir.cli import main as cli_main

    cli_main(["list-ops"])
    out = capsys.readouterr().out
    assert "Abs" in out
    assert "Add" in out


def test_cli_compliance_not_found(capsys: pytest.CaptureFixture[str]) -> None:
    """Test."""
    import pytest

    from ml_switcheroo_ir.cli import main as cli_main

    with pytest.raises(SystemExit) as e:
        cli_main(["compliance", "nonexistent_path"])
    assert e.value.code == 1


def test_cli_compliance_file(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test."""
    from ml_switcheroo_ir.cli import main as cli_main

    f = tmp_path / "test_file.py"
    f.write_text("class Test:\n  def forward(self):\n    pass")
    cli_main(["compliance", str(f)])
    captured = capsys.readouterr()
    assert "Compliance Report" in captured.out


def test_cli_compliance_register_framework(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test."""
    from ml_switcheroo_ir.cli import main as cli_main

    f = tmp_path / "test_fw.py"
    f.write_text(
        "@register_framework('my_fw')\nclass MyAdapter:\n    def definitions(self):\n        return {'Add': StandardMap()}\n"
    )
    cli_main(["compliance", str(f)])
    captured = capsys.readouterr()
    assert "Compliance Report" in captured.out


def test_cli_compliance_backend_frontend(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test."""
    from ml_switcheroo_ir.cli import main as cli_main

    f = tmp_path / "test_be.py"
    f.write_text(
        "class MyBackend(CompilerBackend):\n    def compile(self, graph):\n        pass\nclass MyFrontend(GraphFrontend):\n    def parse_to_graph(self, code):\n        pass"
    )
    cli_main(["compliance", str(f)])
    captured = capsys.readouterr()
    assert "Compliance Report" in captured.out


def test_cli_compliance_verbose(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test."""
    from ml_switcheroo_ir.cli import main as cli_main

    f = tmp_path / "test_be.py"
    f.write_text('def my_func():\n    return {"Add": 1}\n')
    cli_main(["compliance", str(f), "-v"])
    captured = capsys.readouterr()
    assert "Verbose Missing Operations Report" in captured.out


def test_cli_compliance_no_targets(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test."""
    from ml_switcheroo_ir.cli import main as cli_main

    # Empty directory to avoid dialect ops
    d = tmp_path / "empty_dir"
    d.mkdir()
    cli_main(["compliance", str(d)])
    captured = capsys.readouterr()
    assert "No IR, FrameworkAdapter, or DIALECT targets detected" in captured.out


def test_cli_compliance_directory(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test."""
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


def test_cli_main_validate_warnings_only(
    capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    """Test."""
    from ml_switcheroo_ir.cli import main as cli_main

    f = tmp_path / "graph.json"
    # We need a node that generates a warning.
    # In validator, if node.kind is in registry but some attribute is missing and not strict, is it an error or warning?
    # Actually, missing required attribute is ERROR.
    # Unknown attribute is WARNING.
    f.write_text(
        '{"nodes": [{"id": "1", "kind": "Add", "metadata": {"unknown_attr": "val"}}]}'
    )
    import pytest

    with pytest.raises(SystemExit) as e:
        cli_main(["validate", str(f)])
    assert e.value.code == 0
    assert "WARNING" in capsys.readouterr().out


def test_cli_compliance_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test."""
    from ml_switcheroo_ir.cli import main as cli_main

    # Create an unparseable file to trigger exception
    f = tmp_path / "test_bad.py"
    f.write_text("class Test(:")
    cli_main(["compliance", str(f)])
    assert "Compliance Report" in capsys.readouterr().out


def test_cli_compliance_json_exception(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test."""
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


def test_to_json_from_json() -> None:
    """Test."""
    spec = PartitionSpec(axes=("data", None))
    mesh = LogicalMesh(shape={"data": 4})
    node = LogicalNode(id="x", op_type="Linear", sharding=spec, shape_metadata=(1, 2))
    graph = LogicalGraph(nodes={"x": node}, mesh=mesh)

    json_str = graph.to_json()
    assert "Linear" in json_str

    graph2 = LogicalGraph.from_json(json_str)
    assert graph2.nodes["x"].sharding is not None
    assert graph2.nodes["x"].sharding.axes == ("data", None)
    assert graph2.nodes["x"].shape_metadata == (1, 2)


def test_node_dict_append() -> None:
    """Test NodeDict.append method with deprecation warning."""
    g = LogicalGraph(name="NodeDictGraph")
    node = LogicalNode(id="appended_node", op_type="Relu")
    with pytest.deprecated_call():
        g.nodes.append(node)
    assert "appended_node" in g.nodes
    assert g.nodes["appended_node"] == node
    assert isinstance(g.nodes, dict)


def test_node_dict_and_edge_list_sequence_ergonomics() -> None:
    """Test all NodeDict and EdgeList dual-ergonomics methods for full coverage."""
    from ml_switcheroo_ir import LogicalEdge, NodeDict

    g = LogicalGraph(name="SeqGraph")

    # Test edge appended before target node is added (pending edge)
    g.edges.append(LogicalEdge("n1", "n2"))
    assert len(g._pending_edges) == 1

    # Add n1 and n2 via extend
    n1 = LogicalNode("n1", "Input")
    n2 = LogicalNode("n2", "Relu")
    with pytest.deprecated_call():
        g.nodes.extend([n1, n2])

    # Pending edge wired into n2.inputs automatically
    assert n2.inputs == ["n1"]
    assert len(g.edges) == 1

    # EdgeList.extend
    g.edges.extend([LogicalEdge("n1", "n2")])

    # NodeDict.__getitem__ with int and out of range
    assert g.nodes[0] == n1
    assert g.nodes[1] == n2
    with pytest.raises(IndexError, match="out of range"):
        _ = g.nodes[999]

    # NodeDict.__contains__ with node object and invalid type
    assert g.nodes.__contains__(n1)
    assert "n1" in g.nodes
    assert not g.nodes.__contains__(12345)

    # NodeDict.index
    assert g.nodes.index(n1) == 0
    assert g.nodes.index("n2") == 1
    with pytest.raises(ValueError, match="not in NodeDict"):
        g.nodes.index("non_existent")

    # NodeDict.insert
    n3 = LogicalNode("n3", "Linear")
    g.nodes.insert(0, n3)
    assert "n3" in g.nodes

    # NodeDict.pop by int and by str
    popped = g.nodes.pop(0)
    assert popped == n1

    popped_str = g.nodes.pop("n3")
    assert popped_str == n3

    # NodeDict.__delitem__ by int and by str
    n4 = LogicalNode("n4", "GelU")
    with pytest.deprecated_call():
        g.nodes.append(n4)
    del g.nodes[0]  # deletes n2
    assert "n2" not in g.nodes

    del g.nodes["n4"]
    assert "n4" not in g.nodes

    # __setattr__ assignments on graph.nodes
    g.nodes = NodeDict(g, {"x": LogicalNode("x", "Op")})
    assert "x" in g.nodes

    g.nodes = [LogicalNode("y", "Op")]  # type: ignore[assignment]
    assert "y" in g.nodes

    g.nodes = {"z": LogicalNode("z", "Op")}  # type: ignore[assignment]
    assert "z" in g.nodes

    g.nodes = "invalid_not_dict"  # type: ignore
    assert len(g.nodes) == 0

    # LogicalNode 3-arg positional with attributes and explicit None op_type
    n_pos = LogicalNode("p1", "Conv", {"k": 3})
    assert n_pos.attributes == {"k": 3}
    assert n_pos.domain == "ai.onnx"

    n_none = LogicalNode("p2", None)
    assert n_none.op_type == ""

    # Non-dict non-str domain branch
    n_int_dom = LogicalNode("p3", "Conv", domain=12345)  # type: ignore[call-overload]
    assert n_int_dom.domain == "12345"


def test_nodedict_and_edge_list_edge_branches() -> None:
    """Test edge branches for NodeDict, EdgeList, and LogicalGraph edges."""
    from ml_switcheroo_ir import EdgeList, LogicalEdge, NodeDict

    # 1. EdgeList with no linked graph
    standalone_edges = EdgeList()
    standalone_edges.append(LogicalEdge("x", "y"))
    assert len(standalone_edges) == 1

    # 2. Edge append when source is not in target node inputs (line 567) and duplicate edge
    g = LogicalGraph(name="EdgeBranchGraph")
    b = LogicalNode("b", "Relu", inputs=["a"])
    g.nodes["b"] = b
    g.edges.append(LogicalEdge("a", "b"))  # already in inputs
    assert b.inputs == ["a"]

    c = LogicalNode("c", "Relu", inputs=[])
    g.nodes["c"] = c
    g.edges.append(LogicalEdge("a", "c"))  # not in inputs -> hits line 567
    assert c.inputs == ["a"]

    # 3. Standalone NodeDict without graph (line 615->exit)
    standalone_nd = NodeDict()
    standalone_nd["x"] = LogicalNode("x", "Op")
    assert "x" in standalone_nd

    nd_kwargs = NodeDict(a=LogicalNode("a", "Op"))
    assert "a" in nd_kwargs

    # 4. Wire pending edges: one matching target already containing source, one targeting another node
    g2 = LogicalGraph(name="PendingGraph")
    g2._pending_edges = [
        LogicalEdge("src_dup", "target_node"),
        LogicalEdge("other_src", "other_node"),
    ]
    target_node = LogicalNode("target_node", "Op", inputs=["src_dup"])
    g2.nodes["target_node"] = target_node
    assert target_node.inputs == ["src_dup"]

    # 5. LogicalGraph.edges with duplicate inputs in node
    dup_in_node = LogicalNode("dup_in", "Add", inputs=["s1", "s1"])
    g_dup = LogicalGraph(nodes={"dup_in": dup_in_node})
    assert len(g_dup.edges) == 2

    # 6. LogicalGraph.edges setter with duplicate edge
    with pytest.deprecated_call():
        g_dup.edges = [LogicalEdge("s1", "dup_in"), LogicalEdge("s1", "dup_in")]
    assert dup_in_node.inputs == ["s1"]

    # 7. __setattr__ exception handling on self.edges access and pending edge preservation
    g_edges_test = LogicalGraph(
        nodes={
            "surviving": LogicalNode("surviving", "Op", inputs=["in_a"]),
            "to_del": LogicalNode("to_del", "Op", inputs=["in_b"]),
        },
    )
    g_edges_test.nodes = [LogicalNode("surviving", "Op")]  # type: ignore[assignment]
    assert any(e.target == "to_del" for e in g_edges_test._pending_edges)

    err_prop = property(
        lambda self: (_ for _ in ()).throw(AttributeError("Mock error accessing edges"))
    )
    with patch.object(LogicalGraph, "edges", new=err_prop):
        g_dup.nodes = [LogicalNode("new_n", "Op")]  # type: ignore[assignment]
    assert "new_n" in g_dup.nodes


def test_output_pseudo_node_outputs_deduction() -> None:
    """Test that a pseudo Output node populates graph.outputs when outputs is omitted."""
    n1 = LogicalNode(id="n1", op_type="Relu")
    n_out = LogicalNode(id="out_node", op_type="Output", inputs=["n1", "n1"])
    graph = LogicalGraph(nodes={"n1": n1, "out_node": n_out})
    assert graph.outputs == ["n1"]


def test_from_dict_subgraph_edge_branches() -> None:
    """Test edge branches in from_dict for subgraphs and input_specs."""
    direct_sub = LogicalGraph(
        name="DirectSub",
        nodes={"s": LogicalNode(id="s", op_type="Relu")},
        outputs=["s"],
    )
    raw = {
        "nodes": {
            "n": {
                "id": "n",
                "op_type": "Op",
                "subgraphs": {"ignored": 123},
                "attributes": {
                    "body_subgraph": direct_sub,
                    "jvp_graph": "invalid_subgraph_type",
                },
            }
        },
        "input_specs": {"ignored_spec": 456},
    }
    g = LogicalGraph.from_dict(raw)
    assert "body" in g.nodes["n"].subgraphs
    assert g.nodes["n"].subgraphs["body"] == direct_sub


def test_from_dict_and_from_json_comprehensive() -> None:
    """Test comprehensive branches of from_dict with string dtypes, output_specs, input_specs, and subgraphs."""
    sub_graph = LogicalGraph(
        name="SubModel",
        nodes={"sub1": LogicalNode(id="sub1", op_type="Relu")},
        outputs=["sub1"],
    )
    spec_obj = TensorSpec(shape=(1, 4), dtype=DType.float32)

    raw_dict: dict[str, Any] = {
        "name": "FullTestGraph",
        "nodes": {
            "n1": {
                "id": "n1",
                "op_type": "Conv",
                "dtype": "float32",
                "output_specs": [{"shape": (1, 16), "dtype": "float32"}],
                "subgraphs": {
                    "sub_direct": sub_graph,
                    "sub_dict": {
                        "name": "NestedDict",
                        "nodes": {
                            "sub_dict_n": {"id": "sub_dict_n", "op_type": "Identity"}
                        },
                        "outputs": ["sub_dict_n"],
                    },
                },
            }
        },
        "inputs": ["in_x"],
        "input_specs": {
            "in_x": spec_obj,
            "in_y": {"shape": (2, 8), "dtype": "int32"},
        },
        "outputs": ["n1"],
    }

    g = LogicalGraph.from_dict(raw_dict)
    assert g.name == "FullTestGraph"
    assert g.nodes["n1"].dtype == DType.float32
    assert len(g.nodes["n1"].output_specs) == 1
    assert g.nodes["n1"].output_specs[0].shape == (1, 16)
    assert isinstance(g.nodes["n1"].subgraphs["sub_direct"], LogicalGraph)
    assert isinstance(g.nodes["n1"].subgraphs["sub_dict"], LogicalGraph)
    assert g.input_specs["in_x"] == spec_obj
    assert g.input_specs["in_y"].shape == (2, 8)
    assert g.input_specs["in_y"].dtype == DType.int32


def test_transforms_subgraphs_non_graph_branch() -> None:
    """Test eliminate_dead_nodes when subgraphs contains non-LogicalGraph values."""
    node = LogicalNode(
        id="custom_node",
        op_type="CustomOp",
        subgraphs={"metadata_str": "not_a_logical_graph"},
    )
    g = LogicalGraph(nodes={"custom_node": node}, outputs=["custom_node"])
    cleaned = eliminate_dead_nodes(g)
    assert (
        cleaned.nodes["custom_node"].subgraphs["metadata_str"] == "not_a_logical_graph"
    )


def test_legacy_ir_syntax_deprecations() -> None:
    """Test that legacy IR syntax produces structured DeprecationWarnings targeting removal in 0.1.0."""
    import warnings

    # 1. node.kind getter & setter
    n = LogicalNode(id="n1", op_type="Relu")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", DeprecationWarning)
        _ = n.kind
        assert any(
            "The 'kind' property is deprecated and will be removed in version 0.1.0"
            in str(w.message)
            for w in record
        )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", DeprecationWarning)
        n.kind = "Gelu"
        assert n.op_type == "Gelu"
        assert any(
            "The 'kind' setter is deprecated and will be removed in version 0.1.0"
            in str(w.message)
            for w in record
        )

    # 2. node.metadata getter & setter
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", DeprecationWarning)
        _ = n.metadata
        assert any(
            "The 'metadata' property is deprecated and will be removed in version 0.1.0"
            in str(w.message)
            for w in record
        )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", DeprecationWarning)
        n.metadata = {"axis": 1}
        assert n.attributes == {"axis": 1}
        assert any(
            "The 'metadata' setter is deprecated and will be removed in version 0.1.0"
            in str(w.message)
            for w in record
        )

    # 3. List-based nodes in LogicalGraph.__init__
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", DeprecationWarning)
        g = LogicalGraph(nodes=[LogicalNode(id="n_list", op_type="Relu")])
        assert "n_list" in g.nodes
        assert any(
            "Passing a list of nodes to LogicalGraph is deprecated and will be removed in version 0.1.0"
            in str(w.message)
            for w in record
        )

    # 4. NodeDict append, extend, insert
    g2 = LogicalGraph()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", DeprecationWarning)
        g2.nodes.append(LogicalNode(id="n_app", op_type="Relu"))
        assert "n_app" in g2.nodes
        assert any(
            "Using graph.nodes.append() is deprecated and will be removed in version 0.1.0"
            in str(w.message)
            for w in record
        )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", DeprecationWarning)
        g2.nodes.extend([LogicalNode(id="n_ext", op_type="Relu")])
        assert "n_ext" in g2.nodes
        assert any(
            "Using graph.nodes.extend() is deprecated and will be removed in version 0.1.0"
            in str(w.message)
            for w in record
        )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", DeprecationWarning)
        g2.nodes.insert(0, LogicalNode(id="n_ins", op_type="Relu"))
        assert "n_ins" in g2.nodes
        assert any(
            "Using graph.nodes.insert() is deprecated and will be removed in version 0.1.0"
            in str(w.message)
            for w in record
        )

    # 5. edges setter
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", DeprecationWarning)
        g2.edges = [LogicalEdge(source="n_app", target="n_ext")]
        assert any(
            "Setting 'edges' directly is deprecated and will be removed in version 0.1.0"
            in str(w.message)
            for w in record
        )

    # 6. Legacy Input and Output pseudo-nodes
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", DeprecationWarning)
        _ = LogicalGraph(nodes={"in_node": LogicalNode(id="in_node", op_type="Input")})
        assert any(
            "Legacy 'Input' pseudo-node auto-detection is deprecated and will be removed in version 0.1.0"
            in str(w.message)
            for w in record
        )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", DeprecationWarning)
        _ = LogicalGraph(
            nodes={
                "out_node": LogicalNode(id="out_node", op_type="Output", inputs=["x"])
            }
        )
        assert any(
            "Legacy 'Output' pseudo-node auto-detection is deprecated and will be removed in version 0.1.0"
            in str(w.message)
            for w in record
        )

    # 7. Legacy subgraph remappings in from_dict
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", DeprecationWarning)
        _ = LogicalGraph.from_dict(
            {
                "nodes": {
                    "n_sub": {
                        "id": "n_sub",
                        "op_type": "CustomLoop",
                        "attributes": {
                            "body_subgraph": {
                                "nodes": {"b": {"id": "b", "op_type": "Add"}}
                            }
                        },
                    }
                }
            }
        )
        assert any(
            "Legacy subgraph key 'body_subgraph' is deprecated and will be removed in version 0.1.0"
            in str(w.message)
            for w in record
        )
