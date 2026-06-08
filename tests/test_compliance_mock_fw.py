from ml_switcheroo_ir.cli import main as cli_main


def test_cli_compliance_framework_adapter(tmp_path, capsys):
    """Test compliance subcommand scanning mock framework adapters."""
    import json

    workspace = tmp_path / "workspace"
    frameworks_dir = workspace / "src" / "ml_switcheroo" / "frameworks"
    defs_dir = frameworks_dir / "definitions"

    frameworks_dir.mkdir(parents=True)
    defs_dir.mkdir(parents=True)

    # Create mock dynamic json
    mock_json = {"Abs": {"api": "foo.abs"}, "Add": {"api": "foo.add"}}
    with open(defs_dir / "mockfw.json", "w") as f:
        json.dump(mock_json, f)

    # Create mock python adapter
    adapter = frameworks_dir / "mockfw.py"
    adapter.write_text("""
@register_framework("mockfw")
class MockAdapter:
    def convert(self, data):
        pass
    def definitions(self):
        return load_definitions("mockfw")
""")

    cli_main(["compliance", str(workspace)])

    captured = capsys.readouterr()

    # Assert Framework tracking
    assert "FrameworkAdapter Compliance:" in captured.out
    assert "mockfw.py" in captured.out
    assert "100%" in captured.out
    assert "4/4" in captured.out

    # Assert Dialect tracking picked up from dynamic definitions
    assert "DIALECT Compliance" in captured.out
    assert "mockfw.py" in captured.out
