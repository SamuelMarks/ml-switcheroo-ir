from ml_switcheroo_ir.cli import main as cli_main


def test_cli_compliance_verbose(tmp_path, capsys):
    """Test compliance subcommand verbose output."""
    from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY

    # We just need to mock an incomplete dialect file so the verbose block triggers
    dialect_file = tmp_path / "my_dialect.py"
    dialect_file.write_text("def Abs(x):\n    pass\n")

    cli_main(["compliance", str(tmp_path), "--verbose"])
    captured = capsys.readouterr()

    assert "Verbose Missing Operations Report" in captured.out
    assert "### my_dialect.py" in captured.out

    # Find some op we know is missing, e.g., 'Add' if it's in the registry
    dialect_ops = {k.split(".")[-1] for k in ONNX_REGISTRY.keys()}
    if "Add" in dialect_ops:
        assert "- [ ] **Add**" in captured.out
        assert "```json" in captured.out
        assert '"domain":' in captured.out
        assert '"inputs":' in captured.out
