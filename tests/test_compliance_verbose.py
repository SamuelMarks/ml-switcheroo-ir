from ml_switcheroo_ir.cli import main as cli_main


def test_cli_compliance_verbose(tmp_path, capsys, monkeypatch):
    """Test compliance subcommand verbose output."""
    from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY
    import ml_switcheroo_ir.compliance

    original_get_dialect_ops = ml_switcheroo_ir.compliance.get_dialect_ops

    def mock_get_dialect_ops():
        ops = original_get_dialect_ops()
        ops.add("FakeOp")
        return ops

    monkeypatch.setattr(
        ml_switcheroo_ir.compliance, "get_dialect_ops", mock_get_dialect_ops
    )

    # We just need to mock an incomplete dialect file so the verbose block triggers
    dialect_file = tmp_path / "my_dialect.py"
    dialect_file.write_text("def Abs(x):\n    pass\n")

    cli_main(["compliance", str(tmp_path), "--verbose"])
    captured = capsys.readouterr()

    assert "Verbose Missing Operations Report" in captured.out
    assert "### " in captured.out

    # Find some op we know is missing, e.g., 'Add' if it's in the registry
    dialect_ops = {k.split(".")[-1] for k in ONNX_REGISTRY.keys()}
    if "Add" in dialect_ops:
        assert "- [ ] **Add**" in captured.out
        assert "```json" in captured.out
        assert '"domain":' in captured.out
        assert '"inputs":' in captured.out


def test_cli_compliance_verbose_mapping(tmp_path, capsys):
    """Test compliance subcommand verbose output with mapping."""
    import json

    dialect_file = tmp_path / "my_dialect.py"
    dialect_file.write_text("def Abs(x):\n    pass\n")

    mapping_file = tmp_path / "mapping.json"
    mapping_data = {
        "Add": {"api": "os.path.join"},  # valid function with signature and doc
        "Sub": {"api": "builtins.int"},  # covers type/class branch
        "Mul": {"api": "tf.add"},  # tests tf. replacement
        "Div": {"api": "sys.stdout"},  # not callable / no signature
        "Mod": {"api": "os.name"},  # valid module attribute
        "Exp": {"api": "jnp.exp"},  # tests jnp. replacement
    }
    with open(mapping_file, "w") as f:
        json.dump(mapping_data, f)

    cli_main(
        ["compliance", str(dialect_file), "--verbose", "--mapping", str(mapping_file)]
    )
    captured = capsys.readouterr()

    assert "Verbose Missing Operations Report" in captured.out
    assert "Mapped API targets" in captured.out
    assert "os.path" in captured.out
    assert "join" in captured.out


def test_cli_compliance_verbose_mapping_empty(tmp_path, capsys):
    """Test compliance subcommand verbose output with empty mapping."""
    import json

    dialect_file = tmp_path / "my_dialect.py"
    dialect_file.write_text("def Abs(x):\n    pass\n")

    mapping_file = tmp_path / "mapping.json"
    with open(mapping_file, "w") as f:
        json.dump({}, f)

    cli_main(
        ["compliance", str(dialect_file), "--verbose", "--mapping", str(mapping_file)]
    )
    captured = capsys.readouterr()

    assert "Verbose Missing Operations Report" in captured.out
    assert "No mappings found in" in captured.out
