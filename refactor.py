import os


def replace_in_file(path, replacements):
    with open(path, "r") as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)


base_dir = "/Users/samuel/repos/ml-switcheroo-ir"

# Find all python files
py_files = []
for root, _, files in os.walk(base_dir):
    if ".venv" in root or ".git" in root or "third_party" in root:
        continue
    for file in files:
        if file.endswith(".py"):
            py_files.append(os.path.join(root, file))

for py_file in py_files:
    if "ghost.py" in py_file:
        # Ghost params use 'kind' and it might conflict. Let's be careful.
        pass
    else:
        replace_in_file(
            py_file,
            [
                ("node.op_type", "node.op_type"),
                ("node.attributes", "node.attributes"),
                ("op_type=", "op_type="),
                ('op_type="', 'op_type="'),
                ("op_type='", "op_type='"),
                ("attributes=", "attributes="),
            ],
        )
