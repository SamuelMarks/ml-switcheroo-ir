import subprocess
import json
import re
import sys


def run_command(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
        sys.exit(result.returncode)
    return result.stdout


def get_color(cov):
    if cov >= 90:
        return "brightgreen"
    elif cov >= 80:
        return "green"
    elif cov >= 70:
        return "yellow"
    elif cov >= 60:
        return "orange"
    else:
        return "red"


def main():
    print("Running tests and calculating coverage...")
    subprocess.run(
        ["pytest", "tests/", "--cov=src/ml_switcheroo_ir", "--cov-report=json"],
        check=True,
    )

    with open("coverage.json", "r") as f:
        cov_data = json.load(f)
    test_cov = round(cov_data["totals"]["percent_covered"], 1)

    print("Running interrogate for doc coverage...")
    result = subprocess.run(
        ["interrogate", "src/ml_switcheroo_ir"], capture_output=True, text=True
    )
    match = re.search(r"actual: ([\d\.]+)%", result.stdout)
    if match:
        doc_cov = float(match.group(1))
    else:
        print("Could not parse interrogate output:")
        print(result.stdout)
        sys.exit(1)

    test_badge = f"![Test Coverage](https://img.shields.io/badge/Test_Coverage-{test_cov}%25-{get_color(test_cov)})"
    doc_badge = f"![Doc Coverage](https://img.shields.io/badge/Doc_Coverage-{doc_cov}%25-{get_color(doc_cov)})"

    print(f"Test Coverage: {test_cov}%")
    print(f"Doc Coverage: {doc_cov}%")

    with open("README.md", "r") as f:
        readme = f.read()

    test_regex = re.compile(r"!\[test coverage\][^\n]+", re.IGNORECASE)
    doc_regex = re.compile(r"!\[doc coverage\][^\n]+", re.IGNORECASE)

    if test_regex.search(readme):
        readme = test_regex.sub(test_badge, readme)
    else:
        readme = re.sub(r"(# ml-switcheroo-ir\n+)", rf"\1{test_badge}\n", readme)

    if doc_regex.search(readme):
        readme = doc_regex.sub(doc_badge, readme)
    else:
        readme = re.sub(r"(!\[Test Coverage\][^\n]+\n)", rf"\1{doc_badge}\n", readme)

    with open("README.md", "w") as f:
        f.write(readme)


if __name__ == "__main__":
    main()
