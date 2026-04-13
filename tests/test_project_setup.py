"""Tests for project infrastructure setup (Issue #1).

Verifies that all required configuration files exist with correct content.
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_pyproject_toml_exists():
    assert (ROOT / "pyproject.toml").exists(), "pyproject.toml must exist at repo root"


def test_pyproject_toml_has_build_system():
    content = (ROOT / "pyproject.toml").read_text()
    assert "[build-system]" in content


def test_pyproject_toml_has_project_section():
    content = (ROOT / "pyproject.toml").read_text()
    assert "[project]" in content
    assert 'requires-python = ">=3.12"' in content


def test_pyproject_toml_has_pytest_config():
    content = (ROOT / "pyproject.toml").read_text()
    assert "[tool.pytest.ini_options]" in content
    assert 'testpaths = ["tests"]' in content


def test_environment_yml_exists():
    assert (ROOT / "environment.yml").exists(), (
        "environment.yml must exist at repo root"
    )


def test_environment_yml_specifies_monopoly_env():
    content = (ROOT / "environment.yml").read_text()
    assert "monopoly" in content, "environment.yml must name the 'monopoly' conda env"


def test_environment_yml_specifies_python_312():
    content = (ROOT / "environment.yml").read_text()
    assert "3.12" in content, "environment.yml must specify Python 3.12"


def test_gitignore_exists():
    assert (ROOT / ".gitignore").exists(), ".gitignore must exist at repo root"


def test_gitignore_has_standard_python_ignores():
    content = (ROOT / ".gitignore").read_text()
    required_patterns = [
        "__pycache__/",
        "*.egg-info/",
        ".pytest_cache/",
        "dist/",
        "build/",
        "*.pyc",
    ]
    for pattern in required_patterns:
        assert pattern in content, f".gitignore must contain '{pattern}'"


def test_src_monopoly_init_exists():
    assert (ROOT / "src" / "monopoly" / "__init__.py").exists()


def test_tests_init_exists():
    assert (ROOT / "tests" / "__init__.py").exists()
