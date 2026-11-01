"""Workspace layout checks that the package build does not catch by itself."""

from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["remedi"]
# The sdist refuses symlinks that leave the package directory, so each package
# carries real copies of these root files; they must not drift.
COPIED_ROOT_FILES = ["README.md", "LICENSE", "NOTICE"]


@pytest.mark.parametrize("package", PACKAGES)
@pytest.mark.parametrize("file_name", COPIED_ROOT_FILES)
def test_package_copy_of_root_file_is_identical(package: str, file_name: str):
    root_file = WORKSPACE_ROOT / file_name
    package_file = WORKSPACE_ROOT / "packages" / package / file_name
    assert package_file.read_bytes() == root_file.read_bytes(), (
        f"packages/{package}/{file_name} differs from the root {file_name}; "
        f"copy the root file over it"
    )
