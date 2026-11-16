"""Source files pinned by sha256: fetch a missing one if allowed, then verify it.

A preparer that reads a released file names it as a :class:`PinnedFile` and
calls :func:`ensure_pinned_file` before reading; the returned
:class:`~remedi_data.bundle.provenance.FileHash` goes into ``provenance.yaml``.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from remedi_data.bundle.bundle import sha256_of_file
from remedi_data.bundle.provenance import FileHash

logger = logging.getLogger(__name__)

#: Seconds a download may stall before it fails.
_DOWNLOAD_TIMEOUT = 60


class PinnedFileError(ValueError):
    """A pinned file is missing, cannot be fetched, or has another sha256."""


class PinnedFile(BaseModel):
    """One released file: its name in the raw directory, its sha256, its url."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file_name: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    #: Where a missing file is fetched from; ``None`` if it must be placed by hand.
    url: str | None = None


def _download(url: str, target: Path) -> None:
    """Fetch ``url`` to ``target`` through a temporary file beside it, so an
    interrupted transfer never leaves a truncated file under the real name."""
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s", url)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as response:
                shutil.copyfileobj(response, handle)
        except (urllib.error.URLError, OSError) as error:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise PinnedFileError(f"downloading {url} failed: {error}") from error
    temporary.replace(target)


def ensure_pinned_file(
    pinned: PinnedFile, directory: Path, *, download: bool
) -> FileHash:
    """The verified hash of ``directory / pinned.file_name``.

    A missing file is downloaded from ``pinned.url`` when ``download`` is true;
    a file already present is never fetched again.

    Raises:
        PinnedFileError: if the file is missing and cannot be downloaded, the
            download fails, or the file's sha256 is not the pinned one.
    """
    path = Path(directory) / pinned.file_name
    if not path.is_file():
        if pinned.url is None:
            raise PinnedFileError(
                f"missing raw file {path}; it has to be placed by hand"
            )
        if not download:
            raise PinnedFileError(
                f"missing raw file {path}; enable the download to fetch {pinned.url}"
            )
        _download(pinned.url, path)
    actual = sha256_of_file(path)
    if actual != pinned.sha256:
        raise PinnedFileError(
            f"{path} has sha256 {actual}, but {pinned.sha256} is pinned"
        )
    return FileHash(sha256=actual)
