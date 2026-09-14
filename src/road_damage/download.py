"""Download and safely extract the official RDD2022 archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import struct
import sys
import zipfile
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from road_damage.constants import (
    ARCHIVE_NAME,
    COUNTRIES,
    DEFAULT_COUNTRIES,
    FIGSHARE_DOWNLOAD_URL,
    FIGSHARE_MD5,
    FIGSHARE_SIZE_BYTES,
    PROJECT_ROOT,
)


class HttpRangeReader(io.RawIOBase):
    """Seekable, bounded-memory reader backed by HTTP byte-range requests."""

    def __init__(
        self, url: str, chunk_size: int = 8 * 1024 * 1024, cache_chunks: int = 4
    ):
        self.url = url
        self.chunk_size = chunk_size
        self.cache_chunks = cache_chunks
        self.position = 0
        self.session = requests.Session()
        retries = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.size = self._remote_size()
        self.cache: OrderedDict[int, bytes] = OrderedDict()

    def _remote_size(self) -> int:
        with self.session.get(
            self.url,
            headers={"Range": "bytes=0-0"},
            stream=True,
            timeout=(30, 300),
        ) as response:
            response.raise_for_status()
            content_range = response.headers.get("Content-Range", "")
            if response.status_code != 206 or "/" not in content_range:
                raise RuntimeError(
                    "The dataset host does not support selective byte-range downloads"
                )
            return int(content_range.rsplit("/", 1)[1])

    def _chunk(self, index: int) -> bytes:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start = index * self.chunk_size
        end = min(self.size - 1, start + self.chunk_size - 1)
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=(30, 300),
        )
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError("Dataset server ignored an HTTP range request")
        content = response.content
        expected = end - start + 1
        if len(content) != expected:
            raise IOError(
                f"Incomplete byte range {start}-{end}: received {len(content)} bytes"
            )
        self.cache[index] = content
        self.cache.move_to_end(index)
        while len(self.cache) > self.cache_chunks:
            self.cache.popitem(last=False)
        return content

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        output = bytearray()
        while len(output) < size:
            chunk_index = self.position // self.chunk_size
            chunk_offset = self.position % self.chunk_size
            chunk = self._chunk(chunk_index)
            take = min(size - len(output), len(chunk) - chunk_offset)
            output.extend(chunk[chunk_offset : chunk_offset + take])
            self.position += take
        return bytes(output)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if position < 0:
            raise ValueError("Cannot seek before the start of the remote file")
        self.position = position
        return self.position

    def tell(self) -> int:
        return self.position

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def close(self) -> None:
        self.session.close()
        super().close()


class BoundedReader(io.RawIOBase):
    """Expose a seekable window within another seekable binary stream."""

    def __init__(self, source: BinaryIO, start: int, size: int):
        self.source = source
        self.start = start
        self.size = size
        self.position = 0

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        self.source.seek(self.start + self.position)
        data = self.source.read(size)
        self.position += len(data)
        return data

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if position < 0:
            raise ValueError("Cannot seek before the start of the bounded stream")
        self.position = position
        return self.position

    def tell(self) -> int:
        return self.position

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - required to verify the publisher-provided checksum
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_resume(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        resumed = existing > 0 and response.status_code == 206
        mode = "ab" if resumed else "wb"
        initial = existing if resumed else 0
        total = int(response.headers.get("content-length", 0)) + initial
        with ExitStack() as stack:
            target = stack.enter_context(destination.open(mode))
            progress = stack.enter_context(
                tqdm(
                    total=total or None,
                    initial=initial,
                    unit="B",
                    unit_scale=True,
                    desc=destination.name,
                )
            )
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    target.write(chunk)
                    progress.update(len(chunk))


def _member_country(member: zipfile.ZipInfo) -> str | None:
    parts = PurePosixPath(member.filename).parts
    return next(
        (
            candidate
            for part in parts
            if (candidate := PurePosixPath(part).stem) in COUNTRIES
        ),
        None,
    )


def _stored_member_reader(source: BinaryIO, member: zipfile.ZipInfo) -> BoundedReader:
    """Create a direct view of a stored ZIP member without copying it to disk."""
    if member.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"Expected an uncompressed country archive: {member.filename}")
    source.seek(member.header_offset)
    header = source.read(30)
    if len(header) != 30:
        raise zipfile.BadZipFile(f"Truncated local header for {member.filename}")
    fields = struct.unpack("<4s2B4HL2L2H", header)
    if fields[0] != b"PK\x03\x04":
        raise zipfile.BadZipFile(f"Invalid local header for {member.filename}")
    filename_length, extra_length = fields[-2:]
    data_start = member.header_offset + len(header) + filename_length + extra_length
    return BoundedReader(source, data_start, member.file_size)


def _extract_bundle(
    bundle: zipfile.ZipFile, destination: Path, countries: set[str]
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    members = [
        member for member in bundle.infolist() if _member_country(member) in countries
    ]
    for member in tqdm(members, unit="file", desc="Extracting"):
        relative = PurePosixPath(member.filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe path in archive: {member.filename}")
        target = destination.joinpath(*relative.parts)
        target_resolved = target.resolve()
        if (
            destination_resolved not in target_resolved.parents
            and target_resolved != destination_resolved
        ):
            raise ValueError(f"Unsafe path in archive: {member.filename}")
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if target.is_file() and target.stat().st_size == member.file_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.name}.part")
        with bundle.open(member) as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        partial.replace(target)


def safe_extract(archive: Path, destination: Path, countries: set[str]) -> None:
    with archive.open("rb") as source:
        _extract_country_archives(source, destination, countries)


def _extract_country_archives(
    source: BinaryIO, destination: Path, countries: set[str]
) -> None:
    with zipfile.ZipFile(source) as outer:
        nested_archives = {
            country: member
            for member in outer.infolist()
            if member.filename.lower().endswith(".zip")
            and (country := _member_country(member)) is not None
        }
        if not nested_archives:
            _extract_bundle(outer, destination, countries)
            return

        missing = countries - nested_archives.keys()
        if missing:
            raise FileNotFoundError(
                f"Country subsets not found in archive: {', '.join(sorted(missing))}"
            )
        for country in sorted(countries):
            print(f"Extracting {country}...")
            with _stored_member_reader(
                source, nested_archives[country]
            ) as country_stream:
                with zipfile.ZipFile(country_stream) as country_bundle:
                    _extract_bundle(country_bundle, destination, {country})


def selective_remote_extract(url: str, destination: Path, countries: set[str]) -> None:
    """Extract selected country entries without downloading the other countries."""
    with HttpRangeReader(url) as remote:
        _extract_country_archives(remote, destination, countries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Data root",
    )
    parser.add_argument(
        "--archive", type=Path, help="Use an existing archive instead of downloading"
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        choices=COUNTRIES,
        default=list(DEFAULT_COUNTRIES),
        help="Countries/capture subsets to download; default: India",
    )
    parser.add_argument(
        "--no-extract", action="store_true", help="Download without extracting"
    )
    parser.add_argument(
        "--skip-checksum", action="store_true", help="Skip the 13 GB MD5 pass"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    archive = args.archive or (args.destination / ARCHIVE_NAME)
    countries = set(args.countries)
    selective = (
        args.archive is None and not args.no_extract and countries != set(COUNTRIES)
    )
    if selective:
        if shutil.disk_usage(args.destination).free < 2 * 1024**3:
            print(
                "Warning: less than 2 GiB is free for subset extraction.",
                file=sys.stderr,
            )
        print(
            "Selectively downloading entries for "
            f"{', '.join(args.countries)} from the official Figshare archive..."
        )
        selective_remote_extract(FIGSHARE_DOWNLOAD_URL, args.destination, countries)
        print(f"Extracted {', '.join(args.countries)} into {args.destination}")
        return
    if args.archive is None:
        free_bytes = shutil.disk_usage(args.destination.parent).free
        if free_bytes < 30 * 1024**3:
            print(
                "Warning: less than 30 GiB is free; download plus extraction may not fit.",
                file=sys.stderr,
            )
        archive_size = archive.stat().st_size if archive.exists() else 0
        if archive_size > FIGSHARE_SIZE_BYTES:
            raise RuntimeError(
                f"{archive} is larger than the publisher's file; move it aside and retry"
            )
        if archive_size < FIGSHARE_SIZE_BYTES:
            print(
                "Downloading the official Figshare archive (approximately 13.3 GB)..."
            )
            download_with_resume(FIGSHARE_DOWNLOAD_URL, archive)
        else:
            print(f"Using existing archive: {archive}")
        if archive.stat().st_size != FIGSHARE_SIZE_BYTES:
            raise RuntimeError(
                f"Incomplete download: expected {FIGSHARE_SIZE_BYTES} bytes, "
                f"got {archive.stat().st_size}"
            )
    if not args.skip_checksum:
        actual = md5sum(archive)
        if actual != FIGSHARE_MD5:
            raise RuntimeError(
                f"Checksum mismatch for {archive}: expected {FIGSHARE_MD5}, got {actual}"
            )
        print("Checksum verified.")
    if not args.no_extract:
        safe_extract(archive, args.destination, countries)
        print(f"Extracted {', '.join(args.countries)} into {args.destination}")


if __name__ == "__main__":
    main()
