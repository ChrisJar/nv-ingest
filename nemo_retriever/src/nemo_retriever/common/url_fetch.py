# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded HTTP fetching and format classification for ingest URL sources."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from email.message import Message
from pathlib import PurePosixPath
from typing import Sequence
from urllib.parse import unquote, urlparse

import httpx

from nemo_retriever.common.input_files import AUTO_INPUT_EXTENSIONS, input_type_for_path
from nemo_retriever.common.params import UrlFetchParams


_MIME_DEFAULT_EXTENSIONS: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/json": ".json",
    "application/x-sh": ".sh",
    "text/x-shellscript": ".sh",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
    "image/svg+xml": ".svg",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mp4": ".m4a",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "video/x-msvideo": ".avi",
}
_GENERIC_MIME_TYPES = {"", "application/octet-stream", "binary/octet-stream"}


@dataclass(frozen=True)
class FetchedUrl:
    url: str
    content: bytes
    content_type: str
    classification_filename: str
    input_type: str
    transport_path: str


@dataclass(frozen=True)
class UrlFetchFailure:
    url: str
    error_type: str
    message: str

    def as_record(self) -> dict[str, object]:
        return {
            "row_index": None,
            "source_identifier": self.url,
            "column": "url_fetch",
            "path": "error",
            "error": {
                "stage": "url_fetch",
                "type": self.error_type,
                "message": self.message,
            },
        }


def normalize_urls(urls: str | Sequence[str]) -> list[str]:
    values = [urls] if isinstance(urls, str) else list(urls)
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("urls() entries must be nonempty strings")
        url = value.strip()
        parsed = urlparse(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"urls() requires an absolute HTTP(S) URL; got {value!r}")
        normalized.append(url)
    return normalized


def _content_disposition_filename(header: str) -> str:
    if not header:
        return ""
    message = Message()
    message["content-disposition"] = header
    return message.get_filename() or ""


def _supported_suffix(name: str) -> str:
    suffix = PurePosixPath(unquote(urlparse(name).path)).suffix.lower()
    return suffix if suffix in AUTO_INPUT_EXTENSIONS else ""


def _classify_response(url: str, response: httpx.Response, position: int) -> tuple[str, str, str, str]:
    content_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
    disposition_name = _content_disposition_filename(response.headers.get("content-disposition", ""))
    hinted_suffix = _supported_suffix(disposition_name) or _supported_suffix(url)
    mime_suffix = _MIME_DEFAULT_EXTENSIONS.get(content_type, "")

    if mime_suffix:
        suffix = hinted_suffix
        if not suffix or input_type_for_path(f"source{suffix}") != input_type_for_path(f"source{mime_suffix}"):
            suffix = mime_suffix
    elif content_type in _GENERIC_MIME_TYPES and hinted_suffix:
        suffix = hinted_suffix
    elif hinted_suffix:
        suffix = hinted_suffix
    else:
        displayed = content_type or "missing Content-Type"
        raise ValueError(f"unsupported response format ({displayed})")

    input_type = input_type_for_path(f"source{suffix}")
    if input_type is None:
        raise ValueError(f"unsupported response format ({content_type or suffix})")
    filename = f"url-{position:08d}{suffix}"
    transport_path = f"url-source://{position:08d}/{filename}"
    return content_type or "application/octet-stream", filename, input_type, transport_path


def _fetch_one(client: httpx.Client, url: str, position: int, params: UrlFetchParams) -> FetchedUrl | UrlFetchFailure:
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            if content_length is not None and int(content_length) > params.max_response_bytes:
                raise ValueError(
                    f"response exceeds max_response_bytes={params.max_response_bytes} "
                    f"(Content-Length={content_length})"
                )
            content_type, filename, input_type, transport_path = _classify_response(url, response, position)
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > params.max_response_bytes:
                    raise ValueError(f"response exceeds max_response_bytes={params.max_response_bytes}")
                chunks.append(chunk)
        return FetchedUrl(url, b"".join(chunks), content_type, filename, input_type, transport_path)
    except Exception as exc:  # noqa: BLE001 - every source failure must remain isolated.
        return UrlFetchFailure(url=url, error_type=type(exc).__name__, message=str(exc))


def fetch_urls(urls: Sequence[str], params: UrlFetchParams) -> tuple[list[FetchedUrl], list[UrlFetchFailure]]:
    """Fetch URL sources concurrently while preserving caller order."""

    if not urls:
        return [], []
    timeout = httpx.Timeout(params.request_timeout_s)
    limits = httpx.Limits(max_connections=params.max_concurrency, max_keepalive_connections=params.max_concurrency)
    with httpx.Client(
        headers=params.headers,
        timeout=timeout,
        follow_redirects=params.follow_redirects,
        limits=limits,
    ) as client:
        with ThreadPoolExecutor(max_workers=params.max_concurrency, thread_name_prefix="nrl-url-fetch") as executor:
            outcomes = list(executor.map(lambda item: _fetch_one(client, item[1], item[0], params), enumerate(urls)))

    fetched = [outcome for outcome in outcomes if isinstance(outcome, FetchedUrl)]
    failures = [outcome for outcome in outcomes if isinstance(outcome, UrlFetchFailure)]
    return fetched, failures
