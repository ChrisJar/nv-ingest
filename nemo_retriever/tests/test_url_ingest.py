# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import httpx
import pandas as pd
import pytest

from nemo_retriever.common.params import UrlFetchParams
from nemo_retriever.common.url_fetch import FetchedUrl, UrlFetchFailure, fetch_urls, normalize_urls
from nemo_retriever.ingestor.graph_ingestor import GraphIngestionError, GraphIngestor
from nemo_retriever.service.client import InMemoryUpload
from nemo_retriever.service.service_ingestor import ServiceIngestor
from nemo_retriever.service.services.pipeline_executor import _merge_document_metadata


PDF_URL = "https://example.test/document"
PDF_BYTES = b"%PDF-1.7\n"


def test_url_fetch_params_defaults() -> None:
    params = UrlFetchParams()

    assert params.request_timeout_s == 30.0
    assert params.follow_redirects is True
    assert params.max_response_bytes == 10_000_000
    assert params.max_concurrency == 8


@pytest.mark.parametrize("value", ["relative/path.pdf", "file:///tmp/a.pdf", "", "   "])
def test_normalize_urls_rejects_non_http_sources(value: str) -> None:
    with pytest.raises(ValueError, match=r"HTTP\(S\)|nonempty"):
        normalize_urls(value)


def test_fetch_urls_dispatches_extensionless_pdf_from_content_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=PDF_BYTES)

    # Exercise the classifier without relying on external networking while
    # retaining httpx's real streaming response behavior.
    from nemo_retriever.common import url_fetch

    with httpx.Client(transport=httpx.MockTransport(handler), headers={"Authorization": "Bearer secret"}) as client:
        outcome = url_fetch._fetch_one(client, PDF_URL, 0, UrlFetchParams(headers={"Authorization": "Bearer secret"}))

    assert isinstance(outcome, FetchedUrl)
    assert outcome.input_type == "pdf"
    assert outcome.classification_filename.endswith(".pdf")
    assert outcome.content == PDF_BYTES


def test_fetch_urls_collects_http_and_size_failures(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/missing":
            return httpx.Response(404, content=b"missing")
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"x" * 11)

    real_client = httpx.Client

    def client_factory(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(**kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    fetched, failures = fetch_urls(
        ["https://example.test/missing", "https://example.test/large"],
        UrlFetchParams(max_response_bytes=10),
    )

    assert fetched == []
    assert [failure.url for failure in failures] == [
        "https://example.test/missing",
        "https://example.test/large",
    ]
    assert "404" in failures[0].message
    assert "max_response_bytes=10" in failures[1].message


def test_graph_ingestor_returns_url_fetch_failures(monkeypatch) -> None:
    failure = UrlFetchFailure(PDF_URL, "HTTPStatusError", "HTTP 404")
    monkeypatch.setattr(
        "nemo_retriever.ingestor.graph_ingestor.fetch_urls",
        lambda urls, params: ([], [failure]),
    )

    result, failures = GraphIngestor().urls(PDF_URL).extract().ingest(return_failures=True)

    assert result.empty
    assert failures == [(PDF_URL, "HTTP 404")]


def test_graph_ingestor_raises_url_fetch_failures_by_default(monkeypatch) -> None:
    failure = UrlFetchFailure(PDF_URL, "HTTPStatusError", "HTTP 500")
    monkeypatch.setattr(
        "nemo_retriever.ingestor.graph_ingestor.fetch_urls",
        lambda urls, params: ([], [failure]),
    )

    with pytest.raises(GraphIngestionError, match="HTTP 500"):
        GraphIngestor().urls(PDF_URL).extract().ingest()


def test_service_collect_inputs_builds_url_upload(monkeypatch) -> None:
    fetched = FetchedUrl(
        url=PDF_URL,
        content=PDF_BYTES,
        content_type="application/pdf",
        classification_filename="url-00000000.pdf",
        input_type="pdf",
        transport_path="url-source://00000000/url-00000000.pdf",
    )
    monkeypatch.setattr(
        "nemo_retriever.service.service_ingestor.fetch_urls",
        lambda urls, params: ([fetched], []),
    )

    inputs = ServiceIngestor().urls(PDF_URL)._collect_inputs()

    assert len(inputs) == 1
    upload = inputs[0]
    assert isinstance(upload, InMemoryUpload)
    assert upload.classification_filename == "url-00000000.pdf"
    assert upload.metadata == {"_nrl_source_url": PDF_URL}


def test_service_metadata_restores_original_url() -> None:
    result = pd.DataFrame(
        [
            {
                "path": "url-source://00000000/url-00000000.pdf",
                "metadata": {
                    "source_path": "url-source://00000000/url-00000000.pdf",
                    "content_metadata": {"type": "text"},
                },
            }
        ]
    )

    _merge_document_metadata(result, {"_nrl_source_url": PDF_URL, "tenant": "test"})

    assert result.iloc[0]["path"] == PDF_URL
    assert result.iloc[0]["metadata"]["source_path"] == PDF_URL
    assert result.iloc[0]["metadata"]["content_metadata"]["tenant"] == "test"
    assert "_nrl_source_url" not in result.iloc[0]["metadata"]["content_metadata"]


def test_service_ingest_returns_fetch_failure_without_contacting_service(monkeypatch) -> None:
    failure = UrlFetchFailure(PDF_URL, "HTTPStatusError", "HTTP 401")
    monkeypatch.setattr(
        "nemo_retriever.service.service_ingestor.fetch_urls",
        lambda urls, params: ([], [failure]),
    )

    result, failures = ServiceIngestor(base_url="https://service.invalid").urls(PDF_URL).ingest(return_failures=True)

    assert result.document_ids == []
    assert failures[0][0] == PDF_URL
    assert "HTTP 401" in failures[0][1]
