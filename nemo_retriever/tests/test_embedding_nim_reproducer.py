# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline checks for the standalone, image-neutral NIM comparison script."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest


@pytest.fixture
def runner():
    script = Path(__file__).resolve().parents[2] / "scripts/compare_embedding_nims.py"
    spec = importlib.util.spec_from_file_location("embedding_nim_reproducer", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _data(root, runner):
    for dataset in ("bo767", *runner.VIDORE):
        folder = root / ("bo767" if dataset == "bo767" else f"vidore_v3_corpus_pdf/{dataset}")
        folder.mkdir(parents=True)
        (folder / "1.pdf").touch()
    (root / "bo767_query_gt.csv").touch()


def test_jobs_cover_all_five_comparison_groups(runner):
    jobs = runner._jobs()
    assert len(jobs) == 26
    assert sum(arm == "baseline" for arm, _, _ in jobs) == 9
    assert sum(modality == "text_image" for _, _, modality in jobs) == 8
    assert all(arm == "candidate" and dataset in runner.VIDORE for arm, dataset, mode in jobs if mode == "text_image")


def test_dry_run_does_not_start_services_or_load_fixtures(runner, tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    _data(data, runner)
    output = tmp_path / "report"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare",
            "--data-root",
            str(data),
            "--output-dir",
            str(output),
            "--baseline-image",
            "example/baseline",
            "--candidate-image",
            "example/candidate",
            "--dry-run",
        ],
    )
    monkeypatch.setattr(runner, "_start", Mock(side_effect=AssertionError("Docker called")))
    monkeypatch.setattr(runner, "_fixture", Mock(side_effect=AssertionError("Query download called")))
    runner.main()
    plan = json.loads((output / "config/nim_plan.json").read_text())
    assert len(plan["jobs"]) == 26
    assert plan["pdf_extract_workers"] == 4
    assert plan["embed_workers"] == 1
    assert "no services started" in capsys.readouterr().out
    assert "example/candidate" not in (output / "config/nim_plan.json").read_text()


def test_embeddings_use_same_model_and_correct_query_passage_route(runner, monkeypatch):
    response = Mock()
    response.json.return_value = {"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]}
    post = Mock(return_value=response)
    monkeypatch.setattr(runner.requests, "post", post)
    for input_type in ("passage", "query"):
        assert runner._embeddings("http://example", "example/model", ["first", "second"], input_type) == [[1.0], [2.0]]
        payload = post.call_args.kwargs["json"]
        assert payload["model"] == "example/model"
        assert payload["input_type"] == input_type
        assert payload["truncate"] == "END"


@pytest.mark.parametrize("items", [[{"index": 1, "embedding": [1.0]}], [{"index": 0, "embedding": [float("nan")]}]])
def test_embeddings_reject_incomplete_or_nonfinite_results(runner, monkeypatch, items):
    response = Mock()
    response.json.return_value = {"data": items}
    monkeypatch.setattr(runner.requests, "post", Mock(return_value=response))
    with pytest.raises(ValueError):
        runner._embeddings("http://example", "example/model", ["first"], "query")


def test_fixture_reuses_only_queries_and_judgments(runner, tmp_path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    fixture = {
        "queries": [{"query_id": "q", "query": "Exact query "}],
        "qrels": {"q": {"doc": 1}},
        "candidates": ["must not reuse"],
        "vectors": [1],
    }
    (fixtures / "bo767_frozen.json").write_text(json.dumps(fixture))
    result = runner._fixture(
        "bo767", {}, argparse.Namespace(output_dir=tmp_path / "output", frozen_queries_dir=fixtures)
    )
    assert result == {key: fixture[key] for key in ("queries", "qrels")}


@pytest.mark.parametrize(
    "fixture",
    [
        {"queries": [], "qrels": {}},
        {"queries": [{"query_id": "q", "query": "text"}], "qrels": {"other": {"doc": 1}}},
        {"queries": [{"query_id": "q", "query": ""}], "qrels": {"q": {"doc": 1}}},
        {"queries": [{"query_id": "q", "query": "text"}] * 2, "qrels": {"q": {"doc": 1}}},
    ],
)
def test_fixture_rejects_invalid_query_mapping(runner, fixture):
    with pytest.raises(ValueError):
        runner._validate_fixture(fixture, "example")


def test_sampler_timeout_is_bounded_and_recorded(runner, tmp_path, monkeypatch):
    stopped = threading.Event()
    status = {"samples": 0, "errors": 0}

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 5
        stopped.set()
        raise subprocess.TimeoutExpired(args[0], 5)

    monkeypatch.setattr(runner.subprocess, "check_output", timeout)
    runner._sample_memory(tmp_path / "gpu.csv", stopped, 0, status)
    assert status == {"samples": 0, "errors": 1, "last_error": "TimeoutExpired"}


def test_start_failure_stops_owned_container(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("NGC_API_KEY", "test-credential-never-passed-in-command")
    monkeypatch.setattr(runner.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=1)))
    monkeypatch.setattr(runner.subprocess, "check_output", Mock(side_effect=["owned-id\n", OSError("inspect failed")]))
    client = MagicMock()
    client.__enter__.return_value.connect_ex.return_value = 1
    monkeypatch.setattr(runner.socket, "socket", Mock(return_value=client))
    stop = Mock()
    monkeypatch.setattr(runner, "_stop", stop)
    with pytest.raises(OSError, match="inspect failed"):
        runner._start("example/image", "candidate", argparse.Namespace(output_dir=tmp_path, gpu=0, port=18016))
    assert stop.call_args.args[0]["container_id"] == "owned-id"
    launch = runner.subprocess.check_output.call_args_list[0].args[0]
    assert "NGC_API_KEY" in launch
    assert "test-credential-never-passed-in-command" not in launch


def test_stop_refuses_replaced_container(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "check_output", Mock(return_value="someone-else\n"))
    run = Mock()
    monkeypatch.setattr(runner.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="replaced"):
        runner._stop({"container": "owned", "container_id": "expected"}, tmp_path / "container.log")
    run.assert_not_called()


def test_stop_still_stops_if_log_capture_fails(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "check_output", Mock(return_value="owned-id\n"))
    run = Mock(side_effect=[subprocess.TimeoutExpired("logs", 30), SimpleNamespace(returncode=0)])
    monkeypatch.setattr(runner.subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        runner._stop({"container": "owned", "container_id": "owned-id"}, tmp_path / "container.log")
    assert run.call_args.args[0] == ["docker", "stop", "--time", "30", "owned-id"]


@pytest.mark.parametrize(
    "dataset,modality", [("bo767", "text"), ("vidore_v3_energy", "text"), ("vidore_v3_energy", "text_image")]
)
def test_ingest_applies_matched_pipeline_settings(runner, tmp_path, monkeypatch, dataset, modality):
    import lancedb
    import pypdfium2
    import nemo_retriever.ingestor

    folder = tmp_path / "run"
    data = tmp_path / "pdfs"
    data.mkdir()
    (data / "1.pdf").touch()
    document = MagicMock()
    document.__enter__.return_value.__len__.return_value = 3
    monkeypatch.setattr(pypdfium2, "PdfDocument", Mock(return_value=document))
    pipeline = MagicMock()
    for method in ("files", "extract", "embed", "vdb_upload"):
        getattr(pipeline, method).return_value = pipeline
    monkeypatch.setattr(nemo_retriever.ingestor, "create_ingestor", Mock(return_value=pipeline))
    connection = Mock()
    connection.open_table.return_value.count_rows.return_value = 3
    monkeypatch.setattr(lancedb, "connect", Mock(return_value=connection))
    result = runner._ingest(folder, dataset, modality, {"path": str(data)}, "example/model", "http://example", 4, 1)
    extract = pipeline.extract.call_args.args[0]
    embed = pipeline.embed.call_args.args[0]
    assert result["pages"] == 3
    assert extract.use_table_structure and extract.extract_charts and extract.extract_tables
    assert extract.extract_infographics == (dataset != "bo767")
    assert extract.extract_page_as_image == (modality == "text_image")
    assert extract.ocr_version == "v2"
    assert extract.batch_tuning.pdf_extract_workers == 4
    assert embed.batch_tuning.embed_workers == 1
    assert embed.model_name == embed.embed_model_name == "example/model"
    assert embed.embedding_endpoint == "http://example/v1/embeddings"
    assert embed.embed_granularity == ("element" if dataset == "bo767" else "page")


@pytest.mark.parametrize("fail_ingest", [False, True])
def test_main_runs_sequentially_and_leaves_baseline_image_cells_blank(runner, tmp_path, monkeypatch, fail_ingest):
    data = tmp_path / "data"
    _data(data, runner)
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare",
            "--data-root",
            str(data),
            "--output-dir",
            str(output),
            "--baseline-image",
            "example/baseline",
            "--candidate-image",
            "example/candidate",
        ],
    )
    for key in ("CUDA_VISIBLE_DEVICES", "TMPDIR", "RAY_TMPDIR"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(runner.tempfile, "tempdir", None)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: "test-version")
    monkeypatch.setattr(
        runner, "_fixture", lambda *args: {"queries": [{"query_id": "q", "query": "query"}], "qrels": {"q": {"doc": 1}}}
    )
    lifecycle = []

    def start(image, arm, args):
        lifecycle.append(("start", arm))
        return {"arm": arm, "endpoint": "http://" + arm}

    monkeypatch.setattr(runner, "_start", start)
    monkeypatch.setattr(runner, "_stop", lambda manifest, output: lifecycle.append(("stop", manifest["arm"])))
    monkeypatch.setattr(runner, "_probe", lambda manifest, arm, output: arm + "-model")
    monkeypatch.setattr(runner, "_ingest", lambda *args: {"pages": 1, "ingest_seconds": 2, "indexed_rows": 1})

    def evaluate(folder, dataset, fixture, model, endpoint):
        assert model == endpoint.removeprefix("http://") + "-model"
        return {"ndcg@10": 0.5, "recall@5": 0.6, "recall@10": 0.7}

    monkeypatch.setattr(runner, "_evaluate", evaluate)

    def sample(path, stopped, gpu, status):
        path.write_text("timestamp_utc,memory_used_mib,gpu_util_percent\nnow,100,50\n")
        status["samples"] = 1

    monkeypatch.setattr(runner, "_sample_memory", sample)
    if fail_ingest:
        monkeypatch.setattr(runner, "_ingest", Mock(side_effect=ValueError("bad ingestion")))
        with pytest.raises(ValueError, match="bad ingestion"):
            runner.main()
        assert lifecycle == [("start", "baseline"), ("stop", "baseline")]
        return
    runner.main()
    assert lifecycle == [("start", "baseline"), ("stop", "baseline"), ("start", "candidate"), ("stop", "candidate")]
    assert len(list((output / "runs").glob("*/*/*/result.json"))) == 26
    report = (output / "comparison.md").read_text()
    assert "| vidore text_image nDCG@10 |  | 0.5000 |  |" in report
    assert len((output / "summary.csv").read_text().splitlines()) == 6


@pytest.mark.parametrize("mismatched_model", [False, True])
def test_evaluation_checks_identity_and_maps_bo767_pages(runner, tmp_path, monkeypatch, mismatched_model):
    import lancedb
    import nemo_retriever.tools.recall.beir as beir

    table = Mock()
    table.schema.metadata = {
        b"nemo_retriever.embedding_model_name": b"other/model" if mismatched_model else b"example/model"
    }
    table.count_rows.return_value = 1
    search = table.search.return_value
    search.limit.return_value = search
    search.nprobes.return_value = search
    search.refine_factor.return_value = search
    search.to_list.return_value = [{"source": "/corpus/document.pdf", "page": 3}]
    connection = Mock()
    connection.open_table.return_value = table
    monkeypatch.setattr(lancedb, "connect", Mock(return_value=connection))
    embeddings = Mock(return_value=[[1.0, 2.0]])
    monkeypatch.setattr(runner, "_embeddings", embeddings)
    fixture = {"queries": [{"query_id": "q", "query": "query text"}], "qrels": {"q": {"document_3": 1}}}
    if mismatched_model:
        with pytest.raises(ValueError, match="identity mismatch"):
            runner._evaluate(tmp_path, "bo767", fixture, "example/model", "http://example")
        embeddings.assert_not_called()
        return
    monkeypatch.setattr(beir, "_extract_source_path_from_hit", lambda hit: hit["source"])
    monkeypatch.setattr(beir, "_extract_page_number_from_hit", lambda hit: hit["page"])

    def build_run(query_ids, hits, doc_id_field):
        assert query_ids == ["q"]
        assert doc_id_field == "pdf_page"
        assert hits[0][0]["pdf_page"] == "document_3"
        return {"q": {"document_3": 1.0}}

    monkeypatch.setattr(beir, "build_beir_run_from_hits", build_run)
    compute = Mock(return_value={"ndcg@10": 1.0, "recall@5": 1.0})
    monkeypatch.setattr(beir, "compute_beir_metrics", compute)
    runner._evaluate(tmp_path, "bo767", fixture, "example/model", "http://example")
    embeddings.assert_called_once_with("http://example", "example/model", ["query text"], "query")
    search.nprobes.assert_called_once_with(64)
    search.refine_factor.assert_called_once_with(10)
    assert json.loads((tmp_path / "metrics.json").read_text())["ndcg@10"] == 1.0
