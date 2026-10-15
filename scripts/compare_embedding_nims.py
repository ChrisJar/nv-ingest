# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Compare user-supplied embedding NIM images without publishing image defaults."""

from __future__ import annotations

import argparse
import base64
from contextlib import redirect_stderr, redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import io
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import signal
import socket
from statistics import mean
import subprocess
import threading
import time
import tempfile
from typing import Any

import requests

VIDORE = (
    "vidore_v3_computer_science",
    "vidore_v3_energy",
    "vidore_v3_finance_en",
    "vidore_v3_finance_fr",
    "vidore_v3_hr",
    "vidore_v3_industrial",
    "vidore_v3_pharmaceuticals",
    "vidore_v3_physics",
)
RUNTIME = {
    "NIM_HTTP_API_PORT": "8000",
    "NIM_TRITON_LOG_VERBOSE": "1",
    "OMP_NUM_THREADS": "1",
    "NIM_ENGINE_COUNT": "1",
    "NIM_PERFORMANCE_MODE": "1",
}


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _jobs() -> list[tuple[str, str, str]]:
    return [
        (arm, dataset, modality)
        for arm in ("baseline", "candidate")
        for dataset in ("bo767", *VIDORE)
        for modality in (("text", "text_image") if arm == "candidate" and dataset != "bo767" else ("text",))
    ]


def _embeddings(
    endpoint: str, model: str, texts: list[str], input_type: str, modality: str = "text"
) -> list[list[float]]:
    response = requests.post(
        endpoint + "/v1/embeddings",
        json={
            "model": model,
            "input": texts,
            "input_type": input_type,
            "modality": modality,
            "encoding_format": "float",
            "truncate": "END",
        },
        timeout=600,
    )
    response.raise_for_status()
    items = sorted(response.json()["data"], key=lambda item: item["index"])
    if [item["index"] for item in items] != list(range(len(texts))):
        raise ValueError("Embedding response indices do not match the request")
    vectors = [item["embedding"] for item in items]
    if (
        not vectors
        or not vectors[0]
        or any(
            len(vector) != len(vectors[0]) or not all(math.isfinite(value) for value in vector) for vector in vectors
        )
    ):
        raise ValueError("Embedding response contains invalid vectors")
    return vectors


def _start(image: str, arm: str, args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("NGC_API_KEY"):
        raise ValueError("Set NGC_API_KEY in the environment before starting NIM containers")
    name = f"nrl-embed-comparison-{os.getpid()}-{arm}"
    if subprocess.run(["docker", "container", "inspect", name], capture_output=True).returncode == 0:
        raise ValueError(f"Refusing to replace existing container {name}")
    with socket.socket() as client:
        if client.connect_ex(("127.0.0.1", args.port)) == 0:
            raise ValueError(f"Port {args.port} is already in use")
    with (args.output_dir / f"{arm}_launch.log").open("a") as log:
        subprocess.run(["docker", "pull", image], stdout=log, stderr=subprocess.STDOUT, check=True)
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--gpus",
        f"device={args.gpu}",
        "--shm-size",
        "16g",
        "-p",
        f"127.0.0.1:{args.port}:8000",
        "-e",
        "NGC_API_KEY",
        "-v",
        f"{name}-cache:/opt/nim/.cache",
    ]
    for key, value in RUNTIME.items():
        command.extend(["-e", f"{key}={value}"])
    identity = subprocess.check_output([*command, image], text=True).strip()
    manifest = {
        "container": name,
        "container_id": identity,
        "image": image,
        "endpoint": f"http://127.0.0.1:{args.port}",
        "runtime_overrides": RUNTIME,
    }
    ready = False
    try:
        manifest["image_digests"] = json.loads(
            subprocess.check_output(
                ["docker", "image", "inspect", image, "--format", "{{json .RepoDigests}}"], text=True, timeout=30
            )
        )
        _dump(args.output_dir / f"{arm}_nim.json", manifest)
        ready = True
        return manifest
    finally:
        if not ready:
            _stop(manifest, args.output_dir / f"{arm}_container.log")


def _stop(manifest: dict[str, Any], output: Path) -> None:
    identity = subprocess.check_output(
        ["docker", "inspect", manifest["container"], "--format", "{{.Id}}"], text=True, timeout=10
    ).strip()
    if identity != manifest["container_id"]:
        raise RuntimeError("Refusing to stop a replaced container")
    try:
        with output.open("w") as log:
            subprocess.run(["docker", "logs", identity], stdout=log, stderr=subprocess.STDOUT, timeout=30)
    finally:
        subprocess.run(["docker", "stop", "--time", "30", identity], check=True, timeout=45)


def _probe(manifest: dict[str, Any], arm: str, output: Path) -> str:
    endpoint = manifest["endpoint"]
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        try:
            response = requests.get(endpoint + "/v1/models", timeout=5)
            if response.ok and response.json().get("data"):
                models = response.json()
                break
        except requests.RequestException:
            pass
        running = subprocess.check_output(
            ["docker", "inspect", manifest["container"], "--format", "{{.State.Running}}"], text=True
        ).strip()
        if running != "true":
            raise RuntimeError("NIM exited before becoming ready; inspect its local container log")
        time.sleep(5)
    else:
        raise TimeoutError("NIM readiness timed out")
    if len(models["data"]) != 1:
        raise ValueError("Expected exactly one served embedding model")
    model = models["data"][0]["id"]
    passage = _embeddings(endpoint, model, ["Example document."], "passage")
    query = _embeddings(endpoint, model, ["Example question?"], "query")
    if len(passage[0]) != len(query[0]):
        raise ValueError("Query and passage vector widths differ")
    if arm == "candidate":
        from PIL import Image

        buffer = io.BytesIO()
        with Image.new("RGB", (224, 224), "white") as image:
            image.save(buffer, format="PNG")
        pair = "Example page.\ndata:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        vectors = _embeddings(endpoint, model, [pair], "passage", "text_image")
        if len(vectors[0]) != len(query[0]):
            raise ValueError("Text and paired-image vector widths differ")
    _dump(output / f"{arm}_probe.json", {"model": model, "dimension": len(query[0]), "models": models})
    for route, label in (("/v1/metadata", "metadata"), ("/openapi.json", "openapi")):
        response = requests.get(endpoint + route, timeout=10)
        if response.ok:
            _dump(output / f"{arm}_{label}.json", response.json())
    return model


def _fixture(dataset: str, settings: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    target = args.output_dir / "config" / f"{dataset}_frozen.json"
    if target.exists():
        fixture = json.loads(target.read_text())
        _validate_fixture(fixture, dataset)
        return fixture
    if args.frozen_queries_dir:
        fixture = json.loads((args.frozen_queries_dir / f"{dataset}_frozen.json").read_text())
        # Copy only query text and judgments, never prior vectors or candidates.
        fixture = {"queries": fixture["queries"], "qrels": fixture["qrels"]}
    else:
        from transformers import AutoTokenizer
        from nemo_retriever.models.hf_model_registry import get_hf_revision
        from nemo_retriever.tools.recall.beir import load_beir_dataset

        model = "nvidia/Nemotron-3-Embed-1B-BF16"
        tokenizer = AutoTokenizer.from_pretrained(model, revision=get_hf_revision(model))
        data = load_beir_dataset(
            "bo767_csv" if dataset == "bo767" else "vidore_hf",
            dataset_name=settings["query_file"] if dataset == "bo767" else dataset,
            doc_id_field="pdf_page" if dataset == "bo767" else "corpus_id",
        )
        queries = []
        for query_id, text in zip(data.query_ids, data.queries):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > 128:
                text = tokenizer.decode(tokens[:128], skip_special_tokens=True)
            queries.append({"query_id": query_id, "query": text})
        fixture = {"queries": queries, "qrels": data.qrels}
    _validate_fixture(fixture, dataset)
    _dump(target, fixture)
    return fixture


def _validate_fixture(fixture: dict[str, Any], dataset: str) -> None:
    queries, qrels = fixture["queries"], fixture["qrels"]
    if not queries or not isinstance(qrels, dict) or not qrels:
        raise ValueError(f"Empty query/qrel fixture for {dataset}")
    identifiers = [record["query_id"] for record in queries]
    if any(not isinstance(identifier, str) for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Query IDs must be unique strings: {dataset}")
    if any(not isinstance(record["query"], str) or not record["query"].strip() for record in queries):
        raise ValueError(f"Query text must be nonempty strings: {dataset}")
    if any(not isinstance(qrels.get(identifier), dict) or not qrels[identifier] for identifier in identifiers):
        raise ValueError(f"Every query needs relevance judgments: {dataset}")


def _sample_memory(path: Path, stopped: threading.Event, gpu: int, status: dict[str, Any]) -> None:
    started = time.monotonic()
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp_utc", "monotonic_s", "memory_used_mib", "gpu_util_percent"])
        while not stopped.is_set():
            try:
                sample = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={gpu}",
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                memory, utilization = map(float, sample.strip().split(","))
                writer.writerow(
                    [datetime.now(timezone.utc).isoformat(), time.monotonic() - started, memory, utilization]
                )
                stream.flush()
                status["samples"] += 1
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError, ValueError) as error:
                status["errors"] += 1
                status["last_error"] = type(error).__name__
            stopped.wait(0.25)


def _ingest(
    folder: Path,
    dataset: str,
    modality: str,
    settings: dict[str, Any],
    model: str,
    endpoint: str,
    pdf_workers: int,
    embed_workers: int,
) -> dict[str, Any]:
    import lancedb
    import pypdfium2
    from nemo_retriever.common.params import BatchTuningParams, EmbedParams, ExtractParams, VdbUploadParams
    from nemo_retriever.cli.ingest_workflow import _strip_secret_values
    from nemo_retriever.ingestor import create_ingestor

    files = sorted(Path(settings["path"]).glob("*.pdf"))
    pages = 0
    for path in files:
        with pypdfium2.PdfDocument(path) as document:
            pages += len(document)
    tuning = BatchTuningParams(pdf_extract_workers=pdf_workers, embed_workers=embed_workers)
    extract = ExtractParams(
        method="pdfium_hybrid",
        dpi=200,
        extract_text=True,
        extract_images=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=dataset != "bo767",
        use_page_elements=True,
        use_table_structure=True,
        table_output_format="markdown",
        extract_page_as_image=modality == "text_image",
        ocr_version="v2",
        batch_tuning=tuning,
    )
    embed = EmbedParams(
        model_name=model,
        embed_model_name=model,
        embedding_endpoint=endpoint + "/v1/embeddings",
        api_key="",
        embed_modality=modality,
        embed_granularity="element" if dataset == "bo767" else "page",
        batch_tuning=tuning,
        nim_http_max_concurrent=32,
    )
    vdb = VdbUploadParams(
        vdb_kwargs={
            "uri": str(folder / "lancedb"),
            "table_name": "nemo-retriever",
            "overwrite": True,
            "hybrid": False,
            "embedding_model_name": model,
        }
    )
    _dump(
        folder / "settings.json",
        _strip_secret_values(
            {
                "files": list(map(str, files)),
                "extract": extract.model_dump(),
                "embed": embed.model_dump(),
                "model": model,
                "modality": modality,
                "granularity": embed.embed_granularity,
            }
        ),
    )
    started = time.perf_counter()
    create_ingestor(run_mode="batch").files(list(map(str, files))).extract(extract).embed(embed).vdb_upload(
        vdb
    ).ingest()
    seconds = time.perf_counter() - started
    table = lancedb.connect(str(folder / "lancedb")).open_table("nemo-retriever")
    result = {"pages": pages, "ingest_seconds": seconds, "indexed_rows": table.count_rows()}
    if not result["indexed_rows"]:
        raise ValueError("Ingestion produced an empty index")
    _dump(folder / "ingest.json", result)
    return result


def _evaluate(folder: Path, dataset: str, fixture: dict[str, Any], model: str, endpoint: str) -> dict[str, float]:
    import lancedb
    from nemo_retriever.tools.recall.beir import (
        _extract_page_number_from_hit,
        _extract_source_path_from_hit,
        build_beir_run_from_hits,
        compute_beir_metrics,
    )

    table = lancedb.connect(str(folder / "lancedb")).open_table("nemo-retriever")
    stored_model = table.schema.metadata[b"nemo_retriever.embedding_model_name"].decode()
    if stored_model != model:
        raise ValueError("Document/query embedding model identity mismatch")
    run = {}
    for offset in range(0, len(fixture["queries"]), 32):
        batch = fixture["queries"][offset : offset + 32]
        vectors = _embeddings(endpoint, model, [item["query"] for item in batch], "query")
        hits = []
        for vector in vectors:
            items = table.search(vector).limit(10).nprobes(64).refine_factor(10).to_list()
            if len(items) != min(10, table.count_rows()):
                raise ValueError("ANN search did not return the expected number of hits")
            if dataset == "bo767":
                for item in items:
                    source = _extract_source_path_from_hit(item)
                    page = _extract_page_number_from_hit(item)
                    if not source or page is None:
                        raise ValueError("BO767 hit has no source/page identity")
                    item["pdf_page"] = f"{Path(source).stem}_{page}"
            hits.append(items)
        run.update(
            build_beir_run_from_hits(
                [item["query_id"] for item in batch],
                hits,
                doc_id_field="pdf_page" if dataset == "bo767" else "pdf_basename",
            )
        )
    if len(run) != len(fixture["queries"]):
        raise ValueError("Evaluation did not produce a ranking for every query")
    metrics = compute_beir_metrics(fixture["qrels"], run, ks=(1, 3, 5, 10))
    _dump(folder / "retrieval_run.json", run)
    _dump(folder / "metrics.json", metrics)
    _dump(
        folder / "evaluation.json",
        {
            "model": model,
            "queries": len(run),
            "nprobes": 64,
            "refine_factor": 10,
            "query_fixture_sha256": hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest(),
        },
    )
    return metrics


def _summarize(output: Path) -> None:
    rows = []
    for arm in ("baseline", "candidate"):
        for dataset_kind, modality in (("bo767", "text"), ("vidore", "text"), ("vidore", "text_image")):
            if arm == "baseline" and modality == "text_image":
                continue
            folders = [
                output / "runs" / arm / modality / name for name in (("bo767",) if dataset_kind == "bo767" else VIDORE)
            ]
            records = [json.loads((folder / "result.json").read_text()) for folder in folders]
            pages = sum(record["pages"] for record in records)
            seconds = sum(record["ingest_seconds"] for record in records)
            rows.append(
                {
                    "arm": f"{arm}_{dataset_kind}_{modality}",
                    "pages": pages,
                    "ingest_secs": seconds,
                    "pages_per_sec": pages / seconds,
                    "ndcg_10_macro": mean(record["ndcg@10"] for record in records),
                    "recall_5_macro": mean(record["recall@5"] for record in records),
                    "recall_10_macro": mean(record["recall@10"] for record in records),
                    "gpu_peak_mib": (
                        max(record["gpu_peak_mib"] for record in records)
                        if all(record["gpu_peak_mib"] is not None for record in records)
                        else None
                    ),
                }
            )
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    indexed = {row["arm"]: row for row in rows}
    lines = [
        "# Embedding NIM comparison",
        "",
        "| Dataset / metric | Baseline text | Candidate | Change |",
        "|---|---:|---:|---:|",
    ]
    for dataset in ("bo767", "vidore"):
        for modality in ("text", "text_image") if dataset == "vidore" else ("text",):
            candidate = indexed[f"candidate_{dataset}_{modality}"]
            baseline = indexed[f"baseline_{dataset}_text"] if modality == "text" else None
            for key, label, smaller in (
                ("ndcg_10_macro", "nDCG@10", False),
                ("recall_5_macro", "Recall@5", False),
                ("pages_per_sec", "pages/sec", False),
                ("gpu_peak_mib", "GPU peak MiB", True),
            ):
                current = candidate[key]
                prior = baseline[key] if baseline else None
                left = "" if prior is None else f"{prior:.4f}"
                right = "" if current is None else f"{current:.4f}"
                change = "" if prior in (None, 0) or current is None else f"{100 * (current / prior - 1):+.2f}%"
                if prior is not None and current is not None and prior != current:
                    if (current < prior) == smaller:
                        right = f"**{right}**"
                    else:
                        left = f"**{left}**"
                lines.append(f"| {dataset} {modality} {label} | {left} | {right} | {change} |")
    lines.extend(
        [
            "",
            "Baseline text_image cells are blank because that arm is not run.",
            "ViDoRe quality is the unweighted eight-subset macro. "
            "Throughput includes fresh ingestion, not NIM startup or query evaluation.",
            "GPU memory is the sampled whole-device peak during ingestion, including local extraction. "
            "Missing or failed sampling remains blank.",
        ]
    )
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Run the sequential, full-pipeline NIM comparison CLI.

    Parameters
    ----------
    None
        Configuration is read from command-line arguments and image environment variables.

    Returns
    -------
    None
        Writes local configuration, query fixtures, logs, metrics, and summary reports.
        A dry run validates file layout without loading models or starting services.

    Raises
    ------
    ValueError
        If fixtures, model identities, or embedding responses are invalid.
    RuntimeError
        If ingestion, evaluation, monitoring, or container lifecycle checks fail.
    TimeoutError
        If a NIM does not become ready. CLI argument errors exit through argparse.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/nim_comparison"))
    parser.add_argument("--baseline-image", default=os.environ.get("BASELINE_NIM_IMAGE"))
    parser.add_argument("--candidate-image", default=os.environ.get("CANDIDATE_NIM_IMAGE"))
    parser.add_argument("--frozen-queries-dir", type=Path)
    parser.add_argument("--port", type=int, default=int(os.environ.get("NIM_PORT", "18016")))
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("NIM_GPU_DEVICE", "0")))
    parser.add_argument("--pdf-extract-workers", type=int, default=4)
    parser.add_argument("--embed-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.port < 65536 or args.gpu < 0:
        parser.error("Invalid port or GPU device")
    if not args.baseline_image or not args.candidate_image:
        parser.error("Supply --baseline-image/--candidate-image or BASELINE_NIM_IMAGE/CANDIDATE_NIM_IMAGE")
    if min(args.pdf_extract_workers, args.embed_workers) < 1:
        parser.error("Worker counts must be positive")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    if args.frozen_queries_dir:
        args.frozen_queries_dir = args.frozen_queries_dir.expanduser().resolve()
    settings = {
        "bo767": {"path": str(args.data_root / "bo767"), "query_file": str(args.data_root / "bo767_query_gt.csv")}
    }
    settings.update({name: {"path": str(args.data_root / "vidore_v3_corpus_pdf" / name)} for name in VIDORE})
    for dataset in ("bo767", *VIDORE):
        if dataset not in settings or not list(Path(settings[dataset]["path"]).glob("*.pdf")):
            parser.error(f"Missing PDF corpus: {dataset}")
        if args.frozen_queries_dir and not (args.frozen_queries_dir / f"{dataset}_frozen.json").is_file():
            parser.error(f"Missing frozen query/qrel fixture: {dataset}")
    if not Path(settings["bo767"]["query_file"]).is_file():
        parser.error("Missing BO767 query CSV")
    if any(
        (args.output_dir / name).exists() for name in ("runs", "baseline_nim.json", "candidate_nim.json", "summary.csv")
    ):
        parser.error("Refusing to overwrite a prior run; choose a new output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _dump(
        args.output_dir / "config/nim_plan.json",
        {
            "jobs": _jobs(),
            "runtime_overrides": RUNTIME,
            "dry_run": args.dry_run,
            "data": settings,
            "pdf_extract_workers": args.pdf_extract_workers,
            "embed_workers": args.embed_workers,
            "nim_http_max_concurrent": 32,
            "query_and_document_model": "discovered from each NIM /v1/models response",
            "query_policy": "reuse provided fixtures, otherwise freeze public baseline-tokenizer 128-token queries",
            "retrieval": "dense L2 top10, nprobes64, refine_factor10, no reranking",
        },
    )
    if args.dry_run:
        print("NIM configuration validated: five result groups, 26 dataset/modality jobs; no services started.")
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    temporary = args.output_dir / ".tmp"
    temporary.mkdir(mode=0o700, exist_ok=True)
    os.environ["TMPDIR"] = os.environ["RAY_TMPDIR"] = str(temporary)
    tempfile.tempdir = str(temporary)
    _dump(
        args.output_dir / "config/versions.json",
        {
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("nemo-retriever", "torch", "transformers", "ray", "lancedb")
            },
            "python": os.sys.version,
        },
    )
    for fixture_dataset in ("bo767", *VIDORE):
        _fixture(fixture_dataset, settings[fixture_dataset], args)
    for arm, image in (("baseline", args.baseline_image), ("candidate", args.candidate_image)):
        manifest = _start(image, arm, args)
        try:
            model = _probe(manifest, arm, args.output_dir)
            for job_arm, dataset, modality in _jobs():
                if job_arm != arm:
                    continue
                folder = args.output_dir / "runs" / arm / modality / dataset
                folder.mkdir(parents=True, exist_ok=False)
                print(f"Starting {arm}: {dataset}/{modality}", flush=True)
                stopped = threading.Event()
                status = {"samples": 0, "errors": 0}
                monitor = threading.Thread(
                    target=_sample_memory, args=(folder / "gpu_memory.csv", stopped, args.gpu, status), daemon=True
                )
                monitor.start()
                try:
                    with (folder / "ingest.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
                        result = _ingest(
                            folder,
                            dataset,
                            modality,
                            settings[dataset],
                            model,
                            manifest["endpoint"],
                            args.pdf_extract_workers,
                            args.embed_workers,
                        )
                finally:
                    stopped.set()
                    monitor.join(timeout=7)
                    status["thread_stopped"] = not monitor.is_alive()
                    _dump(folder / "gpu_sampling.json", status)
                if monitor.is_alive():
                    raise RuntimeError("GPU sampler did not stop within its bounded shutdown window")
                with (folder / "gpu_memory.csv").open() as stream:
                    samples = list(csv.DictReader(stream))
                result["gpu_peak_mib"] = (
                    max((float(row["memory_used_mib"]) for row in samples), default=None)
                    if not status["errors"]
                    else None
                )
                if status["errors"] or not samples:
                    logging.warning(
                        "GPU sampling incomplete for %s/%s/%s; memory report left blank", arm, dataset, modality
                    )
                with (folder / "evaluation.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
                    result.update(
                        _evaluate(
                            folder, dataset, _fixture(dataset, settings[dataset], args), model, manifest["endpoint"]
                        )
                    )
                _dump(folder / "result.json", result)
                print(f"Completed {arm}: {dataset}/{modality}", flush=True)
        finally:
            _stop(manifest, args.output_dir / f"{arm}_container.log")
    _summarize(args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def _interrupted(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _interrupted)
    try:
        main()
    except Exception:
        logging.exception("Embedding NIM comparison failed")
        raise SystemExit(1)
