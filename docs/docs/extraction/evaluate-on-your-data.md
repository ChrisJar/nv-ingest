# Evaluate on your data

Retrieval and ingestion performance **depend on your documents**, hardware, and pipeline settings. Use the following when measuring quality and throughput on **your** datasets.

## Benchmarking and baselines { #benchmarking-and-baselines }

Use this page as the baseline for methodology and expectations. Use [Operational tuning](#operational-tuning) below to observe production-like runs.

### Match ViDoRe v3 document IDs

The ViDoRe v3 harness benchmarks use `corpus_id` as the BEIR document ID field. Use this setting when each downloaded PDF filename is the raw Hugging Face `corpus_id`, such as `157.pdf`.

During evaluation, the harness compares the query relevance labels against the PDF filename stem. This mapping preserves ViDoRe graded relevance scores. If you use PDFs with original document filenames instead, select a document ID field that matches those filenames and your relevance labels.

## Throughput and dataset effects { #throughput-and-dataset-effects }

Read [Throughput is dataset-dependent](multimodal-extraction.md#extraction-limitations-and-quality) for why raw numbers from generic benchmarks may not match your corpus (layout complexity, file types, image density, and so on).

## Operational tuning { #operational-tuning }

- [Ray and distributed ingest](ray-logging.md)
- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md) for supported configurations
- [Troubleshoot](troubleshoot.md) when results or performance diverge from expectations
