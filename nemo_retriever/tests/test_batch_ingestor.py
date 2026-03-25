from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("ray")

from nemo_retriever.ingest_modes.batch import BatchIngestor


class _DummyClusterResources:
    def total_cpu_count(self) -> int:
        return 4

    def total_gpu_count(self) -> int:
        return 0

    def available_cpu_count(self) -> int:
        return 4

    def available_gpu_count(self) -> int:
        return 0


def test_batch_ingestor_filters_none_runtime_env_vars(monkeypatch) -> None:
    captured: dict[str, object] = {}
    dummy_ctx = SimpleNamespace(enable_rich_progress_bars=False, use_ray_tqdm=True)

    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.resolve_hf_cache_dir",
        lambda: "/tmp/hf-cache",
    )
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.ray.init",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.rd.DataContext.get_current",
        lambda: dummy_ctx,
    )
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.gather_cluster_resources",
        lambda _ray: _DummyClusterResources(),
    )
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.resolve_requested_plan",
        lambda cluster_resources: {"plan": "dummy"},
    )

    BatchIngestor(documents=[])

    assert captured["runtime_env"] == {
        "env_vars": {
            "LOG_LEVEL": "INFO",
            "NEMO_RETRIEVER_HF_CACHE_DIR": "/tmp/hf-cache",
        }
    }
    assert dummy_ctx.enable_rich_progress_bars is True
    assert dummy_ctx.use_ray_tqdm is False


def test_has_error_ignores_filename_keywords_in_metadata() -> None:
    payload = {
        "source_path": (
            "/tmp/Debugging_Unexpected_Errors_and_Exceptions_"
            "Google_IT_Automation_with_Python_Certificate.mp3"
        ),
        "segment_index": 0,
        "segment_count": 388,
        "segment_start": 7920,
        "segment_end": 15440,
    }

    assert BatchIngestor.has_error(payload) is False


def test_extract_error_rows_does_not_flag_paths_with_error_words() -> None:
    batch = pd.DataFrame(
        [
            {
                "metadata": {
                    "source_path": (
                        "/tmp/Debugging_Unexpected_Errors_and_Exceptions_"
                        "Google_IT_Automation_with_Python_Certificate.mp3"
                    )
                },
                "source": None,
                "embedding": [0.1, 0.2],
            },
            {
                "metadata": {"exception": "remote ASR timed out"},
                "source": None,
                "embedding": [0.3, 0.4],
            },
        ]
    )

    result = BatchIngestor.extract_error_rows(batch)

    assert len(result) == 1
    assert result.iloc[0]["metadata"]["exception"] == "remote ASR timed out"
