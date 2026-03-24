import json

import pandas as pd
import pytest

from nemo_retriever.recall.core import _audio_hit_to_window, _is_audio_hit_at_k, _normalize_query_df, is_hit_at_k


@pytest.mark.parametrize(
    "match_mode,df,expected",
    [
        ("pdf_only", pd.DataFrame({"query": ["q1"], "expected_pdf": ["Doc_A.pdf"]}), ["Doc_A"]),
        ("pdf_page", pd.DataFrame({"query": ["q1"], "pdf": ["Doc_A.pdf"], "page": [2]}), ["Doc_A_2"]),
        (
            "audio_time_window",
            pd.DataFrame({"question": ["q1"], "name": ["clip_a.mp3"], "start_time": [5.0], "end_time": [9.0]}),
            ["clip_a"],
        ),
    ],
)
def test_normalize_query_df_modes(match_mode: str, df: pd.DataFrame, expected: list[str]) -> None:
    out = _normalize_query_df(df, match_mode=match_mode)
    assert out["golden_answer"].tolist() == expected


@pytest.mark.parametrize(
    "match_mode,golden,retrieved,k,expected",
    [
        ("pdf_only", "Doc_A", ["Doc_A_7", "Doc_B_1"], 1, True),
        ("pdf_only", "Doc_B", ["Doc_A_7", "Doc_B_1"], 1, False),
        ("pdf_page", "Doc_A_2", ["Doc_A_2", "Doc_A_9"], 1, True),
        ("pdf_page", "Doc_A_2", ["Doc_A_9", "Doc_A_-1"], 1, False),
    ],
)
def test_is_hit_at_k_modes(match_mode: str, golden: str, retrieved: list[str], k: int, expected: bool) -> None:
    assert is_hit_at_k(golden, retrieved, k, match_mode=match_mode) is expected


def test_audio_hit_to_window_normalizes_source_and_times() -> None:
    hit = {
        "metadata": json.dumps({"segment_start": 12.0, "segment_end": 18.0}),
        "source": json.dumps({"source_id": "/tmp/audio/clip_a.mp3"}),
    }
    window = _audio_hit_to_window(hit)
    assert window is not None
    assert window["source"] == "clip_a"
    assert window["midpoint"] == 15.0


def test_audio_time_window_hit_matches_inside_tolerance() -> None:
    row = pd.Series({"expected_source": "clip_a", "start_time": 10.0, "end_time": 20.0})
    hits = [
        {
            "metadata": json.dumps({"segment_start": 12.0, "segment_end": 14.0}),
            "source": json.dumps({"source_id": "/tmp/audio/clip_a.mp3"}),
        }
    ]
    assert _is_audio_hit_at_k(row, hits, 1) is True


def test_audio_time_window_hit_respects_strict_boundary_check() -> None:
    row = pd.Series({"expected_source": "clip_a", "start_time": 10.0, "end_time": 20.0})
    hits = [
        {
            "metadata": json.dumps({"segment_start": 7.0, "segment_end": 9.0}),
            "source": json.dumps({"source_id": "/tmp/audio/clip_a.mp3"}),
        }
    ]
    assert _is_audio_hit_at_k(row, hits, 1) is False
