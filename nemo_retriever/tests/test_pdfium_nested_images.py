# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for raster images nested in PDF Form XObjects."""

from pathlib import Path

import pytest


def test_extract_nested_form_images_caches_sources_and_preserves_placements(monkeypatch) -> None:
    pdfium = pytest.importorskip("pypdfium2")
    from nemo_retriever.common.api.util.pdf.pdfium import extract_nested_simple_images_from_pdfium_page

    original_get_bitmap = pdfium.PdfImage.get_bitmap
    decode_count = 0

    def tracked_get_bitmap(self, *args, **kwargs):
        nonlocal decode_count
        decode_count += 1
        return original_get_bitmap(self, *args, **kwargs)

    monkeypatch.setattr(pdfium.PdfImage, "get_bitmap", tracked_get_bitmap)

    fixture = Path(__file__).resolve().parents[2] / "data" / "test-page-form.pdf"
    document = pdfium.PdfDocument(fixture)
    page = document[0]

    try:
        images = extract_nested_simple_images_from_pdfium_page(page)
    finally:
        page.close()
        document.close()

    assert len(images) == 18
    assert all(0 <= image.bbox[0] < image.bbox[2] <= image.max_width for image in images)
    assert all(0 <= image.bbox[1] < image.bbox[3] <= image.max_height for image in images)
    assert {(image.width, image.height) for image in images} == {(256, 256), (512, 512)}
    assert len({image.image for image in images}) == 5
    assert decode_count == 5

    payload_ids: dict[str, int] = {}
    for image in images:
        payload_ids.setdefault(image.image, id(image.image))
        assert id(image.image) == payload_ids[image.image]


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ({"max_images": 17}, "count exceeds"),
        ({"max_decoded_pixels": 1}, "decoded-pixel limit"),
        ({"max_source_bytes": 1}, "source is"),
    ],
)
def test_extract_nested_form_images_enforces_predecode_budgets(monkeypatch, limits, message) -> None:
    pdfium = pytest.importorskip("pypdfium2")
    from nemo_retriever.common.api.util.pdf.pdfium import extract_nested_simple_images_from_pdfium_page

    fixture = Path(__file__).resolve().parents[2] / "data" / "test-page-form.pdf"
    document = pdfium.PdfDocument(fixture)
    page = document[0]
    original_get_bitmap = pdfium.PdfImage.get_bitmap
    decode_count = 0

    def tracked_get_bitmap(self, *args, **kwargs):
        nonlocal decode_count
        decode_count += 1
        return original_get_bitmap(self, *args, **kwargs)

    monkeypatch.setattr(pdfium.PdfImage, "get_bitmap", tracked_get_bitmap)
    try:
        with pytest.raises(RuntimeError, match=message):
            extract_nested_simple_images_from_pdfium_page(page, **limits)
    finally:
        page.close()
        document.close()

    if "max_images" not in limits:
        assert decode_count == 0


@pytest.mark.parametrize(
    "position",
    [
        (-20.0, 10.0, -10.0, 20.0),
        (110.0, 10.0, 120.0, 20.0),
        (10.0, -20.0, 20.0, -10.0),
        (10.0, 110.0, 20.0, 120.0),
        (10.1, 10.1, 10.2, 10.2),
    ],
)
def test_visible_page_bbox_rejects_off_page_and_degenerate_boxes(position) -> None:
    from nemo_retriever.common.api.util.pdf.pdfium import _visible_page_bbox

    assert _visible_page_bbox(position, 100.0, 100.0) is None


def test_visible_page_bbox_clips_partial_intersection() -> None:
    from nemo_retriever.common.api.util.pdf.pdfium import _visible_page_bbox

    assert _visible_page_bbox((-10.0, 20.0, 30.0, 120.0), 100.0, 100.0) == [0, 0, 30, 80]


@pytest.mark.parametrize(
    "limits",
    [
        {"max_images": 0},
        {"max_decoded_pixels": 0},
        {"max_source_bytes": 0},
    ],
)
def test_extract_nested_form_images_rejects_nonpositive_budgets(limits) -> None:
    from nemo_retriever.common.api.util.pdf.pdfium import extract_nested_simple_images_from_pdfium_page

    with pytest.raises(ValueError, match="must all be positive"):
        extract_nested_simple_images_from_pdfium_page(object(), **limits)
