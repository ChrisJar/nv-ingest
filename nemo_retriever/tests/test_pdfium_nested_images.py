# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for raster images nested in PDF Form XObjects."""

from pathlib import Path

import pytest


def test_extract_nested_form_images_preserves_placements_and_page_coordinates() -> None:
    pdfium = pytest.importorskip("pypdfium2")
    from nemo_retriever.common.api.util.pdf.pdfium import extract_nested_simple_images_from_pdfium_page

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
