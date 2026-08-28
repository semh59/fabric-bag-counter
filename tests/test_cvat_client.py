"""Tests for CvatClient's real CVAT REST API integration (§6.3, §6.4, §9.5).

No existing test file covered this module at all before now -- these tests
exercise the real httpx request/response path via httpx.MockTransport
(a fake transport, not a mocked method), so a broken URL, header, or
payload shape would actually fail these tests the same way it would fail
against a real CVAT server.
"""

import json

import httpx
import numpy as np
import pytest

from packages.cs_data.cvat_client import CvatApiError, CvatClient


def test_get_labels_spec_matches_annotation_guide_two_classes():
    spec = CvatClient().get_labels_spec()
    names = {label["name"] for label in spec}
    assert names == {"bag_body", "print_mark"}

    bag_body = next(label for label in spec if label["name"] == "bag_body")
    assert bag_body["type"] == "polygon"
    print_mark = next(label for label in spec if label["name"] == "print_mark")
    assert print_mark["type"] == "rectangle"


def test_create_task_posts_real_label_spec_and_returns_task_id():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": 42, "name": "Line 1 - Batch 3"})

    client = CvatClient(
        base_url="http://cvat.local/api", auth_token="secrettoken",
        transport=httpx.MockTransport(handler),
    )
    result = client.create_task("Line 1 - Batch 3", project_id=7)

    assert result == {"id": 42, "name": "Line 1 - Batch 3"}
    assert captured["method"] == "POST"
    assert captured["url"] == "http://cvat.local/api/tasks"
    assert captured["headers"]["authorization"] == "Token secrettoken"
    assert captured["body"]["name"] == "Line 1 - Batch 3"
    assert captured["body"]["project_id"] == 7
    label_names = {label["name"] for label in captured["body"]["labels"]}
    assert label_names == {"bag_body", "print_mark"}


def test_create_task_raises_cvat_api_error_on_non_2xx():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Unauthorized")

    client = CvatClient(transport=httpx.MockTransport(handler))
    with pytest.raises(CvatApiError, match="401"):
        client.create_task("Some Task")


def test_upload_task_data_sends_real_multipart_files(tmp_path):
    img1 = tmp_path / "frame_0001.jpg"
    img1.write_bytes(b"\xff\xd8\xff\xe0FAKEJPEGBYTES")
    img2 = tmp_path / "frame_0002.jpg"
    img2.write_bytes(b"\xff\xd8\xff\xe0MOREFAKEBYTES")

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.content
        return httpx.Response(202, json={"status": "queued"})

    client = CvatClient(base_url="http://cvat.local/api", transport=httpx.MockTransport(handler))
    result = client.upload_task_data(42, [img1, img2], image_quality=80)

    assert result == {"status": "queued"}
    assert captured["url"] == "http://cvat.local/api/tasks/42/data"
    assert "multipart/form-data" in captured["content_type"]
    assert b"frame_0001.jpg" in captured["body"]
    assert b"frame_0002.jpg" in captured["body"]
    assert b"FAKEJPEGBYTES" in captured["body"]


def test_upload_task_data_rejects_missing_file(tmp_path):
    client = CvatClient()
    with pytest.raises(FileNotFoundError):
        client.upload_task_data(1, [tmp_path / "does_not_exist.jpg"])


def test_upload_task_data_rejects_empty_list():
    client = CvatClient()
    with pytest.raises(ValueError):
        client.upload_task_data(1, [])


def test_upload_task_data_raises_cvat_api_error_on_non_2xx(tmp_path):
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"fake")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    client = CvatClient(transport=httpx.MockTransport(handler))
    with pytest.raises(CvatApiError, match="500"):
        client.upload_task_data(1, [img])


def test_parse_coco_annotations_splits_by_category():
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "categories": [{"id": 1, "name": "bag_body"}, {"id": 2, "name": "print_mark"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "segmentation": [[0, 0, 10, 0, 10, 10]]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [2, 2, 3, 3], "segmentation": []},
        ],
    }
    parsed = CvatClient().parse_coco_annotations(coco)
    anns = parsed["parsed_annotations"][1]
    categories = sorted(a["category"] for a in anns)
    assert categories == ["bag_body", "print_mark"]
    assert parsed["images"][1]["file_name"] == "a.jpg"


def test_inter_annotator_agreement_identical_masks_is_one():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    agreement = CvatClient().calculate_inter_annotator_agreement([mask], [mask])
    assert agreement == pytest.approx(1.0)


def test_inter_annotator_agreement_disjoint_masks_is_zero():
    m1 = np.zeros((20, 20), dtype=bool)
    m1[0:5, 0:5] = True
    m2 = np.zeros((20, 20), dtype=bool)
    m2[15:20, 15:20] = True
    agreement = CvatClient().calculate_inter_annotator_agreement([m1], [m2])
    assert agreement == pytest.approx(0.0)
