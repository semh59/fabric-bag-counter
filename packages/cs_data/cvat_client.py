"""CVAT REST API client, COCO converter, and inter-annotator agreement verifier (§6.3, §9.5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import httpx
import numpy as np
from packages.cs_core.geometry import compute_mask_iou


class CvatApiError(RuntimeError):
    """Raised when the CVAT REST API returns a non-2xx response."""


class CvatClient:
    """Interface to self-hosted CVAT instance for annotation tasks."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080/api",
        auth_token: str | None = None,
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout = timeout_seconds
        # Test-only injection point for httpx.MockTransport -- production
        # callers never pass this, so httpx.Client builds its real transport.
        self._transport = transport

    def get_labels_spec(self) -> list[dict[str, Any]]:
        """Return standardized 2-class CVAT specification (§6.4)."""
        return [
            {
                "name": "bag_body",
                "type": "polygon",
                "attributes": [
                    {
                        "name": "visible_ratio",
                        "input_type": "number",
                        "default_value": "1.0",
                        "values": ["0.0", "1.0", "0.1"],
                    },
                    {
                        "name": "heavily_occluded",
                        "input_type": "checkbox",
                        "default_value": "false",
                        "values": ["true", "false"],
                    },
                ],
            },
            {
                "name": "print_mark",
                "type": "rectangle",
                "attributes": [],
            },
        ]

    def create_task(self, name: str, project_id: int | None = None) -> dict[str, Any]:
        """Create a new annotation task in CVAT via its REST API.

        Issues a real `POST {base_url}/tasks` request with the standard
        2-class label spec as payload, authenticating via the CVAT token
        auth scheme (`Authorization: Token <auth_token>`). Raises
        `CvatApiError` on any non-2xx response.
        """
        url = f"{self.base_url}/tasks"
        payload: dict[str, Any] = {
            "name": name,
            "labels": self.get_labels_spec(),
        }
        if project_id is not None:
            payload["project_id"] = project_id

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Token {self.auth_token}"

        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                response = client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise CvatApiError(f"CVAT create_task request to {url} failed: {exc}") from exc

        if not (200 <= response.status_code < 300):
            raise CvatApiError(
                f"CVAT create_task failed: HTTP {response.status_code} for {url}: {response.text}"
            )

        return response.json()

    def upload_task_data(
        self,
        task_id: int,
        image_paths: list[Path | str],
        image_quality: int = 70,
    ) -> dict[str, Any]:
        """Upload real frame images into an existing CVAT task via its REST API.

        create_task() only creates the task shell (name + label spec); CVAT
        has no data to annotate until images are actually uploaded to it.
        Issues a real `POST {base_url}/tasks/{task_id}/data` multipart
        request with each image as a `client_files` part, authenticating the
        same way create_task() does. Raises `CvatApiError` on any non-2xx
        response, `FileNotFoundError` if a given path doesn't exist.
        """
        if not image_paths:
            raise ValueError("upload_task_data requires at least one image path")

        paths = [Path(p) for p in image_paths]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Image path does not exist: {p}")

        url = f"{self.base_url}/tasks/{task_id}/data"
        headers = {"Accept": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Token {self.auth_token}"

        files = [("client_files", (p.name, p.read_bytes())) for p in paths]
        data = {"image_quality": str(image_quality)}

        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                response = client.post(url, data=data, files=files, headers=headers)
        except httpx.HTTPError as exc:
            raise CvatApiError(f"CVAT upload_task_data request to {url} failed: {exc}") from exc

        if not (200 <= response.status_code < 300):
            raise CvatApiError(
                f"CVAT upload_task_data failed: HTTP {response.status_code} for {url}: {response.text}"
            )

        return response.json() if response.content else {}

    def calculate_inter_annotator_agreement(
        self,
        annotator1_masks: list[np.ndarray],
        annotator2_masks: list[np.ndarray],
    ) -> float:
        """Calculate mean pairwise Mask-IoU agreement between two independent annotators (§6.3).

        Rule: for the first 100 frames, two annotators work independently. The
        IoU agreement between them must be >= 0.85.
        """
        n = min(len(annotator1_masks), len(annotator2_masks))
        if n == 0:
            return 1.0

        ious = []
        for i in range(n):
            m1 = annotator1_masks[i]
            m2 = annotator2_masks[i]
            iou = compute_mask_iou(m1, m2)
            ious.append(iou)

        return float(np.mean(ious)) if ious else 1.0

    def parse_coco_annotations(self, coco_dict: dict[str, Any]) -> dict[str, Any]:
        """Parse raw COCO export into internal format with amodal segmentation masks."""
        categories = {cat["id"]: cat["name"] for cat in coco_dict.get("categories", [])}
        images = {img["id"]: img for img in coco_dict.get("images", [])}
        annotations = coco_dict.get("annotations", [])

        parsed_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in annotations:
            img_id = ann["image_id"]
            cat_name = categories.get(ann["category_id"], "unknown")
            item = {
                "id": ann["id"],
                "category": cat_name,
                "bbox": ann.get("bbox", []),
                "segmentation": ann.get("segmentation", []),
                "attributes": ann.get("attributes", {}),
            }
            parsed_by_image.setdefault(img_id, []).append(item)

        return {"images": images, "parsed_annotations": parsed_by_image}
