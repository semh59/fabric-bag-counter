# CVAT Setup & Real Bag Dataset Annotation Guide

This guide explains how to deploy CVAT (Computer Vision Annotation Tool), configure the standardized label schema for industrial woven/kraft bags, export annotations in COCO format, and feed them into the RF-DETR fine-tuning pipeline.

---

## 1. Quick CVAT Deployment (Docker Compose)

You can launch the complete self-hosted CVAT stack included directly with the project:

```bash
# Start CVAT services (Postgres, Redis, CVAT Server, RQ Worker, Web UI)
docker compose -f docker-compose.cvat.yml up -d

# Create initial CVAT administrator account
docker compose -f docker-compose.cvat.yml exec cvat-server server/manage.py createsuperuser
```

Once running, the CVAT Web UI is immediately accessible at `http://localhost:8088`.

---

## 2. Standardized Label Schema (§6.4)

The project includes an automated REST client (`packages.cs_data.cvat_client.CvatClient`) that sets up the required 2-class annotation specification:

| Class Name | Type | Attributes | Description |
|---|---|---|---|
| `bag_body` | Polygon | `visible_ratio` (0.0-1.0), `heavily_occluded` (bool) | Exact boundary polygon of the bag instance |
| `print_mark` | Rectangle | — | Text/logo printed mark on the bag face |

---

## 3. Active Learning & Frame Upload Pipeline

1. **Extract Frames / Hard Mining**:
   Factory conveyor videos uploaded via Web Dashboard (`/lines/{line_id}/upload_video`) or mined by `HardFrameMiner` are extracted into `./data/extracted_frames/`.
   
2. **Push to CVAT Task**:
   Use `CvatClient` to create a task and upload frames automatically:
   ```python
   from packages.cs_data.cvat_client import CvatClient

   client = CvatClient(base_url="http://localhost:8080/api", auth_token="<YOUR_CVAT_TOKEN>")
   task = client.create_task("Factory_Conveyor_Line1_Batch1")
   client.upload_task_data(task["id"], image_paths=["data/extracted_frames/frame_0001.jpg", ...])
   ```

3. **Annotate in CVAT**:
   - Draw polygon masks around each individual `bag_body`.
   - Draw bounding rectangles around `print_mark` if printed logos/text exist.

4. **Export Dataset**:
   - In CVAT UI: Click **Export Task Dataset** $\rightarrow$ Format: **COCO 1.0**.
   - Download the zip archive and extract it to `data/real_bags/`:
     ```text
     data/real_bags/
     ├── annotations.json     (COCO instances JSON format)
     └── images/              (JPG/PNG frames matching annotations)
         ├── frame_0001.jpg
         └── frame_0002.jpg
     ```

---

## 4. Fine-Tuning & Evaluation on Real Bags

Once `data/real_bags/annotations.json` is in place:

- **PyTorch Training & ONNX Export**:
  ```bash
  python packages/cs_vision/train_rfdetr.py
  ```
  The trainer automatically loads `data/real_bags/` alongside synthetic pretraining scenes, blending real factory lighting and bag textures.

- **Automated Holdout Evaluation**:
  Background training jobs in `services.jobrunner.worker` evaluate the exported ONNX model directly against `data/real_bags/annotations.json` and mark evaluation metrics as `dataset_type: "real_holdout"`.
