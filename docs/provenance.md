# Model and Code Provenance Registry

## 1. Architecture & Pretrained Model Provenance

### RF-DETR Seg (Nano / Small / Medium)
- **Paper**: DETRs Beat YOLOs on Real-time Object Detection
- **Code License**: Apache-2.0
- **Model Weights License**: Apache-2.0
- **Prohibited Variants**: RF-DETR Plus, RF-DETR-XL, RF-DETR-2XL (licensed under PML 1.0)
- **Verification Date**: 2026-08-24
- **Verification Status**: Approved (Zero Copyleft Risk)

### ByteTrack Base Tracker
- **Paper**: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022, arXiv:2110.06864
- **Code License**: MIT (github.com/FoundationVision/ByteTrack)
- **Modifications**:
  - `BeltMotionModel` conveyor velocity prior integrated into Kalman state transitions.
  - Cost matrix rewritten to compute exact `1.0 - Mask_IoU` and Euclidean centroid distance instead of Bounding Box IoU.
  - Per-track monotonic `crossing_seq` counter for directional gate crossings and backward slip handling.
  - Latent track hypothesis integration for `merge_detector` split events.

## 2. Dataset Provenance
- All synthetic datasets generated using internal `cs_data.synth` module using clean CC0 / custom licensed conveyor background and authorized customer bag templates.
- Real factory dataset versions tracked with SHA-256 manifest hashes and strict separation across site sessions.
