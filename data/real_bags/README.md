# Real Bag Annotated Dataset (`data/real_bags/`)

Place real factory conveyor annotations and frame images here for fine-tuning and holdout evaluation.

## Directory Structure
```text
data/real_bags/
├── annotations.json      # COCO 1.0 format instances exported from CVAT
└── images/               # Directory containing referenced image files (JPG/PNG)
    ├── frame_0001.jpg
    ├── frame_0002.jpg
    └── ...
```

## Annotation Classes
- `bag_body` (polygon / segmentation): Polygon boundary of the bag.
- `print_mark` (bbox / rectangle): Printed brand/product stamp on the bag.

See [docs/cvat_setup.md](../../docs/cvat_setup.md) for full annotation and export instructions.
