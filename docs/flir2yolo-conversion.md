# FLIR → YOLO Conversion (`flir2yolo.py`)

Convert [FLIR ADAS v2](https://www.flir.com/oem/adas/adas-dataset-form/) thermal dataset annotations from COCO format to YOLO format.

## What it does

- Reads `coco.json` from each split folder (train/val)
- Remaps COCO categories to a custom 12-class taxonomy (`person`, `bike`, `car`, `motor`, `bus`, `train`, `truck`, `light`, `sign`, `dog`, `scooter`, `other vehicle`)
- Writes YOLO-format `.txt` labels (normalized `x_center y_center width height`) per image
- Generates a `classes.txt` at the dataset root
- Optionally draws bounding boxes on sample images for a quick visual sanity check

## Expected folder structure

```
FLIR_ADAS_v2/
├── images_thermal_train/
│   ├── coco.json
│   └── ...images...
└── images_thermal_val/
    ├── coco.json
    └── ...images...
```

## Usage

```bash
python flir2yolo.py --base_path ./dataset/FLIR_ADAS_v2
```

Labels are written to `images_thermal_<split>/yolo_annotations/`.
`classes.txt` is written once to the dataset root (`--base_path`).

### Options

| Flag | Default | Description |
|---|---|---|
| `--base_path` | `./dataset/FLIR_ADAS_v2` | Root of the FLIR ADAS v2 dataset |
| `--train_folder` | `images_thermal_train` | Train split folder name |
| `--val_folder` | `images_thermal_val` | Val split folder name |
| `--json_file` | `coco.json` | COCO annotation filename inside each split folder |
| `--draw-samples` | `0` | Number of sample images to draw bounding boxes on, for a sanity check |

### Example with sample bbox drawing

```bash
python flir2yolo.py --base_path ./dataset/FLIR_ADAS_v2 --draw-samples 10
```

Sample images are saved to `images_thermal_<split>/sample_images/`.

## Requirements

```bash
pip install opencv-python numpy
```
