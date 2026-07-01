# thermal-vision-tools

Scripts for thermal (FLIR) dataset prep, object detection, and segmentation workflows.

## Scripts

| Script | Purpose | Docs |
|---|---|---|
| `flir2yolo.py` | Convert FLIR ADAS v2 COCO annotations to YOLO format | [docs/flir2yolo-conversion.md](docs/flir2yolo-conversion.md) |

## Quick start

```bash
pip install -r requirements.txt
python flir2yolo.py --base_path ./dataset/FLIR_ADAS_v2
```

See each script's doc page above for full options and details.
