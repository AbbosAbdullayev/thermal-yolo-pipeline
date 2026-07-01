import os
import numpy as np
import shutil
import yaml
import torch
from ultralytics import YOLO
import argparse
parser = argparse.ArgumentParser(description='Convert FLIR thermal annotations to YOLO format')
parser.add_argument('--yaml_path', type=str,default="./dataset/FLIR_ADAS_v2/data.yaml", help='Training config file')
parser.add_argument('--imgsize', type=int,default=640, help='Training image width')
parser.add_argument('--batch',type=int,default=64,help='Number of images per batch')
parser.add_argument('--device',type=int,default=0, help='Device for training pipeline')
parser.add_argument('--epochs',type=int,default=300,help='Training epoches')
args = parser.parse_args()

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:',device)
model=YOLO("yolo26n.pt")
results=model.train(
    data=args.yaml_path,
    epochs=args.epochs,
    imgsz=args.imgsize,
    device=args.device,
    batch=args.batch
)