import cv2
import numpy as np
import os
import argparse
import json
from pathlib import Path
import random

parser = argparse.ArgumentParser(description='Convert FLIR thermal annotations to YOLO format')
parser.add_argument('--image_folder', type=str,default="./FLIR_ADAS_1_3/val", help='Path to the FLIR image folder')
parser.add_argument('--json', type=str,default="./thermal_annotations.json", help='Path to the JSON annotation file')
parser.add_argument('--draw-samples',default=0, help='Draw bounding boxes on sample images')
args = parser.parse_args()
def draw_bounding_boxes(image, bbox, category_id,color=None,thickness=2):
    color=color or [random.randint(0, 255) for _ in range(3)]
    x, y, w, h = bbox
    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
    cv2.putText(image, str(category_id), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)
def remap_category_id(category_id):
    if category_id == 16 or category_id == 17:
        return 12  # Map 'cat' and 'dog' as animal to class 12
    else:
        return category_id - 1  # Ensures class starts from 0    
def main():
    # Load the JSON annotation file
    json_file = os.path.join(args.image_folder, args.json)
    if not os.path.exists(json_file):
        print(f"JSON annotation file not found: {json_file}")
        return  
    with open(json_file, 'r') as f:
        annotations = json.load(f)
        #Create a folder to save YOLO annotations
    output_folder=os.path.join(args.image_folder, 'yolo_annotations')
    os.makedirs(output_folder, exist_ok=True)
    # Json structure checking 
    #print(f"Json structure: {annotations.keys()}")
    count=0
    if args.draw_samples>0:
       sample_draws=os.path.join(args.image_folder, 'sample_images')
       os.makedirs(sample_draws, exist_ok=True)
       for image in annotations['images'][:5]:  # Draw bounding boxes for the first args.draw_samples images
           image_file = image['file_name']
           image_name = Path(image_file).name
           image_id = image['id']
           image_path=os.path.join(args.image_folder,image_file)
           if not os.path.exists(image_path):
               print(f"Image file not found: {image_path}")
               continue
           # Load the image
           image=cv2.imread(image_path)
           for annotation in annotations['annotations']:
               if annotation['image_id']==image_id:
                   category_id=annotation['category_id']
                   category_id=remap_category_id(category_id)
                   bbox=annotation['bbox']  # [x, y, width, height]
                   print(f"Drawing bounding box for category_id {category_id} on image {image_file}")
                   draw_bounding_boxes(image, bbox, category_id)
           sample_output_path=os.path.join(sample_draws,image_name)
           cv2.imwrite(sample_output_path,image)
           print(f"Sample image with bounding boxes saved: {sample_output_path}")

    selected_classes={'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5,'bus': 6,'train': 7,'truck':8,'boat':9,'traffic light':10, 'stop sign': 12,'cat': 16,'dog': 17}  # Example mapping
    for image in annotations['images'][:10]:
        image_file = image['file_name']
        image_name = Path(image_file).name
        image_id = image['id']
        
        image_path=os.path.join(args.image_folder,image_file)
        if not os.path.exists(image_path):  
            print(f"Image file not found: {image_path}")
            continue
        # Load the image to get its dimensions
        image=cv2.imread(image_path)
        height,width,_=image.shape
        count+=1
        results=[]
        for annotation in annotations['annotations']:
            
            if annotation['image_id']==image_id:
                category_id=annotation['category_id']
                if category_id in selected_classes.values():
                    bbox=annotation['bbox']  # [x, y, width, height]
                    x_center=(bbox[0]+bbox[2]/2)/width
                    y_center=(bbox[1]+bbox[3]/2)/height
                    w=bbox[2]/width
                    h=bbox[3]/height
                    category_id=remap_category_id(category_id)
                    yolo_annotation=f"{category_id} {x_center:.3f} {y_center:.3f} {w:.3f} {h:.3f}\n"
                    #print(f"YOLO annotation: {yolo_annotation.strip()}")
                    results.append(yolo_annotation)
                else:
                    print(f"Skipping category_id {category_id} for image {image_file}")    
                # Save YOLO annotation to a text file            
        if results:
            output__file=os.path.join(output_folder,f"{Path(image_file).stem}.txt")
            with open(output__file, 'w') as f:
                f.writelines(results)
    print(f"Processed {count} images. YOLO annotations saved in {output_folder}")
    print("Done!!!")        
if __name__ == "__main__":
    main()