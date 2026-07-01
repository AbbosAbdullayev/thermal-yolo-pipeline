import cv2
import numpy as np
import os
import argparse
import json
from collections import defaultdict
from pathlib import Path
import random

def draw_bounding_boxes(image, bbox, category_id,class_name,color=None,thickness=2):
    color=color or [random.randint(0, 255) for _ in range(3)]
    x, y, w, h = bbox
    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
    cv2.putText(image, str(category_id), (int(x), int(y) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
    cv2.putText(image, str(class_name), (int(x)+5, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
def coco_category_to_yolo_class(category_id, category_names, selected_classes):
    class_name = category_names.get(category_id)
    if class_name not in selected_classes:
        return None, None
    return selected_classes[class_name], class_name    

def main(json_file,split_folder_path,output_folder):
    # Load the JSON annotation file
    if not os.path.exists(json_file):
        print(f"JSON annotation file not found: {json_file}")
        return  
    with open(json_file, 'r') as f:
        coco = json.load(f)

        #Create a folder to save YOLO annotations
    yolo_labels_folder=os.path.join(split_folder_path, 'yolo_annotations')
    os.makedirs(yolo_labels_folder, exist_ok=True)

    # Json structure checking 
    print(f"Total images:{len(coco['images'])}")
    print(f"Total annotations:{len(coco['annotations'])}")
    print(f"Total categegories:{len(coco['categories'])}")
   
    selected_classes={'person': 0, 'bike': 1, 'car': 2, 'motor': 3, 'bus': 4,'train': 5,'truck': 6,'light':7,'sign': 8,'dog': 9,'scooter': 10,'other vehicle':11}  # Example mapping 
   
    annotations_by_image=defaultdict(list)
    
    for ann in coco['annotations']:
        annotations_by_image[ann['image_id']].append(ann) 
    images_by_id={img['id']: img for img in coco['images']}
    coco_category_names={cat['id']:cat['name'] for cat in coco['categories']}
    converted,skipped=0,0
    
    if args.draw_samples>0:
       sample_draws=os.path.join(split_folder_path, 'sample_images')
       os.makedirs(sample_draws, exist_ok=True)
       for fname in coco['images'][:args.draw_samples]:  # Draw bounding boxes for the first args.draw_samples images
           image_file = fname['file_name']
           image_name = Path(image_file).name
           image_id = fname['id']
           image_path=os.path.join(split_folder_path,image_file)
           if not os.path.exists(image_path):
               print(f"Image file not found: {image_path}")
               continue
           # Load the image
           image=cv2.imread(image_path)
           
           for annotation in annotations_by_image.get(image_id, []):
                   category_id=annotation['category_id']
                   class_id,class_name=coco_category_to_yolo_class(category_id, coco_category_names, selected_classes)
                   bbox=annotation['bbox']  # [x, y, width, height]
                   print(f"Drawing bounding box for category_id {class_id} on image {image_file}")
                   draw_bounding_boxes(image, bbox, class_id,class_name)
           sample_output_path=os.path.join(sample_draws,image_name)
           cv2.imwrite(sample_output_path,image)
           print(f"Sample image with bounding boxes saved: {sample_output_path}")
       
    for image_id,img_info in images_by_id.items():
        height, width = img_info['height'], img_info['width']
        image_filename = os.path.basename(img_info['file_name'])
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(yolo_labels_folder, label_filename)

        lines = []
        for ann in annotations_by_image.get(image_id,[]):
            class_name=coco_category_names[ann['category_id']]
            if class_name not in selected_classes:
                continue
            category_id=selected_classes[class_name]
            
            bbox=ann['bbox']
            x_center=(bbox[0]+bbox[2]/2)/width
            y_center=(bbox[1]+bbox[3]/2)/height
            w=bbox[2]/width
            h=bbox[3]/height
                
            lines.append(f"{category_id} {x_center:.3f} {y_center:.3f} {w:.3f} {h:.3f}\n")
        if lines:    
            with open(label_path,"w") as f:
                f.writelines(lines)
            converted+=1
        else:
            skipped+=1        
    
    print(f"Processed {converted} images. YOLO annotations saved in {yolo_labels_folder}")
    print(f"Generating classes.txt from selected classes !!!!")
    class_path=os.path.join(output_folder,"classes.txt")
    with open(class_path, 'w') as f:
        for name, idx in sorted(selected_classes.items(), key=lambda x: x[1]):
            f.write(name + "\n")
    print(f"classes.txt written to {class_path}")        
    print("Done!!!")    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert FLIR_V2 thermal annotations to YOLO format')
    parser.add_argument('--base_path', type=str,default="./dataset/FLIR_ADAS_v2", help='Path to the FLIR_v2 main folder')
    parser.add_argument('--train_folder', type=str,default="images_thermal_train", help='Path to the train image foler')
    parser.add_argument('--val_folder',type=str,default="images_thermal_val",help='Path to validation image folder')
    parser.add_argument('--draw-samples',type=int,default=0, help='Draw bounding boxes on sample images')
    parser.add_argument('--json_file',type=str,default='coco.json', help='Name of the JSON annotation file')

    args = parser.parse_args()
    json_file=args.json_file
    folders=[]
    if args.train_folder and args.val_folder:
        folders.append(args.train_folder)
        folders.append(args.val_folder)
    for folder in folders:
        print(folder)
        coco_json_path=os.path.join(args.base_path,(folder),json_file)
        yolo_annotations_path=os.path.join(args.base_path,folder)
        main(coco_json_path,yolo_annotations_path,args.base_path)