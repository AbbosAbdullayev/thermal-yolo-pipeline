import os
import yaml
import argparse
    
def generate_data_yaml(base_path, train_folder, val_folder, selected_classes):
    class_names = [name for name, idx in sorted(selected_classes.items(), key=lambda x: x[1])]

    data_config = {
        'path': os.path.abspath(base_path),
        'train': train_folder,
        'val': val_folder,
        'names': class_names,
        'nc': len(class_names)
    }

    yaml_path = os.path.join(base_path, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, sort_keys=False)

    print(f"data.yaml written to {yaml_path} ({len(class_names)} classes)")
    return yaml_path
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert FLIR thermal annotations to YOLO format')
    parser.add_argument('--base_path', type=str,default="./dataset/FLIR_ADAS_v2", help='Path to the FLIR_v2 main folder')
    parser.add_argument('--train_folder', type=str,default="images_thermal_train", help='Path to the train image foler')
    parser.add_argument('--val_folder',type=str,default="images_thermal_val",help='Path to validation image folder')
    args = parser.parse_args()

    SELECTED_CLASSES = {
        'person': 0, 'bike': 1, 'car': 2, 'motor': 3, 'bus': 4,
        'train': 5, 'truck': 6, 'light': 7, 'sign': 8,
        'dog': 9, 'scooter': 10, 'other vehicle': 11
    }

    generate_data_yaml(args.base_path, args.train_folder, args.val_folder, SELECTED_CLASSES)
    
    