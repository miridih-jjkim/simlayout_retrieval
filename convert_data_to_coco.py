
import json
import os
from tqdm import tqdm

# to_sh ={}

def process_json_file(json_file, input_dir):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Parent Group"},
            {"id": 2, "name": "Subgroup"},
            {"id": 3, "name": "a group"},
        ]
    }
    category_id_map = {
        "Parent Group": 1,
        "Subgroup": 2,
        "a group": 3,}
    annotation_id = 1
    image_id = 1

    for filename in json_file:
        if filename.endswith('.json'):
            with open(os.path.join(input_dir, filename), 'r') as file:
                data = json.load(file)
                image_info = {
                    "id": image_id,
                    "file_name": data['data']['image'].split('/')[-1],
                    "width": data['annotations'][0]['result'][0]['original_width'],
                    "height": data['annotations'][0]['result'][0]['original_height']
                }
                coco_format['images'].append(image_info)

        
                for annotation in data['annotations'][0]['result']:
                    if len(annotation['value']['rectanglelabels']) != 3:                        
                        label = 'ignore'

                        # 3개 짜리 그룹이어야 하는데 라벨링 오류인 경우
                        for c1 in ['a group', 'Parent Group', 'Subgroup']:
                            if c1 in annotation['value']['rectanglelabels']:
                                label = c1
                            else:
                                continue

                        if label == 'ignore':
                            iscrowd = 1
                            label = 'Subgroup'
                        else:
                            iscrowd = 0

                        annotation_info = {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": category_id_map[label],
                                "bbox": [
                                    annotation['value']['x'],
                                    annotation['value']['y'],
                                    annotation['value']['width'],
                                    annotation['value']['height']
                                ],
                                "area": annotation['value']['width'] * annotation['value']['height'],
                                "iscrowd": iscrowd
                            }
                        coco_format['annotations'].append(annotation_info)
                        annotation_id += 1
                    # if len(annotation['value']['rectanglelabels']) != 3:
                    #     continue
                    else:
                        for label in annotation['value']['rectanglelabels']:
                            if label not in ['a group', 'Parent Group', 'Subgroup']:
                                continue
                            if label not in category_id_map:
                                category_id_map[label] = len(category_id_map) + 1
                                coco_format['categories'].append({
                                    "id": category_id_map[label],
                                    "name": label
                                })

                            annotation_info = {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": category_id_map[label],
                                "bbox": [
                                    annotation['value']['x'],
                                    annotation['value']['y'],
                                    annotation['value']['width'],
                                    annotation['value']['height']
                                ],
                                "area": annotation['value']['width'] * annotation['value']['height'],
                                "iscrowd": 0
                            }
                            coco_format['annotations'].append(annotation_info)
                            annotation_id += 1
                image_id += 1
    return coco_format

def convert_to_coco(input_dir, output_file, val_image_count):
    
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    val_files = image_files[:val_image_count]
    train_files = image_files[val_image_count:]

    
    

    train_coco_format = process_json_file(train_files, input_dir)
    val_coco_format = process_json_file(val_files, input_dir)

    
    print("number of files: ", len(os.listdir(input_dir)))
    

    with open(output_file.replace('.json', '_train.json'), 'w') as outfile:
        json.dump(train_coco_format, outfile, indent=4)

    with open(output_file.replace('.json', '_val.json'), 'w') as outfile:
        json.dump(val_coco_format, outfile, indent=4)

# Usage
input_directory = '/workspace/data/decorate_detection/labeled_undivided'
output_coco_file = '/data/miricanvas-ai-labs-official/02.Dataset/01.Find-meaning-groups/01.miricanvas/PPT-template_ko/three_groups_32k_with_ignore.json'
val_image_count = 1000  # Specify the number of images for validation
convert_to_coco(input_directory, output_coco_file, val_image_count)