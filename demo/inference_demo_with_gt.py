# Copyright (c) OpenMMLab. All rights reserved.
"""Image Demo.

This script adopts a new inference class, currently supports image path,
np.array and folder input formats, and will support video and webcam
in the future.

Example:
    Save visualizations and predictions results::

        CUDA_VISIBLE_DEVICES=0 python demo/inference_demo.py /data/miricanvas-ai-labs-official/02.Dataset/01.Find-meaning-groups/01.miricanvas/PPT-template_ko/three_groups_val.json projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16x
        b6_16e_o365toCOCOtomiridih.py --weights /data/checkpoints/jjy/three_groups_swin_l_b3_16e/epoch_1.pth --image_path /workspace/data/decorate_detection/Raw/originSize_png

    Visualize prediction results::

        python demo/image_demo.py demo/demo.jpg rtmdet-ins-s --show

        python demo/image_demo.py demo/demo.jpg rtmdet-ins_s_8xb32-300e_coco \
        --show
"""

import ast
import json
import os
from argparse import ArgumentParser

from mmengine.logging import print_log

from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes
from setproctitle import setproctitle
import wandb, cv2
from pycocotools.coco import COCO

def visualize_ground_truth(image_path, annotations, coco):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    for ann in annotations:
        bbox = ann['bbox']
        category_id = ann['category_id']
        category_name = coco.loadCats(category_id)[0]['name']
        
        # Draw bounding box
        x, y, w, h = map(int, bbox)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Put label
        cv2.putText(image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return image

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs', type=str, help='Input image file, folder path, or COCO JSON file.')
    parser.add_argument(
        'model',
        type=str,
        help='Config or checkpoint .pth file or the model name '
        'and alias defined in metafile. The model configuration '
        'file will try to read from .pth if the parameter is '
        'a .pth weights file.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')
    parser.add_argument(
        '--texts', help='text prompt, such as "bench . car .", "$: coco"')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    parser.add_argument(
        '--image_path', type=str, default="./", help='image path')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection vis results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection json results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--palette',
        default='none',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--custom-entities',
        '-c',
        action='store_true',
        help='Whether to customize entity names? '
        'If so, the input text should be '
        '"cls_name1 . cls_name2 . cls_name3 ." format')

    call_args = vars(parser.parse_args())
    
    # Handle COCO JSON input
    if call_args['inputs'].endswith('.json'):
        with open(call_args['inputs'], 'r') as f:
            coco_data = json.load(f)
        image_paths = [os.path.join(call_args['image_path'], coco_data['images'][i]['file_name']) for i in range(len(coco_data['images']))]
        call_args['json_file'] = call_args['inputs']
        call_args['inputs'] = image_paths
        

    del call_args['image_path']

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically '
                  'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)
    
    return init_args, call_args


def main():
    setproctitle('shcho_mmdetection_test')
    init_args, call_args = parse_args()
    inferencer = DetInferencer(**init_args)

    # Load COCO annotations
    coco = COCO(call_args['json_file'])
    del call_args['json_file']

    # Create a new wandb table
    table = wandb.Table(columns=["Index", "Visualization", "Ground Truth"])

    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        wandb.init(project='decorate_detection', entity='miridih-ailabs', name=f'shcho_evaluate_{init_args["model"]}_{init_args["weights"]}')
    
    result_dict, gt_dict = [], []
    # Handle multiple image inputs from COCO JSON
    if isinstance(call_args['inputs'], list):
        for idx, image_path in enumerate(call_args['inputs']):
            if idx > 400:
                break
            call_args['inputs'] = image_path
            output_dict = inferencer(**call_args)

            # Get image id from file name
            image_id = next((img['id'] for img in coco.dataset['images'] if img['file_name'] in image_path), None)
            annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))

            # Visualize ground truth
            gt_image = visualize_ground_truth(image_path, annotations, coco)
            
            template_idx = os.path.basename(image_path).split('.')[0]

            if output_dict['visualization'] is not None:
                table.add_data(template_idx, wandb.Image(output_dict['visualization'][0]), wandb.Image(gt_image))

    else:
        result_dict = inferencer(**call_args)
    
    wandb.log({"results_table": table})

    if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                           and call_args['no_save_pred']):
        print_log(f'results have been saved at {call_args["out_dir"]}')
    
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        wandb.finish()


if __name__ == '__main__':
    main()