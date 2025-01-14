# Copyright (c) OpenMMLab. All rights reserved.

from PIL import Image, ImageDraw
import ast
import json
import os
from argparse import ArgumentParser

# from mmengine.logging import print_log

from mmdet.apis import DetInferencer

import torch
from transformers import AutoImageProcessor, AutoModel

from PIL import Image
import requests

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs', type=str, default='/data/decoreted/test/', help='Input image file, folder path, or COCO JSON file.')
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
    # parser.add_argument(
    #     '--image_path', type=str, default="./", help='image path')
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
    
    import glob
    image_paths = glob.glob(os.path.join(call_args['inputs'], '*.png'))
    call_args['inputs'] = image_paths
    print("************************************************** {} of images detected **************************************************".format(len(image_paths)))
    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''
    if call_args['model'].endswith('.pth'):
        # print_log('The model is a weight file, automatically '
        #           'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None
    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)
    return init_args, call_args

def draw_semantic_groups(output_dict, H, W):
    white_background = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(white_background)

    for label, bbox in zip(output_dict['labels'], output_dict['bboxes']):
        l,t,r,b = bbox
        if label == 'Subgroup':
            draw.rectangle([l, t, r, b], outline=(0, 0, 255), width=2)
        elif label == 'Parent Group':
            draw.rectangle([l, t, r, b], outline=(0, 255, 0), width=2)
    return white_background

def main():

    dinov2_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2_model = dinov2_model.to('cuda')

    init_args, call_args = parse_args()
    inferencer = DetInferencer(**init_args)
    label2name = {0: "Parent Group", 
                  1: "Subgroup", 
                  2: "a group"}

    if isinstance(call_args['inputs'], list):
        for image_path in call_args['inputs']:
            call_args['inputs'] = image_path
            output_dict = inferencer(**call_args)
            new_output_dict = {'labels': [], 'bboxes': []}
            print(output_dict['predictions'][0].keys())

            for label, score, bbox in zip(output_dict['predictions'][0]['labels'], output_dict['predictions'][0]['scores'], output_dict['predictions'][0]['bboxes']):
                if score > 0.3 and label != 3:
                    new_output_dict['labels'].append(label2name[label])
                    new_output_dict['bboxes'].append(bbox)

            input_image = Image.open(image_path)
            W, H = input_image.size
            grouped_image = draw_semantic_groups(new_output_dict, H, W)
            
            inputs = dinov2_processor(images=grouped_image, return_tensors="pt")
            output_embeddings = dinov2_model(inputs['pixel_values'].cuda())
            output_embeddings = output_embeddings.squeeze(0).detach().cpu()

            out_dir = call_args['out_dir']
            save_fname = image_path.split('/')[-1].replace('.png', '.pt')
            torch.save(output_embeddings, os.path.join(out_dir, save_fname))

if __name__ == '__main__':
    main()