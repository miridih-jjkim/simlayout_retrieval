import os
import random
import torch
from dreamsim import dreamsim
from PIL import Image
from tqdm import tqdm
import json

base_dir = '/data/decoreted/group_bbox_images'
image_name_list = [f for f in os.listdir(base_dir) if f.endswith('.png')]

# target_query_images = random.sample(image_name_list, 5)
target_query_images = ['31024.png', '34883.png', '35113.png', '37303.png', '37304.png', '55201.png', '55130.png', '55388.png', '56708.png', '57230.png', '37409.png', '60109.png']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = dreamsim(pretrained=True, device=device)

for target_image in tqdm(target_query_images):

    target_id = target_image.replace('.png', '')
    simdict = {}

    target_image_path = os.path.join(base_dir, target_image)
    target_image = preprocess(Image.open(target_image_path)).to(device)

    target_embedding = model.embed(target_image)

    for key_image in tqdm(image_name_list):

        if key_image == target_image:
            continue

        key_id = key_image.replace('.png', '')

        key_image_path = os.path.join(base_dir, key_image)
        key_image = preprocess(Image.open(key_image_path)).to(device)
        key_embedding = model.embed(key_image)

        distance = model(target_image, key_image)

        simdict[key_id] = distance.cpu().item()

    simdict_sort = sorted(simdict.items(), key=lambda x: x[1], reverse=False)
    json.dump(simdict_sort, open(f'/data/decoreted/sim_results/{target_id}.json', 'w'))

    