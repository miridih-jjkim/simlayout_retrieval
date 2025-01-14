import torch
from torchvision import transforms
from PIL import Image
from torch.nn.functional import cosine_similarity


# DINOv2 모델 로드
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
model.eval()
# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def split_image_into_grid(image, grid_size):
    width, height = image.size
    piece_width = width // grid_size
    piece_height = height // grid_size
    pieces = []
    for row in range(grid_size):
        for col in range(grid_size):
            left = col * piece_width
            top = row * piece_height
            right = left + piece_width
            bottom = top + piece_height
            if col == grid_size - 1:
                right = width
            if row == grid_size - 1:
                bottom = height
            box = (left, top, right, bottom)
            piece = image.crop(box)
            pieces.append(piece)
    print(len(pieces))
    return pieces


def get_image_embedding(image, model, grid_size, img_size=None):

    image = Image.open(image).convert('RGB')

    img_size_out = image.size
    if img_size is not None:
        image = image.resize(img_size, Image.LANCZOS)
    image_pieces = split_image_into_grid(image, grid_size)

    # Transform all pieces and stack them into a single batch tensor
    ## GPU서버
    # batch = torch.stack([transform(piece) for piece in image_pieces]).to('cuda')
    ## 로컬
    batch = torch.stack([transform(piece) for piece in image_pieces]).to('cuda')
    # As-is: [1, 3, 224, 224] x 169 iteration
    # To-be: [169, 3, 224, 224] x 1 iteration
    # [batch_size, channel, height, width]
    with torch.no_grad():
        embeddings = model(batch)

    return embeddings, img_size_out


def compare_images(image1, image2, model, grid_size):
    embeddings1, img_size1 = get_image_embedding(image1, model, grid_size)
    embeddings2, _ = get_image_embedding(image2, model, grid_size, img_size1)
    similarities = cosine_similarity(embeddings1, embeddings2, dim=1).tolist()
    return min(similarities)    

def is_different(image1, image2, model, grid_size, threshold=0.9969):
    similarity = compare_images(image1, image2, model, grid_size)
    return similarity

image1 = '1.png'
image2 = '2.png'
similarity = is_different(image1, image2, model, grid_size=13)
print(similarity)