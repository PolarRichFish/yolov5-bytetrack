import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


def compare_vector(vector1, vector2):
    vector1 = vector1.clone().detach()
    vector2 = vector2.clone().detach()

    v1_norm = F.normalize(vector1, dim=-1).to('cuda')
    # v2_norm = F.normalize(vector2.T, dim=-1).to('cuda')
    v2_norm = F.normalize(vector2, dim=-1).to('cuda')
    # cosa = torch.matmul(v1_norm, v2_norm)
    cosa = F.cosine_similarity(v1_norm, v2_norm, dim=-1)
    # print(cosa.shape)
    # print(cosa)
    # exit()
    return cosa


# 过滤特征相似度高的向量，返回过滤后的向量
def purn_database(vectors):
    # 储存不同的向量
    unique_vectors = []

    for vector in vectors:
        is_unique = True
        for unique_vector in unique_vectors:
            sim = compare_vector(vector, unique_vector)
            if sim > 0.95:
                is_unique = False
                break

        if is_unique:
            unique_vectors.append(vector)

    return unique_vectors


# 将根目录中所有标签名的图片文件都转化为特征并且将重复度高的特征进行过滤
@torch.no_grad()
def create_lib(root_path, device, rec_net, save_root):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])

    for label in os.listdir(root_path):
        img_file = os.path.join(root_path, label)
        vectors = []
        for img_name in os.listdir(img_file):
            img_path = os.path.join(img_file, img_name)

            img = cv2.imread(img_path)
            img = Image.fromarray(img)
            img_data = transform(img)
            img_data = img_data.unsqueeze(0).to(device)
            feature = rec_net.get_feature(img_data)
            vectors.append(feature)

        unique_vectors = purn_database(vectors)
        print(f'处理之前有{len(vectors)}个特征， 处理之后有{len(unique_vectors)}个特征')

        save_path = os.path.join(save_root, f'{label}.txt')
        with open(save_path, 'w') as txt:
            for vector in unique_vectors:
                for i in range(len(vector[0])):
                    txt.write(f"{float(vector[0][i])} ")
                txt.write("\n")