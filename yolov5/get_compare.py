import torch
import os
import cv2
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from compare_ import compare_vector

def compare_database(vector, root_path, face_lib):
    name_label_path = os.path.join(root_path, 'name_label.txt')
    label_name_list = []
    with open(name_label_path, 'r') as f:
        msg_list = f.readlines()
        for msg in msg_list:
            data = msg.strip().split(' ')
            label_name_list.append([int(data[0]), data[1]])

    sorted_list = sorted(label_name_list, key=lambda x: x[0])
    print(sorted_list)
    # exit()

    max_cosa = -1
    max_label = ''

    my_max_cosa = -1

    for file_name in os.listdir(face_lib):
        file_path = os.path.join(face_lib, file_name)
        index = file_name.rfind('.txt')
        label_name = file_name[:index]

        name_index = None
        for i, sub_list in enumerate(sorted_list):
            if sub_list[0] == int(label_name):
                name_index = i
        face_name = sorted_list[name_index][1]
        # print(f'label_name{int(label_name)}, name_index{name_index}')
        # print(label_name)
        with open(file_path, 'r') as f:
            msg_list = f.readlines()

            for msg in msg_list:
                # data = msg.split(' ').remove('\n')
                data = msg.strip().split(' ')

                obj_vector = [float(i) for i in data]
                # for i in data:
                #     obj_vector.append(float(i))

                obj_vector = torch.tensor(obj_vector).unsqueeze(0).to('cuda')
                # obj_vector = obj_vector.unsqueeze(0)

                # exit()
                cosa = compare_vector(vector, obj_vector)

                if face_name == '王平':
                    if cosa > my_max_cosa:
                        my_max_cosa = cosa
                if cosa > max_cosa:
                    max_cosa = cosa
                    max_label = label_name



        index2 = None
        for i, sub_list in enumerate(sorted_list):
            if sub_list[0] == int(max_label):
                index2 = i
        # print(f'max_label{int(max_label)}, index2{index2}')
        if index2 is not None:
            print(max_cosa, max_label, sorted_list[index2][1])
            print(f'我的最大特征相似度为{my_max_cosa}')


    if max_cosa > 0.65:
        return sorted_list[index2][1]
    else:
        return None


def face_recognition(net, img, device, root_path, face_lib_path):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])
    image = Image.fromarray(img)
    img_data = transform(image)
    img_data = img_data.unsqueeze(0).to(device)
    feature = net.get_feature(img_data)
    # compare_database(feature[0])
    name = compare_database(feature, root_path, face_lib_path)
    print(name)
    return name