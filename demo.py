import os
from object_aware_crop import object_aware_crop
IMAGE_READING_PATH = [
    'dataset_1/image/',
    'dataset_2/image/',
    'dataset_3/image/',
]
IMAGE_SAVING_PATH = [
    'dataset_1_patch/image/',
    'dataset_2_patch/image/',
    'dataset_3_patch/image/',
]
LABEL_READING_PATH = [
    'dataset_1/label/',
    'dataset_2/label/',
    'dataset_3/label/',
]
LABEL_SAVING_PATH = [
    'dataset_1_patch/label/',
    'dataset_2_patch/label/',
    'dataset_3_patch/label/',
]

for i in range(0,11):
    print(IMAGE_READING_PATH[i])
    print(IMAGE_SAVING_PATH[i])
    print(LABEL_READING_PATH[i])
    print(LABEL_SAVING_PATH[i])
    image_path_list=[]
    label_path_list=[]
    # Get all the images and labels in a dataset
    for file in os.listdir(IMAGE_READING_PATH[i]):
        image_path_list.append(IMAGE_READING_PATH[i]+file)
        image_path_list.sort()

    for file in os.listdir(LABEL_READING_PATH[i]):
        label_path_list.append(LABEL_READING_PATH[i]+file)
        label_path_list.sort()
    for j in range(len(label_path_list)):
        print(image_path_list[j])
        print(label_path_list[j])
        object_aware_crop(image_path_list[j],label_path_list[j], IMAGE_SAVING_PATH[i], LABEL_SAVING_PATH[i])