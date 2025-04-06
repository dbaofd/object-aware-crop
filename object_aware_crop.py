from PIL import Image
import numpy as np
import random
import math
import copy

# Configurations
H = 1920
W = 2560
CROP_SIZE = 448


def get_boxes_from_label(label_path):
    """
    Given a label file path, read the file get the bounding boxes.
    Convert the bounding boxes from percentage to actual size.
    Args:
        label_path: string. Path of label file.
    Returns:
        boxes: list of lists. Contains all the bounding boxes in a label file.
    """
    boxes = []
    with open(label_path) as f:
        for label in f.readlines():
            label = label.replace("\n", "").split()
            if len(label) != 5:
                break
            class_label, x, y, width, height = [
                float(x) if float(x) != int(float(x)) else int(x)
                for x in label
            ]
            # Convert the x,y,width,height to actual number as in yolo label these are percentages
            #             if class_label==0:
            x = x * W
            y = y * H
            width = width * W
            height = height * H
            boxes.append([class_label, x, y, width, height])
    return boxes


def box_in_patch_coordinate_range(box):
    """
    Given a bounding box from the label, generate a random patch that contains the box.
    This function will generate the x axis and y axis range for the top-left corner of the patch.
    There is a formula to calculate the range, draw some graphs then you will know.
    Args:
        box: list. class_label, x, y, width, height.
    Returns:
        x_start: int.
        x_end: int.
        y_start: int.
        y_end: int.
    """
    x_start = max(0, math.ceil(box[1]) + math.ceil(0.5 * box[3]) - CROP_SIZE)
    x_end = min(math.ceil(box[1]) - math.ceil(0.5 * box[3]), W - CROP_SIZE)
    y_start = max(0, math.ceil(box[2]) + math.ceil(0.5 * box[4]) - CROP_SIZE)
    y_end = min(math.ceil(box[2]) - math.ceil(0.5 * box[4]), H - CROP_SIZE)
    return x_start, x_end, y_start, y_end


def get_boxes_in_patch(boxes, boxes_flag, xp, yp):
    """
    Return all the boxes in the patch. Minimum one box in the patch.
    Here is a important thing to notice. It caused a bug before. It is about the python reference.
    If we append box in to boxes_in_patch without using deep copy, then the later modification on
    the boxes_in_patch will cause the modification on the boxes.This is not accepted, we don't want
    the boxes to be modified. Otherwise, it will cause a bug when recalling this function.

    Args:
        boxes: list of lists. All the bounding boxes in a label file.
        boxes_flag: np array. Indicate if cooresponding patches of a particular
                    bounding box has been generated or not.
        xp, yp: int. top-left corner coordinate of the randomly generated patch.
    Returns:
        boxes_in_patch: list of lists. All the bounding boxes that are fully in the given patch.
        boxes_flag: modified boxes_flag
    """
    boxes_in_patch = []
    for i, box in enumerate(boxes):
        x_start, x_end, y_start, y_end = box_in_patch_coordinate_range(box)
        if xp >= x_start and xp <= x_end and yp >= y_start and yp <= y_end:  # The box is fully in the path
            new_box = copy.deepcopy(box)
            boxes_in_patch.append(new_box)
            boxes_flag[i] = 1
        elif box[1] >= xp and box[1] <= xp + CROP_SIZE and box[2] >= yp and box[
            2] <= yp + CROP_SIZE:  # The box centre is in the patch
            return None, None  # The box is partially in the patch, return none to indicate.
    # Return the boxes in the patch. There is no boxes partially in the patch.
    return boxes_in_patch, boxes_flag


def convert_boxes(boxes_in_patch, xp, yp):
    """
    Make x,y,w,h to be relative to patch.

    Args:
        boxes_in_patch: list of lists. All the bounding boxes that fully fall in the given patch.
        xp, yp: int. top-left corner coordinate of the randomly generated patch.
    Returns:
        boxes_in_patch: list of lists. Modified boxes_in_patch.
    """
    for box in boxes_in_patch:
        box[1] = (box[1] - xp) / CROP_SIZE
        box[2] = (box[2] - yp) / CROP_SIZE
        box[3] = box[3] / CROP_SIZE
        box[4] = box[4] / CROP_SIZE
    return boxes_in_patch


def save_image(img, patch_number, saving_path, file_name):
    """
    Save image patch.
    Args:
        img: PIL.Image, image patch to be saved.
        patch_number: int, number of the patch.
        saving_path: string, saving patch.
        file_name: string, original name of the image.
    Returns:
        None.
    """
    img.save(saving_path + file_name + '_patch_' + str(patch_number) + '.jpg', 'JPEG', quality=95)


def save_label(boxes_in_patch, patch_number, saving_path, file_name):
    """
    Save patch label.
    Args:
        boxes_in_patch: list of lists, all the boxes that fall fully in the given patch.
        patch_number: int, number of the patch.
        saving_path: string, saving patch.
        file_name: string, original name of the image.
    Returns:
        None.
    """
    f = open(saving_path + file_name + '_patch_' + str(patch_number) + ".txt", "w")
    for box in boxes_in_patch:
        new_line = str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]) + " " + str(box[4]) + "\n"
        f.write(new_line)
    f.close()


def object_aware_crop(image_path, label_path, image_saving_path, label_saving_path, extra_patch_num=0,
                      label_format="yolo"):
    """
    Save image patches, patch labels and patch coordinate.

    Args:
        image_path: string
        label_path: string
        image_saving_path: string
        label_saving_path: string
        #coordinate_saving_path: string.
        is_training_set: boolean
        label_format: string, label format.
    Returns:
        None.
    """
    image_name = image_path.split("/")[-1].split(".")[0]  # Get the image and label name
    label_name = label_path.split("/")[-1].split(".")[0]
    # Open the label file, put all the boxes in a list
    boxes = get_boxes_from_label(label_path)  # Get boxes from a given label file.
    img = Image.open(image_path)
    boxes_flag = np.zeros(len(boxes))  # To indicate if a box has already been included in an patch or not.
    current_patch_number = 0  # There will be many patches from an image, to name patch, the patch number is needed
    for i, box in enumerate(boxes):  # Traverse all the boxes in the image.
        if boxes_flag[i] == 0:  # If the current box is not included in any saved patch.
            x_start, x_end, y_start, y_end = box_in_patch_coordinate_range(box)
            print(current_patch_number)
            while 1+extra_patch_num >= 1:
                extra_patch_num -= 1
                while True:
                    xp = random.randint(x_start, x_end)
                    yp = random.randint(y_start, y_end)  # (xp yp) is the top-lef corner coordinate of the patch.
                    # Randomly generate (xp, yp) within the coordinate range.
                    boxes_in_patch, new_boxes_flag = get_boxes_in_patch(boxes, boxes_flag, xp, yp)  # Get
                    # all the boxes that are fully in the given patch.
                    # If boxes_in_patch is None, it means there is box partially located in the
                    # given patch. Need to regenerate patch coordinate.
                    if boxes_in_patch != None:
                        boxes_flag = new_boxes_flag  # Update the indicator array.
                        patch_img = img.crop((xp, yp, xp + CROP_SIZE, yp + CROP_SIZE))  # Get the patch.
                        save_image(patch_img, current_patch_number, image_saving_path, image_name)
                        boxes_in_patch = convert_boxes(boxes_in_patch, xp, yp)
                        save_label(boxes_in_patch, current_patch_number, label_saving_path, label_name)
                        # save_patch_coordinate(xp, yp, current_patch_number, coordinate_saving_path, label_name)
                        current_patch_number += 1
                        break
                    # If boxes_in_patch is None, it means there is a box partially located in the
                    # given patch. Continue the loop, regenerate the patch. The goal is
                    # to generate a patch in which some boxes are fully located and no box
                    # is particially located.
                    print("Genetating random coordinate again!")
