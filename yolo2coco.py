"""
##############################################################################################
##     This is the convert tool for YOLO format datasets labels to COCO datasets label      ##
##           Only for object detect task, no segmentation nor keypoint use                  ##
## Input YOLO labels folder, Output COCO label destination folder, Flag Train or Val label  ##
##                            Develop by Javier Zhang @ Shyechih                            ##
##############################################################################################
"""

import os
import cv2
import datetime
import json
import glob
from detectron2.structures import BoxMode
import argparse


def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[1],
            "height": image_size[0],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }
    return image_info


def create_annotation_info(annotation_id, image_id, category_id, is_crowd,
                           area, bounding_box, segmentation):
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": is_crowd,
        "bbox_mode": BoxMode.XYWH_ABS,
        "area": area,# float
        "bbox": bounding_box,# [x,y,width,height]
        "segmentation": segmentation# [polygon]
    }
    return annotation_info


# def get_segmenation(coord_x, coord_y):
#     seg = []
#     for x, y in zip(coord_x, coord_y):
#         seg.append(x)
#         seg.append(y)
#     return [seg]


def convert(files):
    """
       :param imgdir: directory for your images
       :param annpath: path for your annotations
       :return: coco_output is a dictionary of coco style which you could dump it into a json file
       as for keywords 'info','licenses','categories',you should modify them manually
       """

    coco_output = {}

    coco_output['info'] = {
        "description": "COCO Format Dataset",
        "url": "",
        "version": "1.0",
        "year": 2020,
        "contributor": "Javier Zhang",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    coco_output['licenses'] = [
        {
            "id": 1,
            "name": "What The Fucking License",
            "url": "http://whatever.org/licenses/fucking/2.0/"
        }
    ]

##########################################################################
#                  Modify your own categories here                       #
##########################################################################

    coco_output['categories'] = [
        {
            'id': 1,
            'name': 'P',
            'supercategory': 'PET',
        },
        {
            'id': 2,
            'name': 'C',
            'supercategory': 'PET',
        },
        {
            'id': 3,
            'name': 'O',
            'supercategory': 'PET',
        },
        {
            'id': 4,
            'name': 'S',
            'supercategory': 'PET',
        },
        {
            'id': 5,
            'name': 'Ch',
            'supercategory': 'PET',
        },
        {
            'id': 6,
            'name': 'Ot',
            'supercategory': 'PET',
        },
        {
            'id': 7,
            'name': 'T',
            'supercategory': 'PET',
        }
    ]

    coco_output['images'] = []
    coco_output['annotations'] = []

    ann_id = 0
    img_id = 0

    for file in files:
        file_name = os.path.basename(file)
        imgdir = os.path.join(file.split('labels')[0], 'images/', file_name.split('.')[0]+'.jpg')
        img = cv2.imread(imgdir)
        image_info = create_image_info(img_id, file_name.replace('txt', 'jpg'), img.shape[:2])
        coco_output['images'].append(image_info)
        # print(img.shape[1])
        with open(file) as f:
            data = f.readlines()
            for line in data:
                cat_id = int(line.split(' ')[0]) + 1
                width = float(line.split(' ')[3]) * img.shape[1]
                height = float(line.split(' ')[4].strip('\n')) * img.shape[0]
                x = float(line.split(' ')[1]) * img.shape[1] - width/2
                y = float(line.split(' ')[2]) * img.shape[0] - height/2
                box = [int(x), int(y), int(width), int(height)]
                ann_info = create_annotation_info(ann_id, img_id, cat_id, 0, 0, box, 0)
                coco_output['annotations'].append(ann_info)
                ann_id = ann_id + 1
        img_id += 1
    return coco_output


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Input YOLO format label output coco format label.')
    parser.add_argument('--input', default='/media/shyechih/data/stage2_txt/PET_1_exam/labels',
                        help='Yolo format label folder, .txt files')
    parser.add_argument('--output', default='./datasets/test/annotations',
                        help='coco format label file stored folder, .json file')
    parser.add_argument('--flag', default='val', type=str, help='mode: train or val')
    args = parser.parse_args()

    input_path = glob.glob(os.path.join(args.input, '*.txt'))
    result = convert(input_path)
    output_path = os.path.join(args.output, f'{args.flag}.json')

    with open(output_path, 'w') as file_json:
        json.dump(result, file_json)
