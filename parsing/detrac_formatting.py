import json
import os
import xml.etree.ElementTree as ET
import random
import cv2
import os
from PIL import Image

traffic_labels = ['car', 'van', 'bus', 'others']
# traffic_labels = ['vehicle']  # DETRAC dataset has multiple labels for vehicles
traffic_label_map = {k: v + 1 for v, k in enumerate(traffic_labels)}
traffic_label_map['__background__'] = 0
rev_traffic_label_map = {v: k for k, v in traffic_label_map.items()}  # Inverse mapping

'''
There are 82085 training images containing a total of 594555 objects.
Files have been saved to /media/keyi/Data/Research/course_project/AdvancedCV_2020/AdvanceCV_project/data/DETRAC.
There are 56167 validation images containing a total of 658859 objects.
Files have been saved to /media/keyi/Data/Research/course_project/AdvancedCV_2020/AdvanceCV_project/data/DETRAC.
'''


def parse_annotation(annotation_path, image_folder, down_sample=False):
    tree = ET.parse(annotation_path)

    root = tree.getroot()
    object_list = list()
    image_paths = list()

    # parse ignored regions
    # ignore_regions = list()
    # region_list = root.find('ignored_region')
    # for region in region_list.iter('box'):
    #     left = float(region.attrib['left'])
    #     top = float(region.attrib['top'])
    #     width = float(region.attrib['width'])
    #     height = float(region.attrib['height'])
    #     xmin = max(int(left), 0)
    #     ymin = max(int(top), 0)
    #     xmax = int(left + width)
    #     ymax = int(top + height)
    #     ignore_regions.append([xmin, ymin, xmax, ymax])

    for objects in root.iter('frame'):
        if random.random() > 0.15 and down_sample:
            continue
        boxes = list()
        labels = list()
        class_names = list()
        # difficulties = list()
        # occlusions = list()

        frame_id = int(objects.attrib['num'])
        image_name = 'img{:05d}.jpg'.format(frame_id)
        # unique_image_id = image_folder.split('/')[-1] + '_' + objects.attrib['num']  # This has to be a number
        unique_image_id = int(image_folder.split('/')[-1].split('_')[-1] + objects.attrib['num'])
        # print('unique_image_id:', unique_image_id)
        # print(image_folder.split('/')[-1].split('_')[-1])
        # exit()

        target_list = objects.find('target_list')
        for target in target_list:
            uni_name = target.find('attribute').attrib['vehicle_type']
            if uni_name not in traffic_labels:
                print('Class labels not for vehicles: ', uni_name)
                continue
            else:
                uni_label = traffic_label_map[uni_name]

            bbox = target.find('box').attrib
            left = float(bbox['left'])
            top = float(bbox['top'])
            width = float(bbox['width'])
            height = float(bbox['height'])
            xmin = max(int(left), 0)
            ymin = max(int(top), 0)
            xmax = int(left + width)
            ymax = int(top + height)
            if xmin + 2 > xmax or ymin + 2 > ymax:
                print('Invalid bboxes: ', os.path.join(image_folder, image_name))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(uni_label)
            class_names.append(uni_name)

        if len(boxes) == 0:
            print('Images with no objects: ', os.path.join(image_folder, image_name))
            continue

        image_paths.append(os.path.join(image_folder, image_name))
        object_list.append({'bboxes': boxes, 'labels': labels, 'class_names': class_names,
                            'image_id': unique_image_id, 'image_path': image_paths[-1]})

    assert len(boxes) == len(labels) == len(class_names)
    assert len(object_list) == len(image_paths)

    return object_list, image_paths


def create_data_lists_detrac(root_path, output_folder):
    # training data
    dataType = 'Train'
    train_images = list()
    train_objects = list()
    n_object = 0

    annotation_folder = 'DETRAC-{}-Annotations-XML'.format(dataType)
    annotation_path = os.path.join(root_path, annotation_folder)
    if not os.path.exists(annotation_path):
        print('annotation_path not exist')
        raise FileNotFoundError

    image_folder = 'Insight-MVT_Annotation_{}'.format(dataType)
    image_root = os.path.join(root_path, image_folder)  # under: sequence_name/image_name
    count = 0
    for video in os.listdir(annotation_path):
        # print('Train data: {}/{}'.format(count + 1, len(os.listdir(annotation_path))))
        if video.endswith('.xml'):
            objects, image_frames_path = parse_annotation(os.path.join(annotation_path, video),
                                                          os.path.join(image_root, video.strip('.xml')))

            if len(objects) == 0:
                continue
            else:
                for obj in objects:
                    n_object += len(obj['bboxes'])

            train_objects += objects
            train_images += image_frames_path
            count += 1

    assert len(train_objects) == len(train_images)
    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_object, os.path.abspath(output_folder)))

    # test data
    dataType = 'Test'
    test_images = list()
    test_objects = list()
    n_object = 0

    annotation_folder = 'DETRAC-{}-Annotations-XML'.format(dataType)
    annotation_path = os.path.join(root_path, annotation_folder)
    if not os.path.exists(annotation_path):
        print('annotation_path not exist')
        raise FileNotFoundError

    image_folder = 'Insight-MVT_Annotation_{}'.format(dataType)
    image_root = os.path.join(root_path, image_folder)  # under: sequence_name/image_name
    count = 0
    for video in os.listdir(annotation_path):
        # print('Test data: {}/{}'.format(count + 1, len(os.listdir(annotation_path))))
        if video.endswith('.xml'):

            objects, image_frames_path = parse_annotation(os.path.join(annotation_path, video),
                                                          os.path.join(image_root, video.strip('.xml')), True)

            if len(objects) == 0:
                continue
            else:
                for obj in objects:
                    n_object += len(obj['bboxes'])

            test_objects += objects
            test_images += image_frames_path
            count += 1

    assert len(test_objects) == len(test_images)
    # Save to file
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d validation images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_object, os.path.abspath(output_folder)))


if __name__ == '__main__':
    """
    The root_path is the root folder of the downloaded DETRAC dataset, the structure may look like
    DETRAC:
        - DETRAC-Test-Annotations-XML
        - DETRAC-Test-Det
        - DETRAC-Train-Annotations-XML
        - Insight-MVT_Annotation_Test
        - Insight-MVT_Annotation_Train
        ...
    """
    root_path = '/home/keyi/Documents/Data/DETRAC'
    output_folder = '/home/keyi/Documents/research/code/PointCenterNet_project/Data/DETRAC'

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    create_data_lists_detrac(root_path, output_folder)
