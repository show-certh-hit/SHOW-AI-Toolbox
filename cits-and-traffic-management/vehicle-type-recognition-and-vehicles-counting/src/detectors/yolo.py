'''
Performs vehicle detection using models created with the YOLO (You Only Look Once) neural net.
https://pjreddie.com/darknet/yolo/
'''

import ast
import os
import cv2


with open(os.getenv('YOLO_CLASSES_PATH'), 'r') as classes_file:
    CLASSES = dict(enumerate([line.strip() for line in classes_file.readlines()]))
with open(os.getenv('YOLO_CLASSES_OF_INTEREST_PATH'), 'r') as coi_file:
    CLASSES_OF_INTEREST = tuple([line.strip() for line in coi_file.readlines()])

use_gpu = ast.literal_eval(os.getenv('ENABLE_GPU_ACCELERATION'))

# initialize model with weights and config
if use_gpu:
    from pydarknet import Detector
    net = Detector(bytes(os.getenv('YOLO_CONFIG_PATH'), encoding='utf-8'),
                   bytes(os.getenv('YOLO_WEIGHTS_PATH'), encoding='utf-8'),
                   0,
                   bytes(os.getenv('YOLO_DATA_PATH'), encoding='utf-8'))
else:
    net = cv2.dnn.readNet(os.getenv('YOLO_WEIGHTS_PATH'), os.getenv('YOLO_CONFIG_PATH'))

def get_bounding_boxes_cpu(image):
    import numpy as np

    # create image blob
    scale = 0.00392
    image_blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    # detect objects
    net.setInput(image_blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    _classes = []
    _confidences = []
    boxes = []
    conf_threshold = float(os.getenv('YOLO_CONFIDENCE_THRESHOLD'))
    nms_threshold = 0.4

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and CLASSES[class_id] in CLASSES_OF_INTEREST:
                width = image.shape[1]
                height = image.shape[0]
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                _classes.append(CLASSES[class_id])
                _confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, _confidences, conf_threshold, nms_threshold)

    _bounding_boxes = []
    for i in indices:
        i = i[0]
        _bounding_boxes.append(boxes[i])

    return _bounding_boxes, _classes, _confidences

def get_bounding_boxes_gpu(img):
    from pydarknet import Image

    img_darknet = Image(img)
    results = net.detect(img_darknet)

    _bounding_boxes, _classes, _confidences = [], [], []
    for cat, score, bounds in results:
        _class = str(cat.decode('utf-8'))
        if _class in CLASSES_OF_INTEREST:
            _bounding_boxes.append(bounds)
            _classes.append(_class)
            _confidences.append(score)

    return _bounding_boxes, _classes, _confidences

def get_bounding_boxes(image):
    '''
    Return a list of bounding boxes of vehicles detected,
    their classes and the confidences of the detections made.
    '''
    
    if use_gpu:
        return get_bounding_boxes_gpu(image)
    else:
        return get_bounding_boxes_cpu(image)