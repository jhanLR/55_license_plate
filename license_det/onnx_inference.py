import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime
import torch
import torch.onnx
import glob
from PIL import Image
import torchvision.transforms as transforms
import argparse
import imghdr
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
from sympy import primitive
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

from PIL import Image


def onnx_infer(onnx_path, image_path):
    # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    providers = ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    print("The model expects input shape: ", session.get_inputs()[0].shape)
    
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]
    
    print(IN_IMAGE_H, IN_IMAGE_W)
    
    img = Image.open(image_path)

    resize = transforms.Resize([IN_IMAGE_H, IN_IMAGE_W])
    img_in = resize(img)

    img_ycbcr = img_in.convert('YCbCr')
    # img_y, img_cb, img_cr = img_ycbcr.split()

    to_tensor = transforms.ToTensor()
    img_in = to_tensor(img_ycbcr)
    img_in.unsqueeze_(0)
    print("Shape of the network input: ", img_in.shape)
        
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    # Compute
    input_name = session.get_inputs()[0].name

    t1 = time_synchronized()
    outputs = session.run(None, {input_name: to_numpy(img_in)})

    pred = non_max_suppression(outputs[0], conf_thres=0.5, iou_thres=0.5)
    t2 = time_synchronized()
    
    print('!@$!@$', pred)
    
    def load_classes(path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  # filter removes empty strings (such as last line)

    
    names = load_classes('data/car_plate/data_car_plate.names')
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # Process detections
    for i, det in enumerate(pred):  # detections per image

        s = ''
        s += '%gx%g ' % img_in.shape[2:]  # print string
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=3)

                cv2.imwrite('./test.jpg', img)
                                        

        # Print time (inference + NMS)
        print('%sDone. (%.3fs)' % (s, t2 - t1))



    
if __name__ == '__main__':
    
    onnx_file = 'best_f.onnx'
    image_path = 'C-220712_07_CR14_03_A0895.jpg'
    
    onnx_infer(onnx_file, image_path)