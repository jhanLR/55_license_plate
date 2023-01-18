from cmath import inf
from pyexpat import model
from statistics import mode
import sys
from tkinter import Variable
from turtle import forward
from cv2 import dnn_unregisterLayer
import onnx
import os
import argparse
import numpy as np
import onnxruntime
from tensorboard import summary
import torch
import torchvision
import torchsummary
import torch.onnx
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import io
from torchvision import models
from torchsummary import summary as su1
from torchinfo import summary as su2


def merge_outputs(o1, o2):
    
    boxes_list = []
    confs_list = []
    
    adjust_list1 = []
    adjust_list2 = []
    
    adjust_class_length1 = torch.zeros_like(o2[1])
    adjust_class_length2 = torch.zeros_like(o1[1])
    
    # 11111111 + 0
    adjust_list1.append(o1[1])
    adjust_list1.append(adjust_class_length1)
    # 00000000 + 1
    adjust_list2.append(adjust_class_length2)
    adjust_list2.append(o2[1])
    
    # ACL: [batch, x , num_classes1 + num_classes1]
    o1[1] = torch.cat(adjust_list1, dim=2)
    o2[1] = torch.cat(adjust_list2, dim=2)
    

    boxes_list.append(o1[0])
    confs_list.append(o1[1])
    boxes_list.append(o2[0])
    confs_list.append(o2[1])
        

    # boxes: [batch, num1 + num2 , 1, 4]
    boxes = torch.cat(boxes_list, dim=1)
    # confs: [batch, num1 + num2 ,num_classes]
    confs = torch.cat(confs_list, dim=1)
        
    return [boxes, confs]

class Ensemble_model(nn.Module):
    ##### backbone + head #####
    def __init__(self, model_1_path, model_2_path):
        super().__init__()
                
        # 입력 하나를 자동차, 번호판 동시 검출 앙상블 모델 
        self.model_1 = torch.load(model_1_path)
        self.model_2 = torch.load(model_2_path)
            
    def forward(self, x):
        output_1 = self.model_1(x)
        output_2 = self.model_2(x)        
        
        print('o1', output_1[0])
        print('o2', output_2[0])
        output_t = merge_outputs(output_1, output_2)
        print('ot', output_t[0].size())
        print('ot', output_t[0])
        
        return output_t
        # return output_1, output_2


class Ensemble_model_ocr(nn.Module):
    ##### backbone + head #####
    def __init__(self, model_1_path, model_2_path, model_3_path):
        super().__init__()
                
        # 입력 하나를 자동차, 번호판 동시 검출 앙상블 모델 
        self.model_1 = torch.load(model_1_path) # Vehicle
        self.model_2 = torch.load(model_2_path) # License Plate
        self.model_3 = torch.load(model_3_path) # OCR
            
    def forward(self, x1, x2):
        output_1 = self.model_1(x1)
        output_2 = self.model_2(x1)
        
        output_3 = self.model_3(x2)
        
        print('o1', output_1[0])
        print('o2', output_2[0])
        print('o3', output_3[0])
        output_t = merge_outputs(output_1, output_2)
        print('ot', output_t[0].size())
        print('ot', output_t[0])
        
        return output_t, output_3
        # return output_1, output_2

if __name__ == '__main__':
    # TAS_model 736 416
    # car_plate_model 640 384
    model_1_path = './ensemble_model/TAS_seong_focal_ws3_1.pt'
    model_2_path = './ensemble_model/TAS_car_plate.pt'
    
    model_3_path = './ensemble_model/OCR_test_entire.pt'
    
    output_path = './ensemble_model/ensemble_model_w_ocr.onnx'
    
    model_t = Ensemble_model(model_1_path, model_2_path)
    # model_t = Ensemble_model_ocr(model_1_path, model_2_path, model_3_path)

    batch_size = 1
    input_H = 416
    input_W = 736
    
    dummy_input_1 = torch.randn((batch_size, 3, input_H, input_W), requires_grad=True)
    dummy_input_2 = torch.randn((1, 3, 32, 100), requires_grad=True)

    input_names = ['input']
    output_names = ['boxes_d', 'confs_d', 'boxes_p', 'confs_p']
    # output_names = ['head_0', 'head_1', 'head_2']

    print("Converting to onnx and running ...")
    print("__________________________________")
    
    torch.onnx.export(model_t, dummy_input_1, output_path, opset_version=11)
    # torch.onnx.export(model_t, (dummy_input_1, dummy_input_2), output_path, opset_version=11)




# class Ensemble_TEST(nn.Module):
#     ##### backbone + head #####
#     def __init__(self, n_classes=80, inference=False):
#         super().__init__()
        
#         output_ch = (4 + 1 + n_classes) * 3
        
#         # 입력 하나를 자동차, 번호판 동시 검출 앙상블 모델 
#         self.backbone1 = TEST(n_classes, inference) # ex) 자동차 모델
#         self.backbone2 = TEST(n_classes, inference) # ex) 번호판 모델
        
#         self.head1 = TEST_Head(n_classes, inference)
#         self.head2 = TEST_Head(n_classes, inference)
        
#     def forward(self, x):
#         b11 ,b12, b13 = self.backbone1(x)
#         h1 = self.head1(b11, b12, b13)
        
#         b21, b22, b23 = self.backbone2(x)
#         h2 = self.head2(b21, b22, b23)
        
#         return h1, h2
    
# def create_model_structure(save_path):
#     dummy_input = torch.randn(4, 3, 416, 416)
#     model = Ensemble_TEST(n_classes=10, inference=True)
#     model.eval()
#     jit_model = torch.jit.trace(model, dummy_input)
#     torch.jit.save(jit_model, save_path)
    

# if __name__ == '__main__':
#     save_path = '/home/jhan/laonroad/deeppart/pytorch-YOLOv4/backbone/Ensemble.pt'
#     create_model_structure(save_path)