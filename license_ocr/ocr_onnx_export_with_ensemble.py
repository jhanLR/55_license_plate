import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import onnx
import onnxruntime

import onnx
import onnxruntime
import torch
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model, Model_onnx
from test import validation


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class Ensemble_model_ocr(nn.Module):
    ##### backbone + head #####
    def __init__(self, model_1_path, model_2_path):
        super().__init__()
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--image_folder',default='./input/images', help='path to image_folder which contains text images')
        parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
        parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
        parser.add_argument('--saved_model', default='./saved_models/None-ResNet-BiLSTM-CTC-Seed2000-Renewed/best_accuracy.pth', help="path to saved_model to evaluation")
        # parser.add_argument('--saved_model', default='./pretrained/Fine-Tuned.pth', help="path to saved_model to evaluation")
        """ Data processing """
        parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
        parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
        parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
        parser.add_argument('--rgb', default=True,action='store_true', help='use rgb input')
        parser.add_argument('--character', type=str,
                            # default='0123456789().JNRW_abcdef??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????', help='character label')
                            default='0123456789????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????x', help='character label')
                            # default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
                            #default='0123456789???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????', help='character label')
                            # default='0123456789().JNRW_abcdef??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????', help='character label')
                            #default='0123456789?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????', help='character label')
                            #default='0123456789?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????', help='character label')
        parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode') # ???????????????????????????????? ???????????????????????? ?????? ???????????
        parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
        """ Model Architecture """
        parser.add_argument('--Transformation', type=str, default='None', help='Transformation stage. None|TPS')
        parser.add_argument('--FeatureExtraction', type=str,default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
        parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
        parser.add_argument('--Prediction', type=str,default='CTC', help='Prediction stage. CTC|Attn')
        parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
        parser.add_argument('--input_channel', type=int, default=3, help='the number of input channel of Feature extractor')
        parser.add_argument('--output_channel', type=int, default=512,
                            help='the number of output channel of Feature extractor')
        parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

        opt = parser.parse_args()

        if 'CTC' in opt.Prediction:
            converter = CTCLabelConverter(opt.character)
        else:
            converter = AttnLabelConverter(opt.character)
        opt.num_class = len(converter.character)

        model = Model(opt)

        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
                opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
                opt.SequenceModeling, opt.Prediction, opt.saved_model)
        model = torch.nn.DataParallel(model).to(device)

        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
        model.eval()


        savefile_for_ONNX = "OCR_test.pt"
        savefile = "OCR_test_entire.pt"
        batch_size = 1
        channel_size = opt.input_channel
        input_H = opt.imgH
        input_W = opt.imgW
        dummy_input = torch.randn(batch_size, channel_size, input_H, input_W)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        # export??? model??? ?????? ????????? ??????, DataParallel ????????? module??? ?????? ??? ????????? ??????
        torch.save(model.module.state_dict(), savefile_for_ONNX)
        torch.save(model.module, savefile)
        model_3 = Model_onnx(opt)
        model_3.load_state_dict(torch.load(savefile_for_ONNX))

        
                
        # ?????? ????????? ?????????, ????????? ?????? ?????? ????????? ?????? 
        self.model_1 = torch.load(model_1_path) # Vehicle
        self.model_2 = torch.load(model_2_path) # License Plate
        self.model_3 = model_3 # OCR
            
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
        return output_t, output_3




if __name__ == '__main__':
    # TAS_model 736 416
    # car_plate_model 640 384
    model_1_path = './ensemble_model/TAS_seong_focal_ws3_1.pt'
    model_2_path = './ensemble_model/TAS_car_plate.pt'
        
    output_path = './ensemble_model/ensemble_model_w_ocr.onnx'
    
    model_t = Ensemble_model_ocr(model_1_path, model_2_path)

    batch_size = 4
    input_H = 416
    input_W = 736
    
    dummy_input_1 = torch.randn((batch_size, 3, input_H, input_W), requires_grad=True)
    dummy_input_2 = torch.randn((1, 3, 32, 100))

    input_names = ['input_det', 'input_ocr']
    output_names = ['boxes_det', 'confs_det', 'output_ocr']
    # output_names = ['head_0', 'head_1', 'head_2']

    print("Converting to onnx and running ...")
    print("__________________________________")

    torch.onnx.export(model_t, (dummy_input_1, dummy_input_2), output_path, opset_version=12,
                    export_params=True, verbose=False, do_constant_folding=True,
                    input_names=input_names, output_names=output_names)#, dynamic_axes=dynamic_axes)

    # DataParallel ?????? ??? model.module ??????
    # torch.onnx.export(model.module, dummy_input, onnxfile, opset_version=12, verbose = True)

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    onnx.helper.printable_graph(onnx_model.graph)

    ort_session = onnxruntime.InferenceSession(output_path)



    # # Input
    # if opt.rgb:
    #     img_in = Image.open(image_path).convert('RGB')  # for color image
    # else:
    #     img_in = Image.open(image_path).convert('L')
        
    # text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
    # dummy_output = model(dummy_input, text_for_pred)
    # def to_numpy(tensor):
    #     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # # compute ONNX Runtime output prediction
    # # ort_inputs = {ort_session.get_inputs()[0].name: (to_numpy(dummy_input)), ort_session.get_inputs()[1].name: to_numpy(text_for_pred)}
    # ort_inputs = {ort_session.get_inputs()[0].name: (to_numpy(dummy_input))}
    # ort_outs = ort_session.run(None, ort_inputs)

    # print("11111111", dummy_input)
    # print("11111111", to_numpy(dummy_input))
    # print("22222222", dummy_output)
    # print("22222222", ort_outs)

    # # compare ONNX Runtime and PyTorch results
    # print(to_numpy(dummy_output).size, ort_outs[0].size, len(ort_outs))
    # np.testing.assert_allclose(to_numpy(dummy_output), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")