import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
import onnx
import onnxruntime
import cv2
from PIL import Image

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                    # default='0123456789().JNRW_abcdef가강개걍거겅겨견결경계고과관광굥구금기김깅나남너노논누니다대댜더뎡도동두등디라러로루룰리마머명모무문므미바배뱌버베보부북비사산서성세셔소송수시아악안양어여연영오올용우울원육으을이익인자작저전제조종주중지차처천초추출충층카콜타파평포하허호홀후히ㅣ', help='character label')
                    default='0123456789가나다라마아바사자하거너더러머버서어저허고노두로모보소오조호구누두루무부수우주배x', help='character label')
                    # default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
                    #default='0123456789�??��?��?��마거?��?��?��머버?��?��???고노?��로모보소?��조구?��?��루무�??��?��주하?��?��배공?��?��?���??��?��?��천�???��???구울?���??��광주?��종제주강?��충북충남?��북전?��경북경남경기', help='character label')
                    # default='0123456789().JNRW_abcdef�?강개걍거겅겨견결경계고과�?광굥구금기�??깅나?��?��?��?��?��?��?��????��?��?��?��?��?��?��?��?��?��로루룰리마머명모무문�?미바배뱌버베보�??북비?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��????��?��조종주중�?차처천초추출충층카콜????��?��?��?��?��?��????��?��?��', help='character label')
                    #default='0123456789�?강거경계고�??광구금기�??��?��?��?��?��?��????��?��?��?��?��?��?��로루마머명모무문미바배버보�??북사?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��????��조주중차천초추충카�???��?��?��?��?��?��????��', help='character label')
                    #default='0123456789�?강거경계고�??광구금기�??��?��?��?��?��?��????��?��?��?��?��?��?��로루마머명모무문미바배버보�??북사?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��?��????��?��조종주중차천초추충카????��?��?��?��?��?��???', help='character label')
parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode') # ?��?��?��?���? ????��문자까�?? 구분 �??��
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


print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
         opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
         opt.SequenceModeling, opt.Prediction, opt.saved_model)


savefile = "OCR_test.pt"
image_path = './input/images/7626.jpg'
batch_size = 1
channel_size = opt.input_channel
input_H = opt.imgH
input_W = opt.imgW

model = Model(opt)
model = torch.nn.DataParallel(model).to(device)

model.module.load_state_dict(torch.load(savefile, map_location=device))
model.eval()

# Input
if opt.rgb:
    img_in = Image.open(image_path).convert('RGB')  # for color image
else:
    img_in = Image.open(image_path).convert('L')
    
text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

# Compute
outputs = model(img_in, text_for_pred)
##### torch.max(example) = example.max() #####
# _, preds_index = output_tensor.max(2)
_, preds_index = torch.max(outputs, 2)
print("output_tensor", outputs)
print("preds_index", preds_index)

# compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: (to_numpy(image)), ort_session.get_inputs()[1].name: to_numpy(text_for_pred)}
# ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
# print(to_numpy(dummy_output).size, ort_outs[0].size, len(ort_outs))
# np.testing.assert_allclose(to_numpy(dummy_output), ort_outs[0], rtol=1e-03, atol=1e-05)
