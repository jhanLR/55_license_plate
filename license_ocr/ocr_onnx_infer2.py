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
import time

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset, RawDataset
from model import Model
from test import validation


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



parser = argparse.ArgumentParser()
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--image_folder',default='./input/images/t', help='path to image_folder which contains text images')
parser.add_argument('--rgb', default=True,action='store_true', help='use rgb input')


opt = parser.parse_args()



onnxfile = "OCR_test.onnx"
# image_path = './input/images/8056.jpg'

ort_session = onnxruntime.InferenceSession(onnxfile)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


IN_IMAGE_H = ort_session.get_inputs()[0].shape[2]
IN_IMAGE_W = ort_session.get_inputs()[0].shape[3]

# # Input
# image_src = cv2.imread(image_path)
# resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
# img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
# # print('fff', img_in.shape)
# img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
# img_in = np.expand_dims(img_in, axis=0)
# # print('fff', img_in.shape)
# img_in /= 255.0
# print("!!!!!!!!!!!!!!", img_in)
# print("Shape of the network input: ", img_in.shape)

AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
demo_loader = torch.utils.data.DataLoader(
    demo_data, batch_size=opt.batch_size,
    shuffle=False,
    num_workers=int(opt.workers),
    collate_fn=AlignCollate_demo, pin_memory=True)

print("?????????", demo_data)
print("?????????", demo_loader)

for image_tensors, image_path_list in demo_loader:
    batch_size = image_tensors.size(0)
    # image = image_tensors.to(device)
    print("image", image_tensors.size())
    image = image_tensors.numpy()
    print("image", image.shape)
    

    # Compute
    input_name = ort_session.get_inputs()[0].name
    st = time.time()
    outputs = ort_session.run(None, {input_name: image})
    output_tensor = torch.tensor(outputs[0])
    et1 = time.time()
    ##### torch.max(example) = example.max() #####
    # _, preds_index = output_tensor.max(2)
    # _, preds_index = torch.max(output_tensor, 2)
    et2 = time.time()
    print("output_tensor", output_tensor)
    # print("preds_index", preds_index)

    print("s-e1 \t: \t", et1 - st)
    print("e1-e2 \t: \t", et2 - et1)
    print("s-e2 \t: \t", et2 - st)

    # compute ONNX Runtime output prediction
    # ort_inputs = {ort_session.get_inputs()[0].name: (to_numpy(image)), ort_session.get_inputs()[1].name: to_numpy(text_for_pred)}
    # ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    # print(to_numpy(dummy_output).size, ort_outs[0].size, len(ort_outs))
    # np.testing.assert_allclose(to_numpy(dummy_output), ort_outs[0], rtol=1e-03, atol=1e-05)
