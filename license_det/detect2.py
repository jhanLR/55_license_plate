import argparse
import imghdr
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box, plot_one_box2
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

from PIL import Image

from license_plate.detect2 import *

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Darknet(cfg, imgsz).cuda()
    try:
        # model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
        #model = attempt_load(weights, map_location=device)  # load FP32 model
        #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    except:
        # load_darknet_weights(model, weights[0])
        load_darknet_weights(model, weights)
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, auto_size=64)

    crop_count = 0

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        
                        ####################################################################
                        # crop and save img about bbox
                        ####################################################################
                        print((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
                        xx = int(xyxy[0])
                        yy = int(xyxy[1])
                        ww = int(xyxy[2]) - int(xyxy[0])
                        hh = int(xyxy[3]) - int(xyxy[1])
                        crop_img = im0[yy:yy+hh, xx:xx+ww]
                        print(im0.dtype)
                        crop_count += 1
                        crop_save_path =  "./inference/crop_img/" + str(crop_count) +".jpg"
                        print(crop_save_path)
                        cv2.imwrite(crop_save_path, crop_img, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
                        ####################################################################
                        
                        ####################################################################
                        # OCR : car_plate
                        ####################################################################
                        parser_cp = argparse.ArgumentParser()
                        # parser_cp.add_argument('--image_folder',default='./license_plate/input/car_plate_label_data/test_images', help='path to image_folder which contains text images')
                        parser_cp.add_argument('--workers', type=int, help='number of data loading workers', default=0)
                        parser_cp.add_argument('--batch_size', type=int, default=192, help='input batch size')
                        #parser_cp.add_argument('--saved_model', default='./license_plate/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1235-Renewed/best_accuracy.pth', help="path to saved_model to evaluation")
                        parser_cp.add_argument('--saved_model', default='./license_plate/pretrained/Fine-Tuned.pth', help="path to saved_model to evaluation")
                        """ Data processing """
                        parser_cp.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
                        parser_cp.add_argument('--imgH', type=int, default=32, help='the height of the input image')
                        parser_cp.add_argument('--imgW', type=int, default=100, help='the width of the input image')
                        parser_cp.add_argument('--rgb', action='store_true', help='use rgb input')
                        parser_cp.add_argument('--character', type=str,
                                            default='0123456789().JNRW_abcdef가강개걍거겅겨견결경계고과관광굥구금기김깅나남너노논누니다대댜더뎡도동두등디라러로루룰리마머명모무문므미바배뱌버베보부북비사산서성세셔소송수시아악안양어여연영오올용우울원육으을이익인자작저전제조종주중지차처천초추출충층카콜타파평포하허호홀후히ㅣ', help='character label')
                                            # default='0123456789가나다라마아바사자하거너더러머버서어저허고노두로모보소오조호구누두루무부수우주배x', help='character label')
                                            # default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
                                            #default='0123456789???????????����?????????�ӹ�?????????����???�θ𺸼�???����??????�繫????????����??????���????????????????????õ?????????����????????����???�����ְ�???����泲???����???��ϰ泲���', help='character label')
                                            # default='0123456789().JNRW_abcdef??�������Űϰܰ߰������??�������ݱ�???�볪??????????????????????????????????????????????????????�η�긮���Ӹ��𹫹�??�̹ٹ�������???�Ϻ�?????????????????????????????????????????????????????????????????????????????????????????????????????????��������??��óõ����������ī��?????????????????????????????????', help='character label')
                                            #default='0123456789??���Ű���???�����ݱ�????????????????????????????????????????????�η縶�Ӹ��𹫹��̹ٹ����???�ϻ�?????????????????????????????????????????????????????????????????????��������õ������ī???????????????????????????', help='character label')
                                            #default='0123456789??���Ű���???�����ݱ�????????????????????????????????????????????�η縶�Ӹ��𹫹��̹ٹ����???�ϻ�???????????????????????????????????????????????????????????????????????????����������õ������ī????????????????????????', help='character label')
                        parser_cp.add_argument('--sensitive', action='store_true', help='for sensitive character mode') # ?????????????? ??????���ڱ�??? ���� ?????
                        parser_cp.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
                        """ Model Architecture """
                        parser_cp.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
                        parser_cp.add_argument('--FeatureExtraction', type=str,default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
                        parser_cp.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
                        parser_cp.add_argument('--Prediction', type=str,default='Attn', help='Prediction stage. CTC|Attn')
                        parser_cp.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
                        parser_cp.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
                        parser_cp.add_argument('--output_channel', type=int, default=512,
                                            help='the number of output channel of Feature extractor')
                        parser_cp.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

                        opt_cp = parser_cp.parse_args()
                        opt_cp.image_folder = crop_save_path
                        
                        license_plate_number = demo2(opt_cp)
                        print(license_plate_number)
                        os.remove(crop_save_path)                               
                        
                        label = '%s %.2f' % (names[int(cls)], conf)
                        im0 = plot_one_box2(xyxy, im0, label=license_plate_number, color=colors[int(cls)], line_thickness=3)
                                             

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best_overall.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='/media/jhan/새 볼륨/car_plate_vid/output2', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=True, action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='cfg/TAS_car_plate.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/car_plate/data_car_plate.names', help='*.cfg path')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
