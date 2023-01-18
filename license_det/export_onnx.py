import argparse
import os
import torch
import sys, os
# from models_onnx import Darknet      # TAS에 적용 테스트하기 위해서 여기서 모델 구조 수정 권장
from models.models import Darknet  # 공식 레포지토리 추천 모델 구조 수정하지않는 것을 권장
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.google_utils import attempt_download

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./2nd.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--cfg', type=str, default='./cfg/TAS_car_plate.cfg', help='cfg')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)

    # Input
    img = torch.zeros((opt.batch_size, 3, *opt.img_size))  # image size(1,3,320,192) iDetection

    # Load PyTorch model
    attempt_download(opt.weights)
    # model = torch.load(opt.weights, map_location=torch.device('cpu'))['model'].float()
    # model.eval()
    # model.model[-1].export = True  # set Detect() layer export=True
    # y = model(img)  # dry run

    device = torch.device('cpu')
    model = Darknet(opt.cfg).to(device)
    ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
    ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt['model'], strict=True) # default strict=False
    
    model.eval()
    # model.model[-1].export = True  # set Detect() layer export=True
    # y = model(img)  # dry run
    # print(y)
    y = None

    # # TorchScript export
    # try:
    #     print('\nStarting TorchScript export with torch %s...' % torch.__version__)
    #     f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
    #     ts = torch.jit.trace(model, img)
    #     ts.save(f)
    #     print('TorchScript export success, saved as %s' % f)
    # except Exception as e:
    #     print('TorchScript export failure: %s' % e)

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        model.fuse()  # only for ONNX
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # # CoreML export
    # try:
    #     import coremltools as ct

    #     print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
    #     # convert model from torchscript and apply pixel scaling as per detect.py
    #     model = ct.convert(ts, inputs=[ct.ImageType(name='images', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
    #     f = opt.weights.replace('.pt', '.mlmodel')  # filename
    #     model.save(f)
    #     print('CoreML export success, saved as %s' % f)
    # except Exception as e:
    #     print('CoreML export failure: %s' % e)

    # Finish
    print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')
