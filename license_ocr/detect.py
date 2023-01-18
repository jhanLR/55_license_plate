import string
import argparse
import time

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo(opt):
    start_time = time.time()
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    
    torch.save(model, "test.pt")
    
    #print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
    #      opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
    #      opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    #print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))    

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)


    # predict
    model.eval()
    with torch.no_grad():
        st = time.time()
        for image_tensors, image_path_list in demo_loader:
            sst = time.time()
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            print("!!!!!!!!!!!!!!", image)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)
                print("preds \t: \t", preds, preds_index, preds_size)
                print("preds_len \t: \t", len(preds_index[0]), len(preds_size))
                print("preds_str \t: \t", preds_str)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)
                print("preds \t: \t", preds, preds_index, length_for_pred)
                print("preds_len \t: \t", len(preds_index[0]), len(length_for_pred))
                print("preds_str \t: \t", preds_str)


            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}'
            
            print(f'{dashed_line}\n{head}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                file_name = img_name.split('/')
                file_name = file_name[-1]
                log = open(f'./output/{file_name}.txt', 'a')
                head = f'{"image_path":25s}\t{"predicted_labels":25s}'
                log.write(f'{head}\n{dashed_line}\n')
                print(f'{img_name:25s}\t{pred:25s}\n')
                log.write(f'{img_name:25s}\t{pred:25s}\n')

                log.close()
                
    end_time = time.time()
    print("Infer time : \t", end_time-st)
    print("Total time : \t", end_time-start_time)

if __name__ == '__main__':
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
    parser.add_argument('--rgb', default=True, action='store_true', help='use rgb input')
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
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)
