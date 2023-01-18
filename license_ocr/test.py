# -*- coding:utf-8 -*-
import os
import time
import string
import argparse
import re
import logging
import json
import fire
import os
import lmdb
import cv2
import shutil

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance
from datetime import datetime

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model

import subprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(
    filename='Test_Environment_Log.log',
    filemode='w',
    format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logging.info('python test.py')
def output_command(command):
    bashCommand = command
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error

def benchmark_all_eval(model, criterion, converter, opt, calculate_infer_time=False):
    """ evaluation with 10 benchmark evaluation datasets """
    # The evaluation datasets, dataset order is same with Table 1 in our paper.
    eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
                      'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']

    # # To easily compute the total accuracy of our paper.
    # eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_867',
    #                   'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']

    if calculate_infer_time:
        evaluation_batch_size = 1  # batch_size should be 1 to calculate the GPU inference time per image.
    else:
        evaluation_batch_size = opt.batch_size

    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    #log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a',encoding='UTF8')
    dashed_line = '-' * 80
    print(dashed_line)
    #log.write(dashed_line + '\n')
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        _, accuracy_by_best_model, norm_ED_by_best_model, _, _, _, infer_time, length_of_data = validation(
            model, criterion, evaluation_loader, converter, opt)
        list_accuracy.append(f'{accuracy_by_best_model:0.3f}')
        total_forward_time += infer_time
        total_evaluation_data_number += len(eval_data)
        total_correct_number += accuracy_by_best_model * length_of_data
        #log.write(eval_data_log)
        print(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}')
        #log.write(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}\n')
        print(dashed_line)
        #log.write(dashed_line + '\n')

    averaged_forward_time = total_forward_time / total_evaluation_data_number * 1000
    total_accuracy = total_correct_number / total_evaluation_data_number
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    evaluation_log = 'accuracy: '
    for name, accuracy in zip(eval_data_list, list_accuracy):
        evaluation_log += f'{name}: {accuracy}\t'
    evaluation_log += f'total_accuracy: {total_accuracy:0.3f}\t'
    evaluation_log += f'averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num / 1e6:0.3f}'
    print(evaluation_log)
    #log.write(evaluation_log + '\n')
    #log.close()

    return None


def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    n_ob = 0
    n_total = 0
    n_correct = 0
    n_wrong = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    Intro = 'Ground Truth , Prediction , Correct/Incorrect'
    logging.info(Intro)
    if opt.eval_log:
        # eval_log = open(f'./result/{opt.exp_name}/eval_log.txt', 'a',encoding='UTF8')
        # eval_log.write(f'{"Ground Truth":25s} | {"Prediction":25s} | Correct/Incorrect')
        eval_log = open(f'./result/{opt.exp_name}/eval_log.csv', 'a',encoding='UTF8')
        eval_log2 = open(f'./result/{opt.exp_name}/eval_log_sort.csv', 'a',encoding='UTF8')
        eval_log.write(f'No.,Ground Truth,Prediction,Correct,Incorrect\n')
        eval_log2.write(f'No.,Correct,Incorrect\n')
    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        #print("Index : ",i,"   Length of data :",length_of_data, "   batch_size :",batch_size)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        start_time = time.time()
        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTCloss format
            if opt.baiduCTC:
                cost = criterion(preds.permute(1, 0, 2), text_for_loss, preds_size, length_for_loss) / batch_size
            else:
                cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

            # Select max probabilty (greedy decoding) then decode index to character
            if opt.baiduCTC:
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
            else:
                _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index.data, preds_size.data)

        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        #with open(f'./Failure_log_test.txt', 'a') as log:

        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
                now = time.localtime()
                if 'Attn' in opt.Prediction:
                    gt = gt[:gt.find('[s]')]
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
                if opt.sensitive and opt.data_filtering_off:
                    pred = pred.lower()
                    gt = gt.lower()
                    alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                    out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                    pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                    gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)


                ###################################################################
                # If gt value is x, any value predicted will be treated as correct.
                ###################################################################
                predd = False
                if len(gt) > len(pred):
                    comp_range = len(pred)
                else:
                    comp_range = len(gt)
                    
                n_ob += 1    
                for i in range(comp_range):
                    # if (pred[i] == gt[i]) or (pred[i] == 'x') or (gt[i] == 'x'):
                    if (pred[i] == gt[i]):# or (gt[i] == 'x'):
                        predd = True
                        n_correct += 1
                        n_total += 1
                        line=gt,pred,'Correct'
                        logging.info(line)
                        # logging.info(gt,pred,'Correct')
                        # logging.info(str(gt),'|',pred,'|','Correct','|',"%04d/%02d/%02d %02d:%02d:%02d"%(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
                        # logging.info(gt,'|',pred,'|','Correct')
                        # if opt.eval_log:
                        #     # line_eval = f'\n{gt:25s} | {pred:25s} | Correct'
                        #     # eval_log.write(line_eval)
                        #     eval_log.write(str(n_ob) + ',' + str(gt[i]) + ',' + str(pred[i]) + ',' + str(n_correct) + ',' + str(n_wrong)+'\n')
                    else:
                        predd = False
                        n_wrong += 1
                        n_total += 1
                        line = gt,pred,'Incorrect'
                        logging.info(line)
                        # logging.info(gt,pred,'Incorrect')
                        # logging.info(str(gt),'|',pred,'|','Incorrect','|',"%04d/%02d/%02d %02d:%02d:%02d"%(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
                        # logging.info(gt,'|',pred,'|','Incorrect')
                        # if opt.eval_log:
                        #     # line_eval = f'\n{gt:25s} | {pred:25s} | Incorrect'
                        #     # eval_log.write(line_eval)
                        #     eval_log.write(str(n_ob) + ',' + str(gt[i]) + ',' + str(pred[i]) + ',' + str(n_correct) + ',' + str(n_wrong)+'\n')

                        # break
                        
                    if opt.eval_log:
                        # line_eval = f'\n{gt:25s} | {pred:25s} | Correct'
                        # eval_log.write(line_eval)
                        eval_log.write(str(n_ob) + ',' + str(gt[i]) + ',' + str(pred[i]) + ',' + str(n_correct) + ',' + str(n_wrong)+'\n')    
                        
                # print(gt, pred)
                # print("predd : ", predd)
                # if predd:
                #     n_correct += 1
                #     n_total += 1
                #     line=gt,pred,'Correct'
                #     logging.info(line)
                #     # logging.info(gt,pred,'Correct')
                #     # logging.info(str(gt),'|',pred,'|','Correct','|',"%04d/%02d/%02d %02d:%02d:%02d"%(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
                #     # logging.info(gt,'|',pred,'|','Correct')
                #     if opt.eval_log:
                #         # line_eval = f'\n{gt:25s} | {pred:25s} | Correct'
                #         # eval_log.write(line_eval)
                #         eval_log.write(str(n_total) + ',' + str(gt) + ',' + str(pred) + ',' + str(n_correct) + ',' + str(n_wrong)+'\n')
                # else:
                #     n_wrong += 1
                #     n_total += 1
                #     line = gt,pred,'Incorrect'
                #     logging.info(line)
                #     # logging.info(gt,pred,'Incorrect')
                #     # logging.info(str(gt),'|',pred,'|','Incorrect','|',"%04d/%02d/%02d %02d:%02d:%02d"%(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
                #     # logging.info(gt,'|',pred,'|','Incorrect')
                #     if opt.eval_log:
                #         # line_eval = f'\n{gt:25s} | {pred:25s} | Incorrect'
                #         # eval_log.write(line_eval)
                #         eval_log.write(str(n_total) + ',' + str(gt) + ',' + str(pred) + ',' + str(n_correct) + ',' + str(n_wrong)+'\n')
                ###################################################################
                
                
                # if pred == gt:
                # # if predd:
                #     n_correct += 1
                #     n_total += 1
                #     line=gt,pred,'Correct'
                #     logging.info(line)
                #     #logging.info(gt,pred,'Correct')
                #     #logging.info(str(gt),'|',pred,'|','Correct','|',"%04d/%02d/%02d %02d:%02d:%02d"%(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
                #     #logging.info(gt,'|',pred,'|','Correct')


                # if pred != gt:
                # # else:
                #     n_wrong += 1
                #     n_total += 1
                #     line = gt,pred,'Incorrect'
                #     logging.info(line)
                #     #logging.info(gt,pred,'Incorrect')
                #     #logging.info(str(gt),'|',pred,'|','Incorrect','|',"%04d/%02d/%02d %02d:%02d:%02d"%(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
                #     #logging.info(gt,'|',pred,'|','Incorrect')

                # ICDAR2019 Normalized Edit Distance
                if len(gt) == 0 or len(pred) == 0:
                    norm_ED += 0
                elif len(gt) > len(pred):
                    norm_ED += 1 - edit_distance(pred, gt) / len(gt)
                else:
                    norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
                try:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                except:
                    confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
                confidence_score_list.append(confidence_score)
            #print(pred, gt, pred==gt, confidence_score)

        logging.info('Correct Prediction : %d'%n_correct)
        logging.info('Incorrect Prediction : %d'%n_wrong)
        logging.info('Total : %d' % n_total)
        # accuracy = n_correct / float(length_of_data) * 100
        # norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance
        accuracy = n_correct / float(n_total) * 100
        norm_ED = norm_ED / float(n_total)  # ICDAR2019 Normalized Edit Distance
        # eval_log.write(str(n_total) + ',' +str(n_correct) + ',' + str(n_wrong)+'\n')

        #log.close
        #print("Iteration : ", i, "  Failure : ", n_wrong)

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data, n_total, n_correct, n_wrong


def test(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    #print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
    #      opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
    #      opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    #print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    print('Pretrained model loaded')

    # print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    # os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    """ evaluation """
    model.eval()
    with torch.no_grad():
        if opt.benchmark_all_eval:  # evaluation with 10 benchmark evaluation datasets
            benchmark_all_eval(model, criterion, converter, opt)
        else:
            # log = open(f'./result/{opt.exp_name}/log_evaluation.txt', 'a',encoding='UTF8')
            AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
            eval_data, eval_data_log = hierarchical_dataset(root=opt.eval_data, opt=opt)
            evaluation_loader = torch.utils.data.DataLoader(
                eval_data, batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.workers),
                collate_fn=AlignCollate_evaluation, pin_memory=True)
            _, accuracy_by_best_model, _, preds, confidence_score, labels, _, _, n_total, n_correct, n_wrong = validation(model, criterion, evaluation_loader, converter, opt)

            # show some predicted results
            for gt, pred, confidence in zip(labels[:], preds[:], confidence_score[:]):
                    gt = gt[:gt.find('[s]')]
                    pred = pred[:pred.find('[s]')]

            # log.write(eval_data_log)
        if opt.eval_log:
            eval_log = open(f'./result/{opt.exp_name}/eval_log.csv', 'a',encoding='UTF8')
            eval_log2 = open(f'./result/{opt.exp_name}/eval_log_sort.csv', 'a',encoding='UTF8')
            eval_log2.write(str(n_total) + ',' +str(n_correct) + ',' + str(n_wrong)+'\n')
            eval_log.write('Accuracy' + ',' + str(accuracy_by_best_model)+'\n')
            eval_log2.write('Accuracy' + ',' + str(accuracy_by_best_model)+'\n')
            eval_log.close()
            eval_log2.close()
            print(f'Accuracy : {accuracy_by_best_model:0.3f}%')
            logging.info(f'Accuracy : {accuracy_by_best_model:0.3f}%')
            # log.write(f'{accuracy_by_best_model:0.3f}\n')
            # log.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', default = './input/lmdb_test', help='path to evaluation dataset')
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', default = './saved_models/TPS-ResNet-BiLSTM-Attn-Seed2001-Renewed/best_accuracy.pth', help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=10, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', default=True, action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, 
                        #default='0123456789揶쏉옙占쎄돌占쎈뼄占쎌뵬筌띾뜃援낉옙瑗ワ옙?쐭占쎌쑎?솒紐껋쒔占쎄퐣占쎈선占쏙옙占썸?⑥쥓?걗占쎈즲嚥≪뮆?걟癰귣똻?꺖占쎌궎鈺곌퀗?럡占쎈듇占쎈あ?뙴?뫀龜?겫占쏙옙?땾占쎌뒭雅뚯눛釉?占쎈??占쎌깈獄쏄퀗?궗占쎈퉸占쎌몓占쎈???뤃占쏙옙苑뚳옙?뒻占쎌뵥筌ｌ뮆占쏙옙占쎌읈占쏙옙占썸뤃?딆뒻占쎄텦?겫占쏙옙沅쎿꽴臾믭폒占쎄쉭?넫?굞?젫雅뚯눊而?占쎌뜚?빊?뫖?꽴?빊?뫖沅볩옙?읈?겫怨몄읈占쎄텚野껋럥?꽴野껋럥沅볟칰?럡由?', help='character label')
                        # default='0123456789().JNRW_abcdef揶쏉옙揶쏅벚而삣쳞?씧援끻칰?굛爰쇔칰?덇퍙野껋럡??롦?⑥쥒?궢?꽴占썸꽴臾롫뜲?뤃?덊닊疫꿸퀗占쏙옙繹먮굝援뱄옙沅볩옙瑗ワ옙?걗占쎈걠占쎈듇占쎈빍占쎈뼄占쏙옙占쏙옙?솙占쎈쐭占쎈젇占쎈즲占쎈짗占쎈あ占쎈쾻占쎈탵占쎌뵬占쎌쑎嚥≪뮆竊덄뙴怨뺚봺筌띾뜄?돢筌뤿굝?걟?눧???揆沃섓옙沃섎챶而?獄쏄퀡肄먫린袁⑥퓢癰귣??占쏙옙?겫怨룻돩占쎄텢占쎄텦占쎄퐣占쎄쉐占쎄쉭占쎈?쏉옙?꺖占쎈꽊占쎈땾占쎈뻻占쎈툡占쎈툢占쎈툧占쎈펶占쎈선占쎈연占쎈염占쎌겫占쎌궎占쎌궞占쎌뒠占쎌뒭占쎌뒻占쎌뜚占쎌몓占쎌몵占쎌뱽占쎌뵠占쎌뵡占쎌뵥占쎌쁽占쎌삂占쏙옙占쏙옙?읈占쎌젫鈺곌퀣伊뚥틠?눘夷뤄쭪占쏙㎕?뫁荑귨㎗?뮇?겧?빊遺욱뀱?빊?뫗留곭㎉?똻?맫占쏙옙占쏙옙?솁占쎈즸占쎈７占쎈릭占쎈??占쎌깈占쏙옙占쏙옙?뜎占쎌뿳占쎈??', help='character label')
                        #default='0123456789揶쏉옙揶쏅벚援끻칰?럡??롦?⑥쥒占쏙옙?꽴臾롫럡疫뀀뜃由경틦占쏙옙援뱄옙沅볩옙瑗ワ옙?걗占쎈듇占쎈뼄占쏙옙占쏙옙?쐭占쎈즲占쎈짗占쎈あ占쎈쾻占쎌뵬占쎌쑎嚥≪뮆竊덌쭕?뜄?돢筌뤿굝?걟?눧???揆沃섎챶而?獄쏄퀡苡?癰귣??占쏙옙?겫怨멸텢占쎄텦占쎄퐣占쎈꺖占쎈땾占쎈툡占쎈툢占쎈툧占쎈펶占쎈선占쎈염占쎌겫占쎌궎占쎌뒠占쎌뒭占쎌뒻占쎌뜚占쎌몓占쎌뵠占쎌뵥占쎌쁽占쎌삂占쏙옙占쏙옙?읈鈺곌퀣竊쒍빳臾믨컧筌ｌ뮇?겧?빊遺용븧燁삳똾占쏙옙占쎈솁占쎈즸占쎈７占쎈릭占쎈??占쎌깈占쏙옙占쏙옙?뿳', help='character label')
                        default='0123456789가나다라마아바사자하거너더러머버서어저허고노두로모보소오조호구누두루무부수우주배x', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str,default='TPS' , help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default = 'ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str,default = 'BiLSTM' , help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default = 'Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=10, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--eval_log', type=int, default=True, help='eval_log')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    test(opt)
