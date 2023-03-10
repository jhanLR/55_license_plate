import os
import json
import fire
import os
import lmdb
import cv2
import shutil

import numpy as np


data_path = './input/images'
directory = './input/images'
if not os.path.exists(directory):
    os.makedirs(directory)

folder_list = []
"""
with open('./input/gt_Test.txt', 'a') as gt:
    for folder in os.listdir(data_path):
        img_folder = folder + '.jpg'
        folder_list.append(folder)
        img_path = os.path.join(data_path,folder,'images',img_folder)
        label_path = os.path.join(data_path,folder,'data.json')
        target_path =os.path.join(directory,img_folder)
        shutil.copy(img_path,target_path)
        with open(label_path,'r',encoding='UTF-8') as json_file:
            label = json.load(json_file)
            label = label['value']
            label = label.replace('\n','')
            label = label.replace(' ','')

        #fill = 'Test' + '/'+folder+'/'+'images'+'/'+folder +'.jpg'+'\t'+label +'\n'
        fill = 'Copy' + '/'+'Test'+'/' + img_folder +'\t'+label +'\n'
        gt.write(fill)
"""
with open('./input/gt.txt', 'a') as gt:
    for folder in os.listdir(data_path):
        label=folder.split('.')[0]
        label=label.split('-')[0]
        img_folder = folder
        fill = 'images'+'/' + img_folder +'\t'+label +'\n'
        gt.write(fill)

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def createDataset(mode, gtFile, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    
    # mode = "train" # train / valid / test
    mode = mode # train / valid / test
    
    outputPath = './input/lmdb_%s' %mode
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1e10)
    cache = {}
    cnt = 1

    # gtFile = '/media/jhan/??? ??????/55_LP/labels_%s.txt' %mode
    gtFile = gtFile + '/labels_%s.txt' %mode
    with open(gtFile, 'r', encoding='UTF8') as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    for i in range(nSamples):
        imagePath, label = datalist[i].strip('\n').split('\t')

        print('imagepath',imagePath)
        # inputPath = './input'
        # imagePath = os.path.join(inputPath, imagePath)
        #print('imagepath2',imagePath)

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)

    print('Created test dataset with %d samples' % nSamples)
    #os.remove('./input/gt_train.txt')

if __name__ == '__main__':
    fire.Fire(createDataset)
