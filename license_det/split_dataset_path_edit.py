import argparse
from email import parser
import glob
from sklearn.model_selection import train_test_split
import os
# from difflib import context_diff

PATH_SEP = os.sep #"\\"

def split_data_path(opt):

    # path = '/media/jhan/새 볼륨/split_path_test/' #train.txt와 val.txt를 저장할 path
    path = opt.dataset_path
    image_path = path + "images_T" # 이미지가 있는 디렉토리
    label_path = path + "yolo_gt" # 라벨이 있는 디렉토리
    img_list = glob.glob(f'{image_path}' + os.sep + '**/*.jpg', recursive=True) # 이미지 파일들의 이름 들을 읽어온 후 리스트로 저장
    label_list = glob.glob(f'{label_path}' + os.sep + '**/*.txt', recursive=True) # 라벨 파일들의 이름 들을 읽어온 후 리스트로 저장
    # img_list = glob.glob(f'{path}' + os.sep + '**/*.jpg', recursive=True) # 이미지 파일들의 이름 들을 읽어온 후 리스트로 저장
    # label_list = glob.glob(f'{path}' + os.sep + '**/*.txt', recursive=True) # 라벨 파일들의 이름 들을 읽어온 후 리스트로 저장

    print("================================")
    if len(img_list) == len(label_list):
        print("Image # \t: \t" ,len(img_list))
        print("Label # \t: \t" ,len(label_list))
        print("Images & Labels match")

    else:
        print("Image # \t: \t" ,len(img_list))
        print("Label # \t: \t" ,len(label_list))
        print("Images & Labels not match")

    print("================================")

    def fileopen_img():
        splitdata = []
        for line in img_list:
            line = os.path.basename(line)
            line = line.strip('.jpg')
            splitdata.append(line)
        
        # splitdata = [line.strip('\n').strip('.jpg').strip(image_path) for line in img_list]

        #리스트의 중첩된 부분 삭제하기 
        splitdata = list(dict.fromkeys(splitdata))
    
        return splitdata

    def fileopen_label():
        splitdata = []
        for line in label_list:
            line = os.path.basename(line)
            line = line.strip('.txt')
            splitdata.append(line)
        

        # splitdata = [line.strip('\n').strip('.txt').strip(label_path) for line in label_list]

        #리스트의 중첩된 부분 삭제하기 
        splitdata = list(dict.fromkeys(splitdata))
    
        return splitdata

    total_images_old = fileopen_img()
    total_labels_old = fileopen_label()
    

    dif1_total = list(set(total_labels_old) - set(total_images_old))
    dif2_total = list(set(total_images_old) - set(total_labels_old))

    common_list_total = list((set(total_images_old)-set(dif1_total))-set(dif2_total))

    train_img_list, val_img_list = train_test_split(common_list_total, test_size=0.2,random_state=2000)
    val_img_list, test_img_list = train_test_split(val_img_list, test_size=0.5,random_state=2000)
    train_label_list, val_label_list = train_test_split(common_list_total, test_size=0.2,random_state=2000)
    val_label_list, test_label_list = train_test_split(val_label_list, test_size=0.5,random_state=2000)
    # test_size => 전체 데이터 셋에서 val_img_list 가 가져갈 비율
    # random_state => 섞는 비율

    print("================================")
    print('train # \t: \t', len(train_img_list),
            '\nvalid # \t: \t', len(val_img_list),
            '\ntest  # \t: \t', len(test_img_list))

    # image_path = "/home/ubuntu/laonroad/dataset" +image_path.lstrip(".")
    # label_path = "C:\데이터셋\\"+"06커스텀데이터폴더\group1\국도\labels"
    # image_path = path
    # label_path = path
    # print(test_img_list)
    img_label_list = "./data/car_plate"
    with open(f'{img_label_list}'+os.sep+'train_images.txt','w') as f:
        f.write(image_path + PATH_SEP + ('.jpg\n'+ image_path + PATH_SEP).join(train_img_list)+'.jpg\n')

    with open(f'{img_label_list}'+os.sep+'valid_images.txt','w') as f:
        f.write(image_path + PATH_SEP + ('.jpg\n'+ image_path + PATH_SEP).join(val_img_list)+'.jpg\n')

    with open(f'{img_label_list}'+os.sep+'test_images.txt','w') as f:
        f.write(image_path + PATH_SEP + ('.jpg\n'+ image_path + PATH_SEP).join(test_img_list)+'.jpg\n')

    with open(f'{img_label_list}'+os.sep+'train_labels.txt','w') as f:
        f.write(label_path + PATH_SEP + ('.txt\n'+ label_path + PATH_SEP).join(train_label_list)+'.txt\n')

    with open(f'{img_label_list}'+os.sep+'valid_labels.txt','w') as f:
        f.write(label_path + PATH_SEP + ('.txt\n'+ label_path + PATH_SEP).join(val_label_list)+'.txt\n')

    with open(f'{img_label_list}'+os.sep+'test_labels.txt','w') as f:
        f.write(label_path + PATH_SEP + ('.txt\n'+ label_path + PATH_SEP).join(test_label_list)+'.txt\n')

    print("================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset_path', default='/media/jhan/새 볼륨/55_LP/', help='path of dataset', required=False)
    # parser.add_argument('-d','--dataset_path', default = '/media/jhan/새 볼륨/split_path_test/', help='path of dataset', required=True)
    
    opt = parser.parse_args()
    
    split_data_path(opt)
    