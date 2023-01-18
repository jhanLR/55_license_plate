import argparse
from email import parser
import glob
from sklearn.model_selection import train_test_split
import os
import shutil
import multiprocessing
# from difflib import context_diff

PATH_SEP = os.sep #"\\"
# PATH_SEP = "\\"

def copyNmove_json(files_list, dataset_path, moved_path):
    processing = 0
    for file_name in files_list:
        processing += 1
        input_path = dataset_path + "/" + file_name
        # input_path = glob.glob(str(dataset_path) + '*/**/' + str(file_name), recursive=True)
        # input_path = input_path[0]
        print('#%s' %processing , 'File : ', input_path)
        # copy_tree(img_path, output_path)
        try:
            shutil.copy2(input_path, moved_path)
            # shutil.move(img_path, output_path)
        except:
            print("Skip")


def json_split(dataset_path, output_path):

    # path = '/media/jhan/새 볼륨/split_path_test/' #train.txt와 val.txt를 저장할 path
    path = dataset_path
    label_path = path #+ "labels" # 라벨이 있는 디렉토리
    label_list = glob.glob(label_path + os.sep + '**/*.json', recursive=True) # 라벨 파일들의 이름 들을 읽어온 후 리스트로 저장


    def fileopen_label():
        splitdata = []
        for line in label_list:
            line = os.path.basename(line)
            # line = line.strip('.txt')
            splitdata.append(line)
        

        # splitdata = [line.strip('\n').strip('.txt').strip(label_path) for line in label_list]

        #리스트의 중첩된 부분 삭제하기 
        splitdata = list(dict.fromkeys(splitdata))
    
        return splitdata

    print(len(label_list))
    total_labels_old = fileopen_label()
    
    train_label_list, val_label_list = train_test_split(total_labels_old, test_size=0.2,random_state=2000)
    val_label_list, test_label_list = train_test_split(val_label_list, test_size=0.5,random_state=2000)
    # test_size => 전체 데이터 셋에서 val_img_list 가 가져갈 비율
    # random_state => 섞는 비율

    print("================================")
    print('train # \t: \t', len(train_label_list),
            '\nvalid # \t: \t', len(val_label_list),
            '\ntest  # \t: \t', len(test_label_list))

    print("================================")

    os.makedirs(output_path + "labels_train", exist_ok=True)
    os.makedirs(output_path + "labels_valid", exist_ok=True)
    os.makedirs(output_path + "labels_test", exist_ok=True)
    
    copyNmove_json(train_label_list, dataset_path, output_path + "labels_train")
    copyNmove_json(val_label_list, dataset_path, output_path + "labels_valid")
    copyNmove_json(test_label_list, dataset_path, output_path + "labels_test")
    
    print("================================")
    print('train # \t: \t', len(train_label_list),
            '\nvalid # \t: \t', len(val_label_list),
            '\ntest  # \t: \t', len(test_label_list))

    print("================================")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-d','--dataset_path', default='/media/jhan/새 볼륨/55번_과제(OCR)/labels_T', help='path of dataset')
    # parser.add_argument('-o','--output_path', default='/media/jhan/새 볼륨/55번_과제(OCR)/', help='path of output')
    # parser.add_argument('-d','--dataset_path', default = '/media/jhan/새 볼륨/split_path_test/', help='path of dataset', required=True)
    
    # opt = parser.parse_args()
    
    dataset_path = '/media/jhan/새 볼륨/55번_과제(OCR)/레이블링'
    output_path = '/media/jhan/새 볼륨/55번_과제(OCR)/'
    
    json_split(dataset_path, output_path)
    