import shutil
from distutils.dir_util import copy_tree
# from tqdm import tqdm
import os
import multiprocessing
import argparse

# move_file = "/mnt/storage/DB/car_plate/images"
# output_path = "/mnt/storage/DB/car_plate/images_T"


# possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.png']
# possible_img_extension = ['.txt']
# possible_img_extension = ['.json']

def move2other_path(move_file, output_path, possible_img_extension):

    processing = 0
    for (root, dirs, files) in os.walk(move_file):
        if len(files) > 0:
            for file_name in files:
                if os.path.splitext(file_name)[1] in possible_img_extension:
                    processing += 1
                    img_path = root + "/" + file_name
                    print('#%s' %processing , 'File : ', img_path)
                    # copy_tree(img_path, output_path)
                    try:
                        shutil.copy(img_path, output_path)
                        # shutil.move(img_path, output_path)
                    except:
                        print("Skip")
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--move_file', type=str, default='', help='no / path', required=True)
    parser.add_argument('--output_path', type=str, default='', help='/ path', required=True)
    parser.add_argument('--mode', type=str, default='', help='/ path', required=True)
    opt = parser.parse_args()
    
    # move_file = "/media/jhan/T7 Touch/차량번호판인식/라벨링데이터"
    # output_path = "/media/jhan/T7 Touch/차량번호판인식/labels_T/"
    move_file = opt.move_file
    output_path = opt.output_path
    mode = opt.mode

    if mode == 'jpg':
        possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.png']
    elif mode == 'txt':
        possible_img_extension = ['.txt']
    elif mode == 'json':
        possible_img_extension = ['.json']
    
    move2other_path(move_file, output_path, possible_img_extension)
    # p = multiprocessing.Process(target=move2other_path, args=(move_file, output_path, possible_img_extension))
    # p.start()