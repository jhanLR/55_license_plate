import shutil
from distutils.dir_util import copy_tree
# from tqdm import tqdm
import os
import multiprocessing

# move_file = "/mnt/storage/DB/car_plate/images"
# output_path = "/mnt/storage/DB/car_plate/images_T"


# possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.png']
# possible_img_extension = ['.txt']
# possible_img_extension = ['.json']

def move2other_path(move_file, output_path, possible_img_extension):
    percentage_set = 50
    index_test = round(100 / percentage_set)  
    print(index_test)

    processing = 0
    count = 0
    for (root, dirs, files) in os.walk(move_file):
        if len(files) > 0:
            for file_name in files:
                if os.path.splitext(file_name)[1] in possible_img_extension:
                    processing += 1
                    count += 1
                    img_path = root + "/" + file_name
                    print('#%s' %count , 'File : ', img_path)
                    # copy_tree(img_path, output_path)
                    if processing == index_test:
                        processing = 0
                        try:
                            # shutil.copy(img_path, output_path)
                            shutil.move(img_path, output_path)
                        except:
                            print("Skip")
                    else:
                        print('PASS')
                    
if __name__ == '__main__':
    move_file = "/media/jhan/T7 Touch/차량번호판인식/labels_vsdsdalid"
    output_path = "/media/jhan/T7 Touch/차량번호판인식/labelssdsd_test/"
    
    # possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.png']
    # possible_img_extension = ['.txt']
    possible_img_extension = ['.json']
    
    move2other_path(move_file, output_path, possible_img_extension)
    # p = multiprocessing.Process(target=move2other_path, args=(move_file, output_path, possible_img_extension))
    # p.start()