from data_utils.data_json2ocr import *
from data_utils.data_json2img_crop import *
from data_utils.data_json2yolo import *
from data_utils.data_json_split import *
from data_utils.data_json2img_crop_one_word import *
import multiprocessing
import time
from datetime import datetime
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, default='json_split', required=True, help='Func : json_split / json2img_crop / json2ocr / json2yolo')
    parser.add_argument('--mode', type=str, default='T', required=True, help='mode : train / valid / test / T')
    parser.add_argument('--output_path', type=str, default='', required=True, help='dataset_path')
    opt = parser.parse_args()
    ###################################
    ########## User's Manual ##########
    # 1. Split json train/valid/test
    # 2. Crop images for OCR (mode : T)
    # 3-1. Transform OCR data on json (mode : train / valid / test)
    # 3-2. Transform YOLO data on json (mode : T)
    ###################################
    # Func = "json2img_crop" # json_split / json2img_crop / json2ocr / json2yolo //json2img_crop_one_word
    # mode = "T" # train / valid / test / T
    
    # output_path = "/media/jhan/T7 Touch/차량번호판인식/"
    
    Func = opt.func # json_split / json2img_crop / json2ocr / json2yolo //json2img_crop_one_word
    mode = opt.mode # train / valid / test / T
    
    output_path = opt.output_path
    
    # output_path1 = "/media/jhan/T7 Touch/차량번호판인식/extra_data/"
    images_path = output_path + "images_T"
    labelsT_path = output_path + "labels_T"
    labels_path = output_path + "labels_%s" % mode
           
    
    print("Start Operation...")
    time.sleep(1.0)
    start_time = datetime.now()
    
    if Func == "json_split":
        json_split(labelsT_path, output_path)
    
    else:
        # with open(output_path+'yolo_gt.txt','w') as f:
        #     f.writelines([])
        with open(output_path+'labels_%s.txt' %mode,'w') as f:
            f.writelines([])
            
        os.makedirs(output_path + "yolo_gt/", exist_ok=True)
        os.makedirs(output_path + "crop_images/", exist_ok=True)
        # os.makedirs(output_path + "yolo_gt", exist_ok=True)
        
        json_count = 0
        
        ps = []       
        for pathAndFilename in glob.iglob(os.path.join(labels_path, "*.json")):
            json_count += 1
            
            print("Open JSON file #%s \t:\t" %json_count, pathAndFilename)
            
            if Func == "json2ocr":
                json2ocr(pathAndFilename, output_path, images_path, mode) # don't need multiprocessing
            elif Func == "json2yolo":
                json2yolo(pathAndFilename, output_path, images_path) # don't need multiprocessing
            elif Func == "json2img_crop":
                # json2img_crop(pathAndFilename, output_path, images_path)
                p = multiprocessing.Process(target=json2img_crop, args=(pathAndFilename, output_path, images_path))
                ps.append(p)
                p.start()
        
            if len(ps) > 1000:
                for p in ps:
                    p.join()
                ps = []        
            
            # json2ocr(pathAndFilename, output_path, images_path, mode) # don't need multiprocessing
            # json2yolo(pathAndFilename, output_path, images_path) # don't need multiprocessing
            # json2img_crop(pathAndFilename, output_path, images_path)
            # p = multiprocessing.Process(target=json2yolo, args=(pathAndFilename, output_path))
            # else:
            #     if Func == "json2img_crop_one_word":
            #         json2img_crop_one_word(pathAndFilename, output_path, images_path)
            #         # p = multiprocessing.Process(target=json2img_crop_one_word, args=(pathAndFilename, output_path, images_path))
            #         # p.start()
                    
            #     else:
            #         p = multiprocessing.Process(target=json2img_crop, args=(pathAndFilename, output_path, images_path))
            #         p.start()
            #     # if Func == "json2img_crop":
            #     #     p = multiprocessing.Pool(15) # max 20
            #     #     p.map(json2img_crop, ps)
            #     #     p.close()
            #     #     p.join()    
            # p.join()
        
    end_time = datetime.now()
    print("Start Time\t:\t", start_time)
    print("End Time\t:\t", end_time)
    print("Done.")