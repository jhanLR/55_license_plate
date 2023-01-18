import json
import os, sys
from PIL import Image
# import pyvips
import cv2
import glob
import json
import os, sys
from PIL import Image
# import pyvips
import cv2
import glob
import multiprocessing

def findfile(name, path):
    for dirpath, dirname, filename in os.walk(path):
        # print(dirpath)
        if name in filename:
            return os.path.join(dirpath, name)

def json2img_crop(json_path, output_path, images_path):
    labels = []
    
    try:    
        json_data = json.load(open(json_path))
        Learning_Data_Info = json_data["Learning_Data_Info"]
        json_data_ID = Learning_Data_Info["json_data_ID"]
        
        # print(json_data_ID)
        
        annotations = Learning_Data_Info["annotations"]
        # print(annotations)
        
        for anno in annotations:
            license_plate = anno["license_plate"]
            # print(license_plate)
            
            # 없는 데이터는 스킵
            if license_plate != "unknown":
                # continue
                for lp in license_plate:
                    class_ID = lp["class_ID"]
                    if class_ID != "unknown":
                        text = lp["text"]
                        index = lp["index"]
                        bbox = lp["bbox"]
                        # print(class_ID, text, index)
                        class_ID_index = str(json_data_ID) + "_" + str(index) + ".jpg"
                        # print(class_ID_index, bbox)
                        xx = bbox[0]
                        yy = bbox[1]
                        ww = bbox[2]
                        hh = bbox[3]
                        ll = int(xx)
                        tt = int(yy)
                        rr = int(xx + ww)
                        bb = int(yy + hh)
                                            
                        
                        try:
                            # image_path = images_path + "/" + json_data_ID + ".jpg"
                            image_path = findfile(json_data_ID + ".jpg", images_path)
                            # print(image_path, [ll,tt,rr,bb])
                            img = Image.open(image_path)
                            # img = pyvips.Image.new_from_image(image_path)
                            # img = cv2.imread(image_path)
                            crop_img = img.crop((ll,tt,rr,bb))
                            # crop_img = img[tt: bb, ll: rr]
                            crop_save_path = output_path + "crop_imagesss/" + class_ID_index
                            print("Saved Path \t\t:\t", crop_save_path)

                            # cv2.imwrite(crop_save_path, crop_img, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
                        except:
                            print("No img")
                        
                        # label = class_ID_index + "\t" + text + "\n"
                        # labels.append(label)
                    else:
                        print("skip")
                
                
                # crop & save labels path
                
                
        # print(labels)
        # with open(output_path+'labels.txt','a') as f:
        #     f.writelines(labels)
    
    except:
        print("JSON file ERROR")
    

    
        
# if __name__ == '__main__':
#     images_path = "/media/jhan/새 볼륨/55번_과제(OCR)/images_T"
#     labelsT_path = "/media/jhan/새 볼륨/55번_과제(OCR)/labels_T"
#     mode = "T" # train/valid/test /T
#     labels_path = "/media/jhan/새 볼륨/55번_과제(OCR)/labels_%s" % mode
#     output_path = "/media/jhan/새 볼륨/55번_과제(OCR)/"
#     # with open(output_path+'labels.txt','w') as f:
#     #     f.writelines([])
#     # # json2ocr(json_path, output_path)
    
#     import glob
#     json_count = 0
#     for pathAndFilename in glob.iglob(os.path.join(labels_path, "*.json")):
#         json_count += 1
#         print("FILE # : ", json_count)
#         json2img_crop(pathAndFilename, output_path, images_path)
#         # p = multiprocessing.Process(target=json2img_crop, args=(pathAndFilename, output_path, images_path))
#         # p.start()


        
if __name__ == '__main__':
    images_path = "/media/jhan/새 볼륨/55번_과제(OCR)/차량번호판"
    labels_path = "/media/jhan/새 볼륨/55번_과제(OCR)/labels_test"
    output_path = "/media/jhan/새 볼륨/55번_과제(OCR)/"
    find_file = "C-220806_06_CR18_02_N0432.jpg"
    find = findfile(find_file, images_path)
    print(find)
    
    img = cv2.imread(find, cv2.IMREAD_ANYCOLOR)
    cv2.imshow("img", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
