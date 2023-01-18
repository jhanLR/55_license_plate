import json
from nis import match
import os, sys
from xml.dom import IndexSizeErr
import glob

def json2ocr(json_path, output_path, images_path, mode):
    labels = []
    
    try:        
        json_data = json.load(open(json_path))
        Learning_Data_Info = json_data["Learning_Data_Info"]
        json_data_ID = Learning_Data_Info["json_data_ID"]
        
        print(str(images_path) + "/" + str(json_data_ID)+  ".jpg")
        match_jsonNimg = os.path.isfile(str(images_path) + "/" + str(json_data_ID)+  ".jpg")
        # img_file = glob.glob(str(images_path) + '*/**/' + str(json_data_ID)+'.jpg', recursive=True)
        # print(img_file)
        # match_jsonNimg = os.path.isfile(img_file[0])
        # print(match_jsonNimg)
        if match_jsonNimg == True:
            annotations = Learning_Data_Info["annotations"]
            # print(annotations)   
            
            for anno in annotations:
                license_plate = anno["license_plate"]
                print(license_plate)
                
                # 없는 데이터는 스킵
                if license_plate != "unknown":
                    # continue
                    for lp in license_plate:
                        class_ID = lp["class_ID"]
                        if class_ID != "unknown":
                            text = lp["text"]
                            index = lp["index"]
                            # print(class_ID, text, index)
                            # class_ID_index = str(json_data_ID) + "_" + str(index) + ".jpg"
                            class_ID_index = str(output_path) + "crop_images/" + str(json_data_ID) + "_" + str(index) + ".jpg"
                            print(class_ID_index, text)
                            label = class_ID_index + "\t" + text + "\n"
                            labels.append(label)
                        else:
                            print("skip")           
                
            # print(labels)
            with open(output_path+'labels_%s.txt' %mode,'a') as f:
                f.writelines(labels)    
        
        else:
            print("No img")
    except:
        print("JSON file ERROR")
    
        
if __name__ == '__main__':
    json_path = "/home/jhan/laonroad/deeppart/car-lincenseplateocr-main/car_plate_label_data/C-220705_06_CR13_01_N0028.json"
    images_path = "/media/jhan/새 볼륨/55번_과제(OCR)/images_T"
    labels_path = "/media/jhan/새 볼륨/55번_과제(OCR)/labels_test"
    output_path = "/media/jhan/새 볼륨/55번_과제(OCR)/"
    mode = "test"
    with open(output_path+'labels.txt','w') as f:
        f.writelines([])
    # json2ocr(json_path, output_path)
    
    import glob
    for pathAndFilename in glob.iglob(os.path.join(labels_path, "*.json")):
        print(pathAndFilename)
        json2ocr(json_path, output_path, images_path, mode)
