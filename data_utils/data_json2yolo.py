import json
import os, sys
from PIL import Image
import glob

def json2yolo(json_path, output_path, images_path):
    yolo_gt = []
    
    try:    
        json_data = json.load(open(json_path))
        Raw_Data_Info = json_data["Raw_Data_Info"]
        res = Raw_Data_Info["resolution"]
        Learning_Data_Info = json_data["Learning_Data_Info"]
        json_data_ID = Learning_Data_Info["json_data_ID"]
        resolution=res.split(',')
        print(resolution[0])
        
        resolution[0] = int(resolution[0])
        resolution[1] = int(resolution[1])
        
        print(json_path)
        
        match_jsonNimg = os.path.isfile(str(images_path) + "/" + str(json_data_ID)+  ".jpg")
        # print(match_jsonNimg)
        if match_jsonNimg == True:
        
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
                            
                            # if class_ID == "no_01":
                            #     class_ID = 0
                            # elif class_ID == "no_02":
                            #     class_ID = 1
                            # elif class_ID == "no_03":
                            #     class_ID = 2
                            # elif class_ID == "no_04":
                            #     class_ID = 3
                            # elif class_ID == "no_05":
                            #     class_ID = 4
                            
                            class_ID = 0
                            
                            # text = lp["text"]
                            # index = lp["index"]
                            bbox = lp["bbox"]
                            # print(class_ID, text, index)
                            # class_ID_index = str(json_data_ID) + "_" + str(index) + ".jpg"
                            # print(class_ID_index, bbox)
                            # Left, Top = xx, yy
                            xx = bbox[0]
                            yy = bbox[1]
                            ww = bbox[2]
                            hh = bbox[3]
                            
                            # Excluding resolutions other than official resolutions
                            if resolution[0] > 1920 and resolution[0] < 3840:
                                resolution[0] = 1920
                            if resolution[1] > 1080 and resolution[1] < 2160:
                                resolution[1] = 1080                   
                            # YOLO format -> center X, center Y, W, H
                            cx = (xx + (ww/2)) / resolution[0]
                            cy = (yy + (hh/2)) / resolution[1]
                            cw = (ww) / resolution[0]
                            ch = (hh) / resolution[1]
                            # cx = (xx + (ww/2)) / 1920
                            # cy = (yy + (hh/2)) / 1080
                            # cw = (ww) / 1920
                            # ch = (hh) / 1080
                            print("class_ID \t\t: \t", class_ID)
                            print("YOLO Format \t\t: \t",[cx, cy, cw, ch])
                            yolo_data = str(class_ID) + " " + str(cx) + " " + str(cy) + " " + str(cw) + " " + str(ch) + "\n"
                            yolo_gt.append(yolo_data)
                            # print(yolo_gt)
                        else:
                            print("skip")                
                else:
                    print(json_data_ID)
                    
            # print(yolo_gt)
            if len(yolo_gt) > 0:
                with open(output_path+'yolo_gt/'+ json_data_ID + '.txt','w') as f:
                    f.writelines(yolo_gt)    
        
        else:
            print("No img")    
    except:
        print("JSON file ERROR")

    
        
if __name__ == '__main__':
    # json_path = "/home/jhan/laonroad/deeppart/car-lincenseplateocr-main/car_plate_label_data/C-220705_06_CR13_01_N0028.json"
    labels_path = "/media/jhan/새 볼륨/ANPR/dataset_car_plate/labels_j"
    output_path = "/media/jhan/새 볼륨/ANPR/dataset_car_plate/"
    # with open(output_path+'yolo_gt.txt','w') as f:
    #     f.writelines([])
    
    # os.makedirs(output_path + "yolo_gt", exist_ok=True)
    os.makedirs(output_path + "yolo_gt/", exist_ok=True)
    
    for pathAndFilename in glob.iglob(os.path.join(labels_path, "*.json")):
        # print(pathAndFilename)
        json2yolo(pathAndFilename, output_path)
        