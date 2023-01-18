import json
import os, sys
from PIL import Image
# import pyvips
import cv2
import glob

def findfile(name, path):
    for dirpath, dirname, filename in os.walk(path):
        print(dirpath)
        if name in filename:
            return os.path.join(dirpath, name)


# def json2img_crop(pathAndFilename):
def json2img_crop(pathAndFilename, output_path, images_path):
    labels = []

    # output_path = "/media/jhan/T7 Touch/차량번호판인식/"
    # images_path = output_path + "images_T"
    
    try:    
        json_data = json.load(open(pathAndFilename))
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
                        print(class_ID, text, index)
                        class_ID_index = str(json_data_ID) + "_" + str(index) + ".jpg"
                        print(class_ID_index, bbox)
                        xx = bbox[0]
                        yy = bbox[1]
                        ww = bbox[2]
                        hh = bbox[3]
                        ll = int(xx)
                        tt = int(yy)
                        rr = int(xx + ww)
                        bb = int(yy + hh)
                                            
                        
                        try:
                            crop_save_path = output_path + "crop_images/" + class_ID_index
                            exist_img = os.path.isfile(crop_save_path)
                            if exist_img == False:
                                image_path = images_path + "/" + json_data_ID + ".jpg"
                                # image_path = glob.glob(str(images_path) + '*/**/' + str(json_data_ID)+'.jpg', recursive=True)
                                image_path = image_path[0]
                                # image_path = findfile(json_data_ID + ".jpg", images_path)
                                # print(image_path, [ll,tt,rr,bb])
                                # img = Image.open(image_path)
                                # img = pyvips.Image.new_from_image(image_path)
                                img = cv2.imread(image_path)
                                # crop_img = img.crop((ll,tt,rr,bb))
                                crop_img = img[tt: bb, ll: rr]
                                # crop_save_path = output_path + "crop_images/" + class_ID_index
                            # crop_save_path = "/media/jhan/새 볼륨/55_total/" + "crop_images/" + class_ID_index
                            # exist_img = os.path.isfile(crop_save_path)
                            # if exist_img == False:
                                print("Saved Path \t\t:\t", crop_save_path)
                                # crop_img.save(crop_save_path)
                                cv2.imwrite(crop_save_path, crop_img, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
                            else:
                                print("Already exist img & Pass")
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
    

    
        
if __name__ == '__main__':
    json_path = "/home/jha....................5_06_CR13_01_N0028.json"
    labels_path = "/home/............../car_plate_label_data/labels_j"
    output_path = "/home/jhan/laonr.......car-lincenseplateocr-main/car_plate_label_data/"
    # with open(output_path+'labels.txt','w') as f:
    #     f.writelines([])
    # # json2ocr(json_path, output_path)
    
    import glob
    for pathAndFilename in glob.iglob(os.path.join(labels_path, "*.json")):
        # print(pathAndFilename)
        json2img_crop(pathAndFilename, output_path)
