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


def json2img_crop_one_word(pathAndFilename, output_path, images_path):
    # labels = []

    # output_path = "/media/jhan/새 볼륨/55_LP/"
    # images_path = output_path + "images_T"
    
    try:    
        json_data = json.load(open(pathAndFilename))
        Learning_Data_Info = json_data["Learning_Data_Info"]
        json_data_ID = Learning_Data_Info["json_data_ID"]
        
        # print(json_data_ID)
        
        annotations = Learning_Data_Info["annotations"]
        # print(annotations)
        
        for anno in annotations:
            license_plate_number = anno["license_plate_number"]
           
            # continue
            ll = []
            tt = []
            rr = []
            bb = []
            license_plate = 1
            
            print(len(license_plate_number))
            
            for idx, lp in enumerate(license_plate_number):
                # print(idx)

                text = lp["text"]
                index = lp["index"]
                bbox = lp["bbox"]
                # print(text, index)
                # class_ID_index = str(json_data_ID) + "_" + str(index) + "_" + str(idx) + "_" + str(text) + ".jpg"
                # print(class_ID_index, bbox)    
                
                if license_plate == index:
                    print("bbox %s :\t" %index, bbox)
                    xx = bbox[0]
                    yy = bbox[1]
                    ww = bbox[2]
                    hh = bbox[3]
                    ll.append(int(xx)-1)
                    tt.append(int(yy)-1)
                    rr.append(int(xx + ww)+1)
                    bb.append(int(yy + hh)+1)
                else:                
                                
                    ll = []
                    tt = []
                    rr = []
                    bb = []
                    print("bbox %s :\t" %index, bbox)
                    xx = bbox[0]
                    yy = bbox[1]
                    ww = bbox[2]
                    hh = bbox[3]
                    ll.append(int(xx)-1)
                    tt.append(int(yy)-1)
                    rr.append(int(xx + ww)+1)
                    bb.append(int(yy + hh)+1)
                    license_plate += 1


                llm = min(ll)
                ttm = min(tt)
                rrm = max(rr)
                bbm = max(bb)
                # class_ID_index = str(json_data_ID) + "_" + str(index) + "_" + str(text) + ".jpg"
                class_ID_index = str(json_data_ID) + "_" + str(index) + ".jpg"
                print("Final :\t", llm, ttm, rrm, bbm)
                
                try:
                    image_path = images_path + "/" + json_data_ID + ".jpg"
                    # image_path = findfile(json_data_ID + ".jpg", images_path)
                    # print(image_path, [ll,tt,rr,bb])
                    # img = Image.open(image_path)
                    # img = pyvips.Image.new_from_image(image_path)
                    img = cv2.imread(image_path)
                    # crop_img = img.crop((ll,tt,rr,bb))
                    crop_img = img[ttm: bbm, llm: rrm]
                    crop_save_path = output_path + "crop_imagesss/" + class_ID_index
                    print("Saved Path \t\t:\t", crop_save_path)
                    # crop_img.save(crop_save_path)
                    cv2.imwrite(crop_save_path, crop_img, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
                except:
                    print("No img")
                
                # label = class_ID_index + "\t" + text + "\n"
                # labels.append(label)

            
            # crop & save labels path
            
                
        # print(labels)
        # with open(output_path+'labels.txt','a') as f:
        #     f.writelines(labels)
    
    except:
        print("JSON file ERROR")
    

    
        
if __name__ == '__main__':
    json_path = "/media/jhan/새 볼륨/55_LP/C-220721_09_CR12_03_N1919.json"
    labels_path = "/media/jhan/새 볼륨/55_LP/labels_test"
    output_path = "/media/jhan/새 볼륨/55_LP/"
    
    json2img_crop_one_word(json_path)
    
    # import glob
    # for pathAndFilename in glob.iglob(os.path.join(labels_path, "*.json")):
    #     # print(pathAndFilename)
    #     json2img_crop(json_path)
