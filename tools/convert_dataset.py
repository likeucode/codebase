import cv2
import os 

imgdata_src = "/home/ke/Faces"
imgdata_des = "/home/ke/face_data"

sub_dir = os.listdir(imgdata_src)

for sub in sub_dir:
    subdir_path=os.path.join(imgdata_src,sub)
    img_files=os.listdir(subdir_path)
    des_sub=os.path.join(imgdata_des,sub)
    os.mkdir(des_sub)
    for img in img_files:
        img_path=os.path.join(subdir_path,img)
        print "processing "+img_path
        img_data=cv2.imread(img_path)

        img_resized=cv2.resize(img_data,(256,256))
        cv2.imwrite(des_sub+"/"+img,img_resized)

