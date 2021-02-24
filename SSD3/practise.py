'''
f=open("/home/ubuntu/MyFiles/YOLO/Gaussian_YOLOv3/valid_kitti_3cls_list.txt")
new_file="/home/ubuntu/MyFiles/YOLO/Gaussian_YOLOv3/valid_kitti_3cls_list_new.txt"
lines=f.readlines()
for line in lines:
    line.strip("\n")
    temp=line[-11:-4]
    new_temp="/home/ubuntu/MyFiles/VOCdevkit/VOC2012/JPEGImages_sparse/"+temp+"png"
    with open(new_file,"a") as f1:
        f1.write(new_temp)
        f1.write("\n")
f1.close()
print("finished")
'''
import torch
a=
