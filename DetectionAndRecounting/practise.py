'''
dict_pre=[[1,2,3],[2,4,6]]
dict_pre = sorted(dict_pre, key=(lambda x: float(x[0])), reverse=True)
print(dict_pre)

import cv2 as cv
import os
path = "F:\\tiny_dataset\\"
i =0
for file in os.listdir(path):
    img = cv.imread(path + file)
    img = cv.resize(img,(500,333))
    img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    cv.imwrite("F:/tiny_deal/"+str(i)+".jpg",img)
    print(str(i)+"finished")
    i+=1
'''
print(sum([]))