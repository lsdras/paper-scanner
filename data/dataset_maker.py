import cv2
import numpy as np
import csv
import os

X_RESIZE = 672
Y_RESIZE = 672

os.mkdir("./note")
os.mkdir("./note/images")
os.mkdir("./note/masks")
os.mkdir("./note/test")

######## TRAIN IMAGES ##########

f = open('./label/GT_edge_clockwise_from_left_top.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
iter =0
ERRORS = []
for line in rdr:
    try:
        iter +=1
        if line[0] == 'PASS': break

        name = line[0]
        if name[-4:]=='.JPG':
            name.replace('JPG','jpg')
        elif name[-4:] != '.jpg':
            name = name + '.jpg'
        print(name)

        original = cv2.imread('./label/'+name)
        mask = np.zeros_like(original)
        pts = np.array(line[1:],np.int32)
        edges = pts.reshape(-1,2)#lt,rt,rb,lb clockwise pixel(x,y)
        mask = cv2.fillPoly(mask, [edges], (255,255,255))
        mask = mask[:,:,0]

        #COMPARE
        #numpy_horizontal_concat = np.concatenate((original, img), axis=1)
        #COMBINE&OVERLAP
        #dst = cv2.addWeighted(original,0.3,img,0.7,0)
        #cv2.imwrite("./GT/"+name,dst)
        #RESIZE
        [y,x] = original.shape[0:2]
        x_ratio = X_RESIZE/x
        y_ratio = Y_RESIZE/y
        original = cv2.resize(original, None, fx=x_ratio, fy=y_ratio, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, None, fx=x_ratio, fy=y_ratio, interpolation=cv2.INTER_AREA)

        #os.mkdir("../GT/"+str(iter))
        #os.mkdir("../GT/"+str(iter)+"/image")
        #os.mkdir("../GT/"+str(iter)+"/mask")

        #cv2.imwrite("../GT/"+str(iter)+"/image/"+str(iter)+".png",original)
        #cv2.imwrite("../GT/"+str(iter)+"/mask/M_"+str(iter)+".png",mask)
        name = name.split('.')[0]
        cv2.imwrite("./note/images/"+name+".png",original)
        cv2.imwrite("./note/masks/"+name+".png",mask)
    
    except:
        print(name+"______ERROR!!!!!")
        ERRORS.append(name)

######## TEST IMAGES ##########

files = os.listdir('./no_label/')
iter = 0
for name in files:
    try:
        iter +=1
        original = cv2.imread(os.path.join('no_label',name))
        print(original.shape)

        #COMPARE
        #numpy_horizontal_concat = np.concatenate((original, img), axis=1)
        #COMBINE&OVERLAP
        #dst = cv2.addWeighted(original,0.3,img,0.7,0)
        #cv2.imwrite("./GT/"+name,dst)
        #RESIZE
        [y,x] = original.shape[0:2]
        x_ratio = X_RESIZE/x
        y_ratio = Y_RESIZE/y
        original = cv2.resize(original, None, fx=x_ratio, fy=y_ratio, interpolation=cv2.INTER_AREA)

        #os.mkdir("./stage1_test/"+str(iter))
        #os.mkdir("./stage1_test/"+str(iter)+"/image")
        #cv2.imwrite("./stage1_test/"+str(iter)+"/image/"+str(iter)+".png",original)
        name = name.split('.')[0]
        cv2.imwrite("./note/test/"+name+".png",original)

    except: print(name+"______ERROR!!!!!")
        
print(ERRORS)