#inspired by https://arxiv.org/pdf/1702.07836.pdf, https://arxiv.org/pdf/1807.07428v1.pdf  making basic model
#make pseudo training image which disguise real image and it's mask

import os
import cv2
import numpy as np
import random

class IMG_COMBIN:
    def __init__(self, height, width, frontimg_dir,backimg_dir, backimg_ratio):
        self.height = height
        self.width = width
        self.points = self.make_random_4points(height,width)
        self.front_img = cv2.imread(frontimg_dir,0)
        self.back_img = cv2.imread(backimg_dir)
        self.anchor_ratio = backimg_ratio

    def background_anchors(self, ratio):
        height, width, _  = self.back_img.shape
        col_board = np.random.randint(0,height*(1-1/ratio))
        row_board = np.random.randint(0,0.75*height/ratio)
        height = int(height/ratio)
        width = int(height*0.75)
        return col_board, row_board, height, width

    def mk_background(self):
        bg = self.back_img
        anchor = self.background_anchors
        col_board, row_board, height, width = anchor(self.anchor_ratio)
        img_slice =  bg[col_board:col_board+height,row_board:row_board+width]
        return img_slice

    def resize_foreground(self):
        cols,rows = self.front_img.shape
        pts1 = np.array([[0,0],[0,cols],[rows,0],[rows,cols]], np.float32)
        pts2 = self.points

        M = cv2.getPerspectiveTransform(pts1,pts2)
        print(pts1,"pts1")
        print(pts2,"pts2")
        #dst = cv2.warpPerspective(original_img,M,(self.height,self.width))
        dst = cv2.warpPerspective(self.front_img,M,(3000,4000))
        return dst


    def make_random_4points(self,height,width):
        # it will change to perspective projection at some day.
        col_ratio = height/self.height
        row_ratio = width/self.width
        x1 = np.random.randint(0.1*width,0.2*width)/row_ratio
        y1 = np.random.randint(0.05*height,0.15*height)/col_ratio
        x2 = np.random.randint(0.8*width,0.9*width)/row_ratio
        y2 = np.random.randint(0.05*height,0.15*height)/col_ratio
        x3 = np.random.randint(0.8*width,0.9*width)/row_ratio
        y3 = np.random.randint(0.85*height,0.95*height)/col_ratio
        x4 = np.random.randint(0.1*width,0.2*width)/row_ratio
        y4 = np.random.randint(0.85*height,0.95*height)/col_ratio
        four_points = np.array([[x1,y1],[x4,y4],[x2,y2],[x3,y3]],np.float32)

        #four_points = np.array([[0, 0], [0, height], [width, 0], [width, height]], np.float32)

        print(four_points,"FP")
        return four_points#, FP_for_mask#[4,2]ndarray

    def gen_gt_mask(self,height,width):
        edges = self.points
        #points = np.array([edges[0,1],edges[0,0],edges[2,1],edges[2,0],edges[3,1],edges[3,0],edges[1,1],edges[1,0]],"F")
        #points = np.array([edges[0,0],edges[0,1],edges[2,0],edges[2,1],edges[3,0],edges[3,1],edges[1,0],edges[1,1]])
        points = np.array([edges[0,0],edges[0,1],edges[2,0],edges[2,1],edges[3,0],edges[3,1],edges[1,0],edges[1,1]])
        edges = np.int64(points.reshape(-1,2))
        mask = np.zeros((height,width,3))
        gt_mask = cv2.fillPoly(mask,[edges],(255,255,255))
        print(edges,"mask")
        return gt_mask

    def combin_back_front(self):
        #combined_img = cv2.addWeighted(self.mk_background,1,self.resize_foreground(self.front_img),1)
        back = self.mk_background()
        back = cv2.resize(back,(3000,4000),interpolation=cv2.INTER_LINEAR)
        front = self.resize_foreground()
        roi = back[0:front.shape[0],0:front.shape[1]]
        print(roi)
        _ , mask = cv2.threshold(front, 10, 255, cv2.THRESH_BINARY)
        back_bg = cv2.bitwise_and(roi,roi,mask = cv2.bitwise_not(mask))
        front_fg = cv2.bitwise_and(front,front,mask = mask)
        print(back_bg.shape,"back")
        front_fg = cv2.cvtColor(front_fg,cv2.COLOR_GRAY2RGB)
        print(front_fg.shape,"front")
        dst = back_bg+front_fg
        back[0:front.shape[0], 0:front.shape[1]] = dst
        combin = cv2.GaussianBlur(back,(3,3),1)

        i= str(imgnum)
        '''submit_dir = os.path.join("gen_img/", i)
        os.makedirs(submit_dir)
        os.makedirs(os.path.join(submit_dir, "mask"))
        os.makedirs(os.path.join(submit_dir, "image"))
        img_name = i + ".png"
        gt_dir = os.path.join(submit_dir,"mask",img_name)
        img_dir = os.path.join(submit_dir,"image",img_name)
        cv2.imwrite(img_dir, combin)
        cv2.imwrite(gt_dir, mask)'''
        submit_dir = "gen_img/"
        img_name = i + ".png"
        img_dir = os.path.join(submit_dir,img_name)
        cv2.imwrite(img_dir, combin)
        return 0



if __name__ == '__main__':
    global imgnum
    imgnum=1
    for backimg in os.listdir("background/"):
        for ratio in [1.5,2,3]:
            Img_combin = IMG_COMBIN(4000,3000,"../../private_info/original_note.jpg", os.path.join("background/",backimg), ratio)
            Img_combin.combin_back_front()
            imgnum=imgnum+1

