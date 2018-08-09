import cv2
import numpy as np
import random

class IMG_COMBIN:
    def __init__(self, height, width, frontimg_dir):
        self.height = height
        self.width = width
        self.points = self.make_random_4points(height,width)
        self.front_img = cv2.imread(frontimg_dir)

    def mk_background(self,sigma=15,mu=128):
        height = self.height
        width = self.width
        bg = sigma*np.random.randn(height*width*3)+mu
        bg = bg.astype(int)
        background = []
        for i in bg:
            if i<0:
                i=0
            if i>255:
                i=255
            background.append(i)
        background = np.asarray(background)
        background.resize((height,width,3))
        return background

    def resize_foreground(self):
        rows,cols,_ = self.front_img.shape
        pts1 = np.array([[0,0],[rows,0],[0,cols],[rows,cols]], np.float32)
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
        y1 = np.random.randint(0,0.125*width)/row_ratio
        x1 = np.random.randint(0,0.125*height)/col_ratio
        y2 = np.random.randint(0.875*width,width)/row_ratio
        x2 = np.random.randint(0,0.125*height)/col_ratio
        y3 = np.random.randint(0.875*width,width)/row_ratio
        x3 = np.random.randint(0.875*height,height)/col_ratio
        y4 = np.random.randint(0,0.125*width)/row_ratio
        x4 = np.random.randint(0.875*height,height)/col_ratio
        four_points = np.array([[x1,y1],[x4,y4],[x2,y2],[x3,y3]],np.float32)
        print(four_points,"FP")
        return four_points#[4,2]ndarray

    def gen_gt_mask(self,height,width):
        edges = self.points
        points = np.array([edges[0,1],edges[0,0],edges[2,1],edges[2,0],edges[3,1],edges[3,0],edges[1,1],edges[1,0]],"F")
        edges = np.int64(points.reshape(-1,2))
        mask = np.zeros((height,width,3))
        gt_mask = cv2.fillPoly(mask,[edges],(255,255,255))
        print([[edges[0],edges[1],edges[3],edges[2]]],"mask")
        return gt_mask

    def combin_back_front(self):
        #combined_img = cv2.addWeighted(self.mk_background,1,self.resize_foreground(self.front_img),1)
        back = self.mk_background()
        front = self.resize_foreground()
        roi = back[0:front.shape[0],0:front.shape[1]]
        print(roi)
        img2gray = cv2.cvtColor(front,cv2.COLOR_BGR2GRAY)
        _ , mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        back_bg = cv2.bitwise_and(roi,roi,mask = cv2.bitwise_not(mask))
        front_fg = cv2.bitwise_and(front,front,mask = mask)
        print(back_bg.shape,"back")
        print(front_fg.shape,"front")
        dst = back_bg+front_fg
        back[0:front.shape[0], 0:front.shape[1]] = dst
        return back



if __name__ == '__main__':

    Img_combin = IMG_COMBIN(4000,3000,"../../private_info/original_note.jpg")
    #cv2.imwrite("gt.png",Img_combin.gen_gt_mask(4000,3000))
    cv2.imwrite("front.png",Img_combin.resize_foreground())
    cv2.imwrite("combine.png",Img_combin.combin_back_front())
