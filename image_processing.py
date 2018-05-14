from PIL import Image 
import cv2 
import numpy as np 
class images ():
    def  __init__(self, image_path, **kwargs):
        self.image_path=image_path
        self.im = Image.open(self.image_path)
        self.format = kwargs.pop('format', "RGB")
        self.color = kwargs.pop('color', "Balck")
   
    def Size(self):
        self.width, self.height =self.im.size
        self.target_size=(self.width, self.height)
        return(self.target_size, self.width, self.height)
    def video_allframes(self, output_path, **kwargs):
        self.output_path=output_path
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame indexZ
        vidcapture = cv2.VideoCapture(self.image_path)
        c= 0
        while vidcapture.isOpened():
            success, image = vidcapture.read()
            if success:
                if kwargs['framing_method']==1:
                    vidcapture.set(cv2.CAP_PROP_POS_MSEC,c*1000)
                    print ("single frame")
                cv2.imwrite(os.path.join(self.output_path, '%d.png') % c, image)
                c += 1
            else:
                break
        cv2.destroyAllWindows()
        vidcapture.release()
    def IOU(self,bb,gt):
        self.bb=bb
        self.gt=gt
        for items in self.bb : 
            float(items)
        for it in self.gt: 
            float(it)
        x1=max(self.bb[0],self.gt[0])
        y1=max(self.bb[1],self.gt[1])
        x2=min(self.bb[2],self.gt[2])
        y2=min(self.bb[3],self.gt[3])
        w=x2-x1
        h=y2-y1
        area=w*h
        area_bb=(self.bb[2]-self.bb[0])*(self.bb[3]-self.bb[1])
        area_gt=(self.gt[2]-self.gt[0])*(self.gt[3]-self.gt[1])
        combined_area = area_bb+area_gt-area
        iou=area/combined_area
        
        return iou
       
    def display(self): 
        i1=cv2.imread(self.image_path)
        x=tuple(map(int,tuple(self.gt[:2])))
        y=tuple(map(int,tuple(self.gt[2:])))
        z=tuple(map(int,tuple(self.bb[:2])))
        w=tuple(map(int,tuple(self.bb[2:])))
        cv2.rectangle(i1,x,y,(0,255,0),3)  #3:thicknes of rectangle
        cv2.rectangle(i1,z,w,(0,0,255),3)
        I=self.IOU(self.bb,self.gt)
        cv2.putText(i1, "IoU: {:.4f}".format(I), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print("{}: {:.4f}".format(i1,I))
        cv2.imshow("image",i1)
        cv2.waitKey(0)
        