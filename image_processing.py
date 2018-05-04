
# coding: utf-8

def video_allframes(input_path, output_path):
    # extract frames from a video and save to directory as 'x.png' where 
    # x is the frame indexZ
    vidcapture = cv2.VideoCapture(input_path)
    c= 0
    while vidcapture.isOpened():
        success, image = vidcapture.read()
        if success:
            cv2.imwrite(os.path.join(output_path, '%d.png') % c, image)
            c += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcapture.release()

    
######

    
def video_1frame(input_path, output_path):
    # extract frames from a video and save to directory as 'x.jpg' where 
    # x is the frame indexZ
    vidcapture = cv2.VideoCapture(input_path) 
    c= 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            vidcapture.set(cv2.CAP_PROP_POS_MSEC,c*1000)
            cv2.imwrite(os.path.join(output_path, '%d.JPG') % c, image)
            c += 1
            
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()

    
######   
    

def IOU(bb,gt):
    x1=max(bb[0],gt[0])
    y1=max(bb[1],gt[1])
    x2=min(bb[2],gt[2])
    y2=min(bb[3],gt[3])
    w=x2-x1
    h=y2-y1
    area=w*h
    area_bb=(bb[2]-bb[0])*(bb[3]-bb[1])
    area_gt=(gt[2]-gt[0])*(gt[3]-gt[1])
    combined_area = area_bb+area_gt-area
    iou=area/combined_area
    
    return iou
###############


def display(exemple,detector):
    for detector in exemple: 
        image=cv2.imread(detector.image)
        cv2.rectangle(image,tuple(detector.gt[:2]),tuple(detector.gt[2:]),(0,255,0),3)  #3:thicknes of rectangle
        cv2.rectangle(image,tuple(detector.prediction[:2]),tuple(detector.prediction[2:]),(0,0,255),3)
        I=IOU(detector.prediction,detector.gt) 
        cv2.imshow("image",image)
        cv2.waitKey(0)
############



