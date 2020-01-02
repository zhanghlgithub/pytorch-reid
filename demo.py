from detection import cam_detection
from reid import cam_reid
import cv2
import time

detect_model = cam_detection.detection_model()
reid_mode = cam_reid.reid_model()

# encode origin image
compare = cam_reid.Compare(model=reid_mode, origin_img="/home/hlzhang/project/detection_reid/image/origin")
origin_f, origin_name = compare.encode_origin_image()

def handle_signal_image(img):
    
#    # encode origin image
#    compare = cam_reid.Compare(model=reid_mode, origin_img=origin_img)
#    origin_f, origin_name = compare.encode_origin_image()
    
    # person detection
    img = cv2.resize(img, (640, 480))
    idxs, classes, anchors = cam_detection.detection_person(img, detect_model)
    
    bounding_boxs = []
    for j in range(idxs[0].shape[0]):
        bbox = anchors[idxs[0][j], :]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        
        if(classes[idxs[0][j]] == 0):
            person = img[y1:y2, x1:x2, :]
            
            # person reid
            identify_name, score = compare.run(person, origin_f, origin_name)
            
            if(identify_name in ["Zhang HL1", "Zhang HL2", "Zhang HL3", "Zhang HL4"]):
                identify_name = "Zhang HL"
            elif(identify_name in ["Yang JM1", "Yang JM2"]):
                identify_name = "Yang JM"
                
            print("identify name:{}, score:{}".format(identify_name, round(1-score, 2)))
            
            bounding_boxs.append([(x1,y1,x2,y2), identify_name+' '+str(round(1-score, 2))])
            #img = cam_detection.draw_rectangle(img, (x1,y1,x2,y2), identify_name+'  '+str(round((1-score), 2)))
            
    for box in bounding_boxs:
        print(box)
        img = cam_detection.draw_rectangle(img, box[0], box[1])
        
            #cv2.imwrite(str(j)+".jpg", img)
    return img
            
#img = cv2.imread("Zhang HL4.jpg")
#handle_signal_image(img)    


def handle_usb_cam():
    
    cap = cv2.VideoCapture(0)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    fource = cv2.VideoWriter_fourcc('m','p','4','v')
    out = cv2.VideoWriter("out7.avi", fource, fps, (640,480))
    
    count = 0
    while(True):
        
        st_time = time.time()
        ret, frame = cap.read()
        
        #if(count % 2 == 0):
        if 1:
            count = 0
            try:          
                frame = handle_signal_image(frame)
            except:
                continue
        
        print("cost time:", time.time()-st_time)
                
        frame = cv2.resize(frame, (640, 480))
        out.write(frame)
        
        cv2.imshow("output", frame)
        if(cv2.waitKey(1) & 0xFF==ord("q")):
            break
        
        count += 1
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()    
    

if __name__=="__main__":
    handle_usb_cam()
    