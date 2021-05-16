import numpy as np
import cv2
import time

#from edgetpu.detection.engine import DetectionEngine
from PIL import Image
import PIL

videos ="matching_system/tests/videos/example_02.mp4"
kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

time_start = time.time()
#engine = DetectionEngine("mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")


def get_persons(og_frame):
      

    persons = []
    frame = og_frame.copy()        
    ans = detect_persons(frame)
       
    if ans:
        for obj in ans:
            if obj.label_id == 0:
                score = obj.score * 100
               

      

                  
                persons.append(1)
    

    return len(persons)>0

def detect_persons(bgr_frame):
       
    image = Image.fromarray(bgr_frame)
    

        # Run Inference
        # threshold - float defining the minimun confidence threshold
        # top_k - int defining maximum number of objects to detect
        # relative_coord - bool defining if returns float coords or int
    results = engine.DetectWithImage(image, threshold=0.3, keep_aspect_ratio=True,  #PRESO AQUI
                                                relative_coord=False, top_k=10, resample=PIL.Image.BICUBIC)
        
    return results

def frame_difference(video,write_path):
    cap_obj = cv2.VideoCapture(video)
   
    # video information
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(write_path,fourcc,15,(640,480))
    _,previous_frame=cap_obj.read()
    _,current_frame = cap_obj.read()

    time_passed = time.time() - time_start
    
    
    #while True and time_passed<240:
    while True and time_passed<259200:
        
        time_passed = time.time() - time_start

        if current_frame is None:
            break
        difference_between_frames = cv2.absdiff(previous_frame,current_frame)
        gray_frame = cv2.cvtColor(difference_between_frames, cv2.COLOR_BGR2GRAY)
        th_frame = cv2.threshold(gray_frame, 50, 255, cv2.THRESH_BINARY)[1]
        dilated = cv2.dilate(th_frame, kernel)
        _, contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        if get_persons(current_frame):
#            print("Saving Frame")
#            out.write(current_frame)
        for c in contours:
            if cv2.contourArea(c)> 50:
                out.write(current_frame)
                break
        
#        cv2.imshow("Background image ", current_frame.astype('uint8'))
        #cv2.imwrite(write_path, background_img)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
        
    
        previous_frame=current_frame
        _,current_frame=cap_obj.read()


    cap_obj.release()
    out.release()
    cv2.destroyAllWindows()

frame_difference(0,"gravado.avi")