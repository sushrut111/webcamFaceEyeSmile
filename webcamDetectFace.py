import numpy as np
import cv2
from numpy.lib.twodim_base import eye

def open_webcam_and_get_video():
    captureStream = cv2.VideoCapture(0)
    return captureStream

def close_webcam(captureStream):
    captureStream.release()

def close_everything(captureStream):
    cv2.destroyAllWindows()
    close_webcam(captureStream)
    

class CascadeModels:
    cascdes_dict = {
        "frontalface": "./haarcascades/haarcascade_frontalface_default.xml",
        "smile": "./haarcascades/haarcascade_smile.xml",
        "eyes": "./haarcascades/haarcascade_eye.xml"
    }
    def __init__(self):
        self.cascdes = {}
        for cascade_name, path in self.cascdes_dict.items():
            self.cascdes[cascade_name] = cv2.CascadeClassifier(path)
    
    def detectAttributesInImage(self, image, classifier_name, scaleFactor=1.1, minNeighbors=5):
        detected = self.cascdes[classifier_name].detectMultiScale(
            image,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
        )
        return detected
    
def run_face_detector():
    models = CascadeModels()
    capture_stream = open_webcam_and_get_video()
    while True:
        _, captured_image = capture_stream.read()
        gray = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)

        # frontalface = models.detectAttributesInImage(gray, "frontalface")
        # for x, y, w, h in frontalface:
        #     cv2.rectangle(captured_image, (x,y), (x+w, y+h), (0,255,0), 2)

        # eyes = models.detectAttributesInImage(gray, "eyes")
        # for x, y, w, h in eyes:
        #     cv2.rectangle(captured_image, (x,y), (x+w, y+h), (0,255,255), 2)

        smile = models.detectAttributesInImage(gray, "smile", scaleFactor=2, minNeighbors=50)
        for x, y, w, h in smile:
            cv2.rectangle(captured_image, (x,y), (x+w, y+h), (255,0,0), 2)

        if len(smile) > 0:
            cv2.putText(captured_image, "You have got a fantastic smile", (50,500), cv2.FONT_ITALIC, 1, (255,0,255), 4)        


        cv2.putText(captured_image, "Presss q to exit", (50,50), cv2.FONT_ITALIC, 1, (0,0,255), 2)        
        cv2.imshow("Video Capture", captured_image)

        key = cv2.waitKey(1)
        if key == ord('q'):
            close_everything(capture_stream)

run_face_detector()

