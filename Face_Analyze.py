import cv2
import numpy as np
from deepface import DeepFace
def face_estimation(img):
    obj = DeepFace.analyze(img_path = img, actions = ['age', 'gender'])
    c_img = cv2.imread(img)
    x1 = obj['region']['x']
    y1 = obj['region']['y']
    x2 = obj['region']['x'] + obj['region']['w']
    y2 = obj['region']['y'] + obj['region']['h']
    c_img = cv2.rectangle(c_img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow('Image',c_img)
    cv2.waitKey()
    cv2.imwrite('./FaceResult.jpg',c_img)
    cv2.destroyAllWindows()
    print(obj)
img = './My_img/me.png'    
face_estimation(img=img)
# print(DeepFace.build_model('Age'))